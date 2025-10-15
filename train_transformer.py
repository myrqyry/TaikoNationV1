import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
import yaml
import argparse

from transformer_dataset import get_transformer_data_loaders
from transformer_model import MultiTaskTaikoTransformer # Use the new multi-task model
from tokenization import PatternAwareTaikoTokenizer
import torch.nn.functional as F

def sliding_window_loss(outputs, targets, window_size=4, pad_token_id=0):
    """
    Calculates a pattern-aware loss by focusing on the coherence of local
    sequences (windows). It encourages the model to predict consistent patterns.

    Args:
        outputs (torch.Tensor): The model's output logits.
                                Shape: [batch_size, seq_len, vocab_size]
        targets (torch.Tensor): The ground truth token IDs.
                                Shape: [batch_size, seq_len]
        window_size (int): The size of the sliding window.
        pad_token_id (int): The ID of the padding token, to be ignored.

    Returns:
        torch.Tensor: The calculated sliding window loss.
    """
    batch_size, seq_len, vocab_size = outputs.shape
    total_loss = 0
    num_windows = 0

    # Slide a window across the sequence
    for i in range(seq_len - window_size + 1):
        window_outputs = outputs[:, i:i+window_size, :]
        window_targets = targets[:, i:i+window_size]

        # --- Pattern Consistency Loss within the window ---
        # We use KL-Divergence to measure if the predicted probability
        # distribution for a token is similar to the distributions of
        # other tokens within the same window. This encourages the model
        # to learn a consistent "style" or "pattern" for local segments.

        # Reshape for easier processing
        window_outputs = window_outputs.reshape(batch_size * window_size, vocab_size)
        window_targets = window_targets.reshape(batch_size * window_size)

        # Ignore padding tokens in the loss calculation
        non_pad_mask = (window_targets != pad_token_id)
        if non_pad_mask.sum() == 0:
            continue # Skip windows that are all padding

        # Apply log-softmax to get log probabilities
        log_probs = F.log_softmax(window_outputs[non_pad_mask], dim=-1)

        # The "pattern" for a window can be represented by the average
        # probability distribution of its non-padding tokens.
        mean_log_probs = log_probs.mean(dim=0, keepdim=True)

        # The loss is the KL-divergence between each token's predicted
        # distribution and the window's average distribution. A lower
        # divergence means the predictions are more consistent.
        # We use the log-target version for stability (target is the mean).
        kl_div = F.kl_div(log_probs, mean_log_probs.expand_as(log_probs),
                          reduction='batchmean', log_target=True)

        total_loss += kl_div
        num_windows += 1

    return total_loss / num_windows if num_windows > 0 else torch.tensor(0.0, device=outputs.device)

def load_config(path="config/default.yaml"):
    """Loads the YAML configuration file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def train_fold(config, fold_idx):
    """
    Main training loop for a single fold of cross-validation.
    """
    # --- Setup ---
    run_name = f"fold_{fold_idx + 1}"
    wandb.init(project="TaikoNation-Transformer", config=config, name=run_name, reinit=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting Fold {fold_idx + 1}/{config['training']['k_folds']} on {device} ---")

    # --- Data ---
    train_loader, val_loader, tokenizer = get_transformer_data_loaders(config, fold_idx)
    if train_loader is None:
        print("Failed to create data loaders. Skipping fold.")
        wandb.finish()
        return

    vocab_size = tokenizer.vocab_size
    pad_token_id = tokenizer.vocab["[PAD]"]

    # --- Model ---
    # Use the new MultiTaskTaikoTransformer
    model = MultiTaskTaikoTransformer(
        vocab_size=vocab_size,
        num_difficulty_classes=config['training']['multi_task']['num_difficulty_classes'],
        **config['model']
    ).to(device)
    wandb.watch(model, log="all")

    # --- Training Components ---
    # Loss for the main token prediction task
    token_criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    # Loss for the tempo regression task
    tempo_criterion = nn.MSELoss()
    # Loss for the difficulty classification task
    difficulty_criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', **config['training']['scheduler']
    )

    # --- Training Loop ---
    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stopping_patience = config['training']['early_stopping']['patience']
    min_delta = config['training']['early_stopping']['min_delta']

    # Get loss weights from config
    pattern_cfg = config['training']['pattern_loss']
    mt_cfg = config['training']['multi_task']
    p_weight = pattern_cfg['loss_weight']
    tempo_weight = mt_cfg['tempo_loss_weight']
    diff_weight = mt_cfg['difficulty_loss_weight']

    model_save_path = f"{config['training']['save_path']}_fold_{fold_idx + 1}.pth"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    for epoch in range(config['training']['num_epochs']):
        model.train()
        # Initialize running losses for all components
        losses = {'total': 0.0, 'token': 0.0, 'pattern': 0.0, 'tempo': 0.0, 'difficulty': 0.0}

        for i, batch in enumerate(train_loader):
            if batch is None: continue

            # Move all parts of the batch to the device
            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            token_target = batch["target"].to(device)
            difficulty_target = batch["difficulty_target"].to(device)
            tempo_target = batch["tempo_target"].to(device)

            optimizer.zero_grad()
            predictions = model(src=encoder_input, tgt=decoder_input)

            # --- Calculate Combined Multi-Task Loss ---
            token_loss = token_criterion(predictions['tokens'].view(-1, vocab_size), token_target.view(-1))
            pattern_loss = sliding_window_loss(
                predictions['tokens'], token_target,
                window_size=pattern_cfg['window_size'],
                pad_token_id=pad_token_id
            )
            tempo_loss = tempo_criterion(predictions['tempo'], tempo_target)
            difficulty_loss = difficulty_criterion(predictions['difficulty'], difficulty_target)

            total_loss = (token_loss +
                          p_weight * pattern_loss +
                          tempo_weight * tempo_loss +
                          diff_weight * difficulty_loss)

            total_loss.backward()
            optimizer.step()

            # --- Logging ---
            losses['total'] += total_loss.item()
            losses['token'] += token_loss.item()
            losses['pattern'] += pattern_loss.item()
            losses['tempo'] += tempo_loss.item()
            losses['difficulty'] += difficulty_loss.item()


            if i % 50 == 49:
                # Average losses over the logging interval
                for key in losses: losses[key] /= 50
                print(f"[Fold {fold_idx+1}, E {epoch+1}, B {i+1}] Loss: {losses['total']:.4f} "
                      f"(Tok: {losses['token']:.4f}, Pat: {losses['pattern']:.4f}, "
                      f"Tem: {losses['tempo']:.4f}, Diff: {losses['difficulty']:.4f})")
                wandb.log({
                    "train_loss_total": losses['total'],
                    "train_loss_token": losses['token'],
                    "train_loss_pattern": losses['pattern'],
                    "train_loss_tempo": losses['tempo'],
                    "train_loss_difficulty": losses['difficulty'],
                    "epoch": epoch
                })
                # Reset running losses
                losses = {k: 0.0 for k in losses}


        # --- Validation ---
        model.eval()
        val_losses = {'total': 0.0, 'token': 0.0, 'pattern': 0.0, 'tempo': 0.0, 'difficulty': 0.0}
        with torch.no_grad():
            for batch in val_loader:
                if batch is None: continue

                encoder_input = batch["encoder_input"].to(device)
                decoder_input = batch["decoder_input"].to(device)
                token_target = batch["target"].to(device)
                difficulty_target = batch["difficulty_target"].to(device)
                tempo_target = batch["tempo_target"].to(device)

                predictions = model(src=encoder_input, tgt=decoder_input)

                # --- Calculate Combined Validation Loss ---
                token_loss = token_criterion(predictions['tokens'].view(-1, vocab_size), token_target.view(-1))
                pattern_loss = sliding_window_loss(
                    predictions['tokens'], token_target,
                    window_size=pattern_cfg['window_size'],
                    pad_token_id=pad_token_id
                )
                tempo_loss = tempo_criterion(predictions['tempo'], tempo_target)
                difficulty_loss = difficulty_criterion(predictions['difficulty'], difficulty_target)

                total_loss = (token_loss +
                              p_weight * pattern_loss +
                              tempo_weight * tempo_loss +
                              diff_weight * difficulty_loss)

                val_losses['total'] += total_loss.item()
                val_losses['token'] += token_loss.item()
                val_losses['pattern'] += pattern_loss.item()
                val_losses['tempo'] += tempo_loss.item()
                val_losses['difficulty'] += difficulty_loss.item()

        # Average the losses over the number of validation batches
        for key in val_losses: val_losses[key] /= len(val_loader)

        print(f"[Fold {fold_idx+1}, E {epoch+1}] Val Loss: {val_losses['total']:.4f} "
              f"(Tok: {val_losses['token']:.4f}, Pat: {val_losses['pattern']:.4f}, "
              f"Tem: {val_losses['tempo']:.4f}, Diff: {val_losses['difficulty']:.4f})")
        wandb.log({
            "val_loss_total": val_losses['total'],
            "val_loss_token": val_losses['token'],
            "val_loss_pattern": val_losses['pattern'],
            "val_loss_tempo": val_losses['tempo'],
            "val_loss_difficulty": val_losses['difficulty'],
            "epoch": epoch,
            "lr": optimizer.param_groups[0]['lr']
        })

        # Use the total validation loss for scheduling and early stopping
        val_loss = val_losses['total']

        scheduler.step(val_loss)

        # --- Save Best Model ---
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model for fold {fold_idx+1} saved with val_loss: {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stopping_patience:
            print(f"Early stopping triggered for fold {fold_idx+1}.")
            break

    print(f"--- Finished Fold {fold_idx + 1} ---")
    wandb.finish()

def main(config):
    os.environ["WANDB_MODE"] = "offline"

    # For a dry run, just test one fold
    num_folds = 1 if config.get('dry_run', False) else config['training']['k_folds']

    for i in range(num_folds):
        train_fold(config, i)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Taiko Transformer model.")
    parser.add_argument('--config', type=str, default='config/default.yaml',
                        help='Path to the configuration file.')
    args = parser.parse_args()

    config = load_config(args.config)

    try:
        main(config)
    except Exception as e:
        print(f"An error occurred during training: {e}")
        if wandb.run:
            wandb.finish()