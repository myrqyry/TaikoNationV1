import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
import yaml
import argparse

from transformer_dataset import get_transformer_data_loaders
from transformer_model import PatternAwareTransformer # Use the new model
from tokenization import TaikoTokenizer
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
    # Use the PatternAwareTransformer
    model = PatternAwareTransformer(
        vocab_size=vocab_size,
        **config['model']
    ).to(device)
    wandb.watch(model, log="all")

    # --- Training Components ---
    # Standard cross-entropy loss for token prediction
    ce_criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', **config['training']['scheduler']
    )

    # --- Training Loop ---
    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stopping_patience = config['training']['early_stopping']['patience']
    min_delta = config['training']['early_stopping']['min_delta']
    pattern_loss_config = config['training']['pattern_loss']
    window_size = pattern_loss_config['window_size']
    pattern_weight = pattern_loss_config['loss_weight']


    model_save_path = f"{config['training']['save_path']}_fold_{fold_idx + 1}.pth"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    for epoch in range(config['training']['num_epochs']):
        model.train()
        running_loss, running_ce_loss, running_p_loss = 0.0, 0.0, 0.0
        for i, batch in enumerate(train_loader):
            if batch is None: continue

            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            target = batch["target"].to(device)

            optimizer.zero_grad()
            output = model(src=encoder_input, tgt=decoder_input)

            # --- Calculate Combined Loss ---
            ce_loss = ce_criterion(output.view(-1, vocab_size), target.view(-1))
            pattern_loss = sliding_window_loss(
                output, target,
                window_size=window_size,
                pad_token_id=pad_token_id
            )
            total_loss = ce_loss + pattern_weight * pattern_loss

            total_loss.backward()
            optimizer.step()

            # --- Logging ---
            running_loss += total_loss.item()
            running_ce_loss += ce_loss.item()
            running_p_loss += pattern_loss.item()

            if i % 50 == 49:
                batch_loss = running_loss / 50
                batch_ce_loss = running_ce_loss / 50
                batch_p_loss = running_p_loss / 50
                print(f"[Fold {fold_idx+1}, Epoch {epoch+1}, Batch {i+1}] total_loss: {batch_loss:.4f} (CE: {batch_ce_loss:.4f}, Pattern: {batch_p_loss:.4f})")
                wandb.log({
                    "train_loss": batch_loss,
                    "train_ce_loss": batch_ce_loss,
                    "train_pattern_loss": batch_p_loss,
                    "epoch": epoch
                })
                running_loss, running_ce_loss, running_p_loss = 0.0, 0.0, 0.0


        # --- Validation ---
        model.eval()
        val_loss, val_ce_loss, val_p_loss = 0.0, 0.0, 0.0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None: continue

                encoder_input = batch["encoder_input"].to(device)
                decoder_input = batch["decoder_input"].to(device)
                target = batch["target"].to(device)

                output = model(src=encoder_input, tgt=decoder_input)

                # --- Calculate Combined Validation Loss ---
                ce_loss = ce_criterion(output.view(-1, vocab_size), target.view(-1))
                pattern_loss = sliding_window_loss(
                    output, target,
                    window_size=window_size,
                    pad_token_id=pad_token_id
                )
                total_loss = ce_loss + pattern_weight * pattern_loss

                val_loss += total_loss.item()
                val_ce_loss += ce_loss.item()
                val_p_loss += pattern_loss.item()

        # Average the losses over the number of validation batches
        val_loss /= len(val_loader)
        val_ce_loss /= len(val_loader)
        val_p_loss /= len(val_loader)

        print(f"[Fold {fold_idx+1}, Epoch {epoch+1}] val_loss: {val_loss:.4f} (CE: {val_ce_loss:.4f}, Pattern: {val_p_loss:.4f})")
        wandb.log({
            "val_loss": val_loss,
            "val_ce_loss": val_ce_loss,
            "val_pattern_loss": val_p_loss,
            "epoch": epoch,
            "lr": optimizer.param_groups[0]['lr']
        })

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