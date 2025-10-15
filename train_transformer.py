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
import json
import numpy as np

def sliding_window_loss(outputs, targets, window_size=4, pad_token_id=0, pattern_weights=None):
    """
    Calculates a weighted, pattern-aware loss focusing on local sequence coherence.
    Rarer or more important patterns can be weighted more heavily.
    """
    if pattern_weights is None:
        pattern_weights = {}

    batch_size, seq_len, vocab_size = outputs.shape
    total_loss = 0
    total_weight = 0

    # Slide a window across the sequence
    for i in range(seq_len - window_size + 1):
        window_outputs = outputs[:, i:i+window_size, :]
        window_targets = targets[:, i:i+window_size]

        # --- Get Pattern Weight ---
        # Convert the target window to a tuple to use as a dictionary key
        pattern_tuple = tuple(window_targets[0].tolist()) # Use first item in batch as representative
        # Default weight is 1.0 if pattern not in dict
        weight = pattern_weights.get(str(pattern_tuple), 1.0)

        # --- Cross-Entropy Loss for the window ---
        # This directly measures the correctness of the predictions against the targets.
        flat_outputs = window_outputs.reshape(-1, vocab_size)
        flat_targets = window_targets.reshape(-1)

        # We use a standard cross-entropy loss for the window
        criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id, reduction='none')
        window_loss = criterion(flat_outputs, flat_targets)

        # Reshape back to (batch_size, window_size) and take the mean
        window_loss = window_loss.view(batch_size, -1).mean()

        # Apply the weight to the calculated loss for this window
        total_loss += weight * window_loss
        total_weight += weight

    # Normalize by the sum of weights used
    return total_loss / total_weight if total_weight > 0 else torch.tensor(0.0, device=outputs.device)

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

    # --- Load Pattern Frequencies for Weighted Loss ---
    pattern_loss_config = config['training']['pattern_loss']
    pattern_weights = None
    if pattern_loss_config.get('pattern_frequency_file'):
        try:
            with open(pattern_loss_config['pattern_frequency_file'], 'r') as f:
                freqs = json.load(f)

            # Use the n-grams corresponding to the window size
            n_gram_key = f"{pattern_loss_config['window_size']}-grams"
            if n_gram_key in freqs:
                pattern_counts = freqs[n_gram_key]
                total_patterns = sum(pattern_counts.values())

                # Calculate inverse frequency weights
                pattern_weights = {
                    p: (total_patterns / count)
                    for p, count in pattern_counts.items()
                }

                # Normalize weights
                max_weight = max(pattern_weights.values())
                pattern_weights = {p: w / max_weight for p, w in pattern_weights.items()}

                # Apply intensity factor
                intensity = pattern_loss_config.get('weight_intensity', 0.5)
                pattern_weights = {
                    p: (1 - intensity) + (intensity * w)
                    for p, w in pattern_weights.items()
                }
                print(f"Loaded and processed {len(pattern_weights)} pattern weights.")

        except FileNotFoundError:
            print(f"Warning: Pattern frequency file not found at {pattern_loss_config['pattern_frequency_file']}. Proceeding without weighted loss.")
        except Exception as e:
            print(f"Warning: Error loading or processing pattern weights: {e}. Proceeding without weighted loss.")

    # --- Training Loop ---
    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stopping_patience = config['training']['early_stopping']['patience']
    min_delta = config['training']['early_stopping']['min_delta']
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
                pad_token_id=pad_token_id,
                pattern_weights=pattern_weights
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
                    pad_token_id=pad_token_id,
                    pattern_weights=pattern_weights
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