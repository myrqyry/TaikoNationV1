import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
import yaml
import argparse

from transformer_dataset import get_transformer_data_loaders
from transformer_model import MultiTaskTaikoTransformer
from tokenization import TaikoTokenizer

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
    model_config = config['model']
    model_config.pop('vocab_size', None) # Remove to avoid conflict
    model = MultiTaskTaikoTransformer(
        vocab_size=vocab_size,
        num_difficulty_classes=config['training']['multi_task']['num_difficulty_classes'],
        patterns_per_diff=config['training']['multi_task']['patterns_per_diff'],
        **model_config
    ).to(device)
    wandb.watch(model, log="all")

    # --- Training Components ---
    token_criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
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
    diff_weight = config['training']['multi_task']['difficulty_loss_weight']

    model_save_path = f"{config['training']['save_path']}_fold_{fold_idx + 1}.pth"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    for epoch in range(config['training']['num_epochs']):
        model.train()
        running_loss, running_token_loss, running_diff_loss = 0.0, 0.0, 0.0
        for i, batch in enumerate(train_loader):
            if batch is None: continue

            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            token_target = batch["target"].to(device)
            difficulty_target = batch["difficulty_target"].to(device)

            optimizer.zero_grad()
            predictions = model(src=encoder_input, tgt=decoder_input, target_difficulty=difficulty_target)

            token_loss = token_criterion(predictions['tokens'].view(-1, vocab_size), token_target.view(-1))
            difficulty_loss = difficulty_criterion(predictions['difficulty'], difficulty_target)

            total_loss = token_loss + diff_weight * difficulty_loss
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            running_token_loss += token_loss.item()
            running_diff_loss += difficulty_loss.item()

            if i % 50 == 49:
                batch_loss = running_loss / 50
                batch_token_loss = running_token_loss / 50
                batch_diff_loss = running_diff_loss / 50
                print(f"[Fold {fold_idx+1}, E {epoch+1}, B {i+1}] Loss: {batch_loss:.4f} (Tok: {batch_token_loss:.4f}, Diff: {batch_diff_loss:.4f})")
                wandb.log({
                    "train_loss_total": batch_loss,
                    "train_loss_token": batch_token_loss,
                    "train_loss_difficulty": batch_diff_loss,
                    "epoch": epoch
                })
                running_loss, running_token_loss, running_diff_loss = 0.0, 0.0, 0.0


        # --- Validation ---
        model.eval()
        val_loss, val_token_loss, val_diff_loss = 0.0, 0.0, 0.0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None: continue

                encoder_input = batch["encoder_input"].to(device)
                decoder_input = batch["decoder_input"].to(device)
                token_target = batch["target"].to(device)
                difficulty_target = batch["difficulty_target"].to(device)

                predictions = model(src=encoder_input, tgt=decoder_input)

                token_loss = token_criterion(predictions['tokens'].view(-1, vocab_size), token_target.view(-1))
                difficulty_loss = difficulty_criterion(predictions['difficulty'], difficulty_target)

                total_loss = token_loss + diff_weight * difficulty_loss

                val_loss += total_loss.item()
                val_token_loss += token_loss.item()
                val_diff_loss += difficulty_loss.item()

        val_loss /= len(val_loader)
        val_token_loss /= len(val_loader)
        val_diff_loss /= len(val_loader)

        print(f"[Fold {fold_idx+1}, E {epoch+1}] Val Loss: {val_loss:.4f} (Tok: {val_token_loss:.4f}, Diff: {val_diff_loss:.4f})")
        wandb.log({
            "val_loss_total": val_loss,
            "val_loss_token": val_token_loss,
            "val_loss_difficulty": val_diff_loss,
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