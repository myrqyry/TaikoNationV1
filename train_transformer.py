import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
import yaml
import argparse

from transformer_dataset import get_transformer_data_loaders
from transformer_model import TaikoTransformer
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
    model = TaikoTransformer(
        vocab_size=vocab_size,
        **config['model'] # Unpack model hyperparameters
    ).to(device)
    wandb.watch(model, log="all")

    # --- Training Components ---
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', **config['training']['scheduler']
    )

    # --- Training Loop ---
    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stopping_patience = config['training']['early_stopping']['patience']
    min_delta = config['training']['early_stopping']['min_delta']

    model_save_path = f"{config['training']['save_path']}_fold_{fold_idx + 1}.pth"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    for epoch in range(config['training']['num_epochs']):
        model.train()
        running_loss = 0.0
        for i, batch in enumerate(train_loader):
            if batch is None: continue

            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            target = batch["target"].to(device)

            optimizer.zero_grad()
            output = model(src=encoder_input, tgt=decoder_input)
            loss = criterion(output.view(-1, vocab_size), target.view(-1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 50 == 49:
                batch_loss = running_loss / 50
                print(f"[Fold {fold_idx+1}, Epoch {epoch+1}, Batch {i+1}] loss: {batch_loss:.4f}")
                wandb.log({"train_loss": batch_loss, "epoch": epoch})
                running_loss = 0.0


        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None: continue

                encoder_input = batch["encoder_input"].to(device)
                decoder_input = batch["decoder_input"].to(device)
                target = batch["target"].to(device)

                output = model(src=encoder_input, tgt=decoder_input)
                loss = criterion(output.view(-1, vocab_size), target.view(-1))
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"[Fold {fold_idx+1}, Epoch {epoch+1}] val_loss: {val_loss:.4f}")
        wandb.log({"val_loss": val_loss, "epoch": epoch, "lr": optimizer.param_groups[0]['lr']})

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