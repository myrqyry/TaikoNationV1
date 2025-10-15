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

def train(config):
    """
    Main training loop for the TaikoNation Transformer model.
    """
    # --- Setup ---
    os.environ["WANDB_MODE"] = "offline"
    wandb.init(project="TaikoNation-Transformer", config=config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data ---
    print("Loading data...")
    # We pass the data config to the data loader function
    train_loader, test_loader, tokenizer = get_transformer_data_loaders(
        batch_size=config['training']['batch_size'],
        max_sequence_length=config['data']['max_sequence_length'],
        time_quantization_ms=config['data']['time_quantization_ms'],
        source_resolution_ms=config['data']['source_resolution_ms']
    )

    vocab_size = tokenizer.vocab_size
    pad_token_id = tokenizer.vocab["[PAD]"]

    # --- Model ---
    model = TaikoTransformer(
        vocab_size=vocab_size,
        d_model=config['model']['d_model'],
        nhead=config['model']['nhead'],
        num_encoder_layers=config['model']['num_encoder_layers'],
        num_decoder_layers=config['model']['num_decoder_layers'],
        dim_feedforward=config['model']['dim_feedforward'],
        dropout=config['model']['dropout'],
        audio_feature_size=config['model']['audio_feature_size']
    ).to(device)
    wandb.watch(model, log="all")

    # --- Training Components ---
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        'min',
        patience=config['training']['scheduler']['patience'],
        factor=config['training']['scheduler']['factor'],
        min_lr=config['training']['scheduler']['min_lr']
    )

    os.makedirs(os.path.dirname(config['training']['save_path']), exist_ok=True)

    # --- Training Loop ---
    print("Starting training...")
    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stopping_patience = config['training']['early_stopping']['patience']
    min_delta = config['training']['early_stopping']['min_delta']

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
                print(f"[Epoch {epoch + 1}, Batch {i + 1}] training loss: {batch_loss:.4f}")
                wandb.log({"train_loss": batch_loss, "epoch": epoch})
                running_loss = 0.0

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                if batch is None: continue

                encoder_input = batch["encoder_input"].to(device)
                decoder_input = batch["decoder_input"].to(device)
                target = batch["target"].to(device)

                output = model(src=encoder_input, tgt=decoder_input)
                loss = criterion(output.view(-1, vocab_size), target.view(-1))
                val_loss += loss.item()

        val_loss /= len(test_loader)
        print(f"Epoch {epoch + 1} validation loss: {val_loss:.4f}")
        wandb.log({"val_loss": val_loss, "epoch": epoch, "lr": optimizer.param_groups[0]['lr']})

        # Step the scheduler
        scheduler.step(val_loss)

        # --- Save Best Model ---
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), config['training']['save_path'])
            print(f"New best model saved to {config['training']['save_path']} with validation loss: {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation loss for {epochs_no_improve} epochs.")

        # --- Early Stopping Check ---
        if epochs_no_improve >= early_stopping_patience:
            print(f"Early stopping triggered after {epochs_no_improve} epochs with no improvement.")
            break

    print("Finished Training")
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Taiko Transformer model.")
    parser.add_argument('--config', type=str, default='config/default.yaml',
                        help='Path to the configuration file.')
    args = parser.parse_args()

    config = load_config(args.config)

    try:
        train(config)
    except Exception as e:
        print(f"An error occurred during training: {e}")
        if wandb.run:
            wandb.finish()