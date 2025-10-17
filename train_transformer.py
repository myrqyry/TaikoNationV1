import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
import yaml
import argparse
import json

from transformer_dataset import get_transformer_data_loaders
from transformer_model import TaikoTransformer
from tokenization import TaikoTokenizer

def load_config(path="config/default.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def train_fold(config, fold_idx):
    run_name = f"fold_{fold_idx + 1}"
    wandb.init(project="TaikoNation-Genre-Conditioning", config=config, name=run_name, reinit=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting Fold {fold_idx + 1} on {device} ---")

    train_loader, val_loader, tokenizer, genre_vocab = get_transformer_data_loaders(config, fold_idx)
    if not train_loader:
        wandb.finish()
        return

    model = TaikoTransformer(
        vocab_size=tokenizer.vocab_size,
        num_genres=len(genre_vocab),
        **config['model']
    ).to(device)
    wandb.watch(model, log="all")

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.vocab["[PAD]"])
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', **config['training']['scheduler'])

    best_val_loss = float('inf')
    model_save_path = f"{config['training']['save_path']}_fold_{fold_idx + 1}.pth"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    for epoch in range(config['training']['num_epochs']):
        model.train()
        for batch in train_loader:
            if not batch: continue
            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            target = batch["target"].to(device)
            genre_id = batch["genre_id"].to(device)

            optimizer.zero_grad()
            output = model(encoder_input, decoder_input, genre_id)
            loss = criterion(output.view(-1, tokenizer.vocab_size), target.view(-1))
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if not batch: continue
                encoder_input = batch["encoder_input"].to(device)
                decoder_input = batch["decoder_input"].to(device)
                target = batch["target"].to(device)
                genre_id = batch["genre_id"].to(device)

                output = model(encoder_input, decoder_input, genre_id)
                loss = criterion(output.view(-1, tokenizer.vocab_size), target.view(-1))
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        wandb.log({"val_loss": val_loss, "epoch": epoch})
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Epoch {epoch+1}: New best model saved with val_loss: {best_val_loss:.4f}")

    print("--- Finished Supervised Training ---")

    genre_vocab_path = os.path.join(os.path.dirname(model_save_path), "genre_vocab.json")
    with open(genre_vocab_path, 'w') as f:
        json.dump(genre_vocab, f)
    print(f"Genre vocabulary saved to {genre_vocab_path}")

    wandb.finish()

def main(config):
    os.environ["WANDB_MODE"] = "offline"
    num_folds = 1 if config.get('dry_run') else config['training']['k_folds']
    for i in range(num_folds):
        train_fold(config, i)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Taiko Transformer model with genre conditioning.")
    parser.add_argument('--config', type=str, default='config/default.yaml', help='Path to config file.')
    args = parser.parse_args()
    config = load_config(args.config)
    main(config)