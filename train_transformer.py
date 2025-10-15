import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os

from transformer_dataset import get_transformer_data_loaders, TaikoTransformerDataset
from transformer_model import TaikoTransformer

# --- Configuration ---
LEARNING_RATE = 1e-4
BATCH_SIZE = 8 # Transformers are memory-intensive, so a smaller batch size is safer
NUM_EPOCHS = 25
MAX_SEQUENCE_LENGTH = 512
MODEL_SAVE_PATH = "output/taiko_transformer.pth"
WANDB_PROJECT = "TaikoNation-Transformer"

def train():
    """
    Main training loop for the TaikoNation Transformer model.
    """
    # --- Setup ---
    os.environ["WANDB_MODE"] = "offline"
    wandb.init(project=WANDB_PROJECT, config={
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epochs": NUM_EPOCHS,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "architecture": "TaikoTransformer_v1"
    })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data ---
    print("Loading data...")
    train_loader, test_loader = get_transformer_data_loaders(
        batch_size=BATCH_SIZE,
        max_sequence_length=MAX_SEQUENCE_LENGTH
    )

    # Need vocab_size for model and loss function
    # A bit of a hack to get it from the dataset instance
    temp_dataset = TaikoTransformerDataset()
    vocab_size = temp_dataset.tokenizer.vocab_size
    pad_token_id = temp_dataset.tokenizer.vocab["[PAD]"]
    del temp_dataset

    # --- Model ---
    model = TaikoTransformer(vocab_size=vocab_size).to(device)
    wandb.watch(model, log="all")

    # --- Training Components ---
    # We use CrossEntropyLoss, but we must ignore the padding token in the loss calculation.
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    # --- Training Loop ---
    print("Starting training...")
    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for i, batch in enumerate(train_loader):
            if batch is None: continue

            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            target = batch["target"].to(device)

            optimizer.zero_grad()

            # Forward pass
            output = model(src=encoder_input, tgt=decoder_input)

            # Reshape for loss function: CrossEntropyLoss expects [N, C, ...]
            # Output: [batch_size, seq_len, vocab_size] -> [batch_size * seq_len, vocab_size]
            # Target: [batch_size, seq_len] -> [batch_size * seq_len]
            loss = criterion(output.view(-1, vocab_size), target.view(-1))

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 50 == 49: # Log every 50 batches
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
        wandb.log({"val_loss": val_loss, "epoch": epoch})

        # --- Save Best Model ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"New best model saved to {MODEL_SAVE_PATH} with validation loss: {best_val_loss:.4f}")

    print("Finished Training")
    wandb.finish()

if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print(f"An error occurred during training: {e}")
        if wandb.run:
            wandb.finish()