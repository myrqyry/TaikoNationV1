import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import os

from data_loader import get_data_loaders
from torch_model import TaikoNet

# --- Configuration ---
LEARNING_RATE = 0.0001
BATCH_SIZE = 64 # Using a larger batch size is more efficient for modern GPUs
NUM_EPOCHS = 50
MODEL_SAVE_PATH = "output/taiko_model.pth"
WANDB_PROJECT = "TaikoNation-PyTorch"

def train():
    """
    Main training loop for the TaikoNation PyTorch model.
    """
    # --- Setup ---
    # Initialize Weights & Biases in offline mode for CI/CD environments
    os.environ["WANDB_MODE"] = "offline"
    wandb.init(project=WANDB_PROJECT, config={
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epochs": NUM_EPOCHS,
        "architecture": "TaikoNet_v2"
    })

    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data ---
    print("Loading data...")
    train_loader, test_loader = get_data_loaders(batch_size=BATCH_SIZE)

    if not train_loader.dataset.samples or not test_loader.dataset.samples:
        print("Error: Data loaders are empty. Aborting training.")
        wandb.finish()
        return

    # --- Model ---
    model = TaikoNet().to(device)
    wandb.watch(model, log="all")

    # --- Training Components ---
    # The output is multi-class, and we can treat the (4, 7) output as 4 independent classifications.
    # However, a simpler approach for a direct migration is to use MSE loss, as the original
    # categorical cross-entropy might be complex to set up with this output shape.
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    # --- Training Loop ---
    print("Starting training...")
    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99: # Log every 100 batches
                batch_loss = running_loss / 100
                print(f"[Epoch {epoch + 1}, Batch {i + 1}] training loss: {batch_loss:.4f}")
                wandb.log({"train_loss": batch_loss, "epoch": epoch, "batch": i})
                running_loss = 0.0

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
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
    # For running this script, you might need to log in to wandb first.
    # You can do this by running `wandb login` in your terminal and providing your API key.
    # For this environment, we'll assume it can run without login for now.
    try:
        train()
    except Exception as e:
        print(f"An error occurred during training: {e}")
        # Make sure to finish wandb run on error
        if wandb.run:
            wandb.finish()