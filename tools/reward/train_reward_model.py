import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import os
import argparse

# Assuming reward_model.py is in the parent directory or accessible
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from reward_model import RewardModel

def train(args):
    """
    Trains the reward model on the exported human evaluation data.
    """
    # --- 1. Load Data ---
    try:
        df = pd.read_csv(args.data_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {args.data_path}")
        print("Please generate the data by running the human eval server and the export script first.")
        return

    # --- 2. Prepare Data (with placeholders for now) ---
    # In a real implementation, you would load chart features corresponding to each row.
    # For now, we'll create dummy features.
    num_samples = len(df)
    # This input_size must match the RewardModel's expectation
    input_size = args.input_size
    dummy_features = torch.randn(num_samples, input_size)

    # Use the 'fun' column as the target label
    labels = torch.tensor(df['fun'].values, dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(dummy_features, labels)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # --- 3. Train Model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}...")

    model = RewardModel(input_size=input_size, hidden_size=args.hidden_size).to(device)
    criterion = nn.MSELoss() # Mean Squared Error for regression
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        for features, targets in loader:
            features, targets = features.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(loader)
        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {epoch_loss:.4f}")

    # --- 4. Save Model ---
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(model.state_dict(), args.save_path)
    print(f"Trained reward model saved to {args.save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the reward model.")
    parser.add_argument('--data_path', type=str, default='output/reward_training_data.csv', help='Path to the training data CSV file.')
    parser.add_argument('--save_path', type=str, default='output/reward_model.pth', help='Path to save the trained model.')
    parser.add_argument('--input_size', type=int, default=256, help='Input feature size for the reward model.')
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden layer size for the reward model.')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size.')

    args = parser.parse_args()
    train(args)