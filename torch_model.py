import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Constants from the original model ---
SONG_CHUNK_SIZE = 16
NOTE_CHUNK_SIZE = 12
NOTE_VECTOR_SIZE = 7
SONG_VECTOR_SIZE = 80

# Model Architecture
CONV_FILTERS_1 = 16
CONV_FILTER_SIZE_1 = 3
CONV_FILTERS_2 = 32
CONV_FILTER_SIZE_2 = 3
FC_UNITS_1 = 128
LSTM_UNITS_1 = 64
LSTM_UNITS_2 = 64
OUTPUT_UNITS = 28

class TaikoNet(nn.Module):
    """
    A PyTorch implementation of the TaikoNation model, modernized for clarity and robustness.
    This version captures the spirit of the original (CNN for song, LSTM for sequence)
    but uses a standard, modern approach to combine features.
    """
    def __init__(self):
        super(TaikoNet, self).__init__()

        # --- Song Encoder Path (CNN) ---
        self.conv1 = nn.Conv1d(in_channels=SONG_VECTOR_SIZE, out_channels=CONV_FILTERS_1, kernel_size=CONV_FILTER_SIZE_1)
        self.dropout1 = nn.Dropout(0.2)
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=CONV_FILTERS_1, out_channels=CONV_FILTERS_2, kernel_size=CONV_FILTER_SIZE_2)
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)

        # Calculate the flattened size after convolutions and pooling
        # Input length: 16
        # After conv1 (len-ker+1): 16-3+1 = 14. After pool1: 7
        # After conv2 (len-ker+1): 7-3+1 = 5. After pool2: 2
        cnn_out_size = CONV_FILTERS_2 * 2
        self.fc1 = nn.Linear(in_features=cnn_out_size, out_features=FC_UNITS_1)

        # --- Feature Combination ---
        # We will combine the CNN output (128) and the flattened previous notes (15*7=105)
        combined_feature_size = FC_UNITS_1 + (15 * NOTE_VECTOR_SIZE)

        # This layer will project the combined features into the LSTM's input space
        self.fc_combine = nn.Linear(combined_feature_size, LSTM_UNITS_1)

        # --- Sequence Path (LSTM) ---
        self.lstm1 = nn.LSTM(input_size=LSTM_UNITS_1, hidden_size=LSTM_UNITS_1, batch_first=True)
        self.dropout_lstm1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(input_size=LSTM_UNITS_1, hidden_size=LSTM_UNITS_2, batch_first=True)
        self.dropout_lstm2 = nn.Dropout(0.2)

        # --- Output Layer ---
        self.fc_out = nn.Linear(in_features=LSTM_UNITS_2, out_features=OUTPUT_UNITS)

    def forward(self, x):
        # x has shape [batch, 1385]

        # --- Split Input ---
        song_features = x[:, :1280].view(-1, SONG_CHUNK_SIZE, SONG_VECTOR_SIZE)
        prev_notes = x[:, 1280:].view(-1, 15, NOTE_VECTOR_SIZE)

        # --- Song Encoder ---
        # Permute for Conv1d: [batch, length, channels] -> [batch, channels, length]
        song_features = song_features.permute(0, 2, 1)

        s = F.relu(self.conv1(song_features))
        s = self.dropout1(s)
        s = self.maxpool1(s)
        s = F.relu(self.conv2(s))
        s = self.maxpool2(s)

        s = torch.flatten(s, 1)
        s = F.relu(self.fc1(s)) # Song features processed by CNN -> [batch, 128]

        # --- Feature Combination ---
        notes_flat = torch.flatten(prev_notes, 1) # -> [batch, 105]
        combined = torch.cat((s, notes_flat), dim=1) # -> [batch, 233]

        # Project combined features into a space suitable for the LSTM
        lstm_input_features = F.relu(self.fc_combine(combined)) # -> [batch, 64]

        # --- LSTM Path ---
        # LSTM expects a sequence: [batch, seq_len, features]. We add a sequence dimension of 1.
        lstm_input = lstm_input_features.unsqueeze(1)

        lstm_out, _ = self.lstm1(lstm_input)
        lstm_out = self.dropout_lstm1(lstm_out)

        lstm_out, _ = self.lstm2(lstm_out)
        lstm_out = self.dropout_lstm2(lstm_out)

        # We take the output of the last time step
        last_lstm_out = lstm_out[:, -1, :]

        # --- Output Layer ---
        output = self.fc_out(last_lstm_out)

        # Reshape to final desired output: [batch, 4, 7]
        output = output.view(-1, 4, NOTE_VECTOR_SIZE)

        return output

if __name__ == '__main__':
    # Test the model with a dummy input
    print("Testing TaikoNet model...")
    # Batch size of 4, input size of 1385
    dummy_input = torch.randn(4, 1385)

    model = TaikoNet()
    print(model)

    # Forward pass
    try:
        output = model(dummy_input)
        print("\nForward pass successful!")
        print("Input shape:", dummy_input.shape)
        print("Output shape:", output.shape) # Should be [4, 4, 7]
    except Exception as e:
        print(f"\nAn error occurred during the forward pass: {e}")

    print("\nModel test complete.")