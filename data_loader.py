import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# --- Constants ---
# These should be kept in sync with the model and original data constants
SONG_CHUNK_SIZE = 16
NOTE_CHUNK_SIZE = 12
NOTE_VECTOR_SIZE = 7
SONG_VECTOR_SIZE = 80
INPUT_CHART_DIR = "input_charts_nr"
INPUT_SONG_DIR = "input_songs"
TEST_DATA_INDICES = [2, 5, 9, 82, 28, 22, 81, 43, 96, 97]

def _load_song_data(song_path):
    """Loads and reshapes the song data from a .npy file."""
    try:
        song_mm = np.load(song_path, mmap_mode="r")
        song_data = np.frombuffer(buffer=song_mm, dtype=np.float32, count=-1)
        song_data = song_data[0:song_mm.shape[0]*song_mm.shape[1]]
        return np.reshape(song_data, song_mm.shape)
    except Exception as e:
        print(f"An error occurred while loading song data at {song_path}: {e}")
        return None

def _load_note_data(chart_path, song_data_len):
    """Loads, reshapes, and pads the note data from a .npy file."""
    try:
        note_mm = np.load(chart_path, mmap_mode="r")
        note_data = np.frombuffer(note_mm, dtype=np.int32, count=-1)
        note_data = np.reshape(note_data, [len(note_mm), NOTE_VECTOR_SIZE])

        # Pad the note data
        diff = song_data_len - len(note_data)
        padding = np.zeros((diff + SONG_CHUNK_SIZE, NOTE_VECTOR_SIZE))
        return np.append(note_data, padding, axis=0)
    except Exception as e:
        print(f"An error occurred while loading note data at {chart_path}: {e}")
        return None

def _package_data_torch(song_data, note_data):
    """
    Packages the song and note data into input/output pairs for PyTorch.
    This is a reimplementation of the original _package_data function.
    """
    inputs, outputs = [], []

    for j in range(len(song_data)):
        song_input, note_input, output_chunk = [], [], []
        for k in range(SONG_CHUNK_SIZE):
            is_padding = j - k < 0

            # Song input
            song_vec = np.zeros([SONG_VECTOR_SIZE]) if is_padding else song_data[j - k]
            song_input.append(song_vec)

            # Note input
            if k < NOTE_CHUNK_SIZE:
                note_vec = np.zeros([NOTE_VECTOR_SIZE]) if is_padding else note_data[j - k]
                note_input.append(note_vec)
            elif k != 15:
                note_input.append(np.ones([NOTE_VECTOR_SIZE]))

            # Output chunk
            if k > 11:
                output_vec = np.zeros([NOTE_VECTOR_SIZE]) if is_padding else note_data[j - k]
                output_chunk.append(output_vec)

        # The input to the model is a flattened concatenation of song and note data
        input_chunk = np.concatenate([np.array(song_input).flatten(), np.array(note_input).flatten()])
        output_chunk = np.reshape(np.concatenate(output_chunk), [4, NOTE_VECTOR_SIZE])

        inputs.append(input_chunk)
        outputs.append(output_chunk)

    return np.array(inputs, dtype=np.float32), np.array(outputs, dtype=np.float32)


class TaikoDataset(Dataset):
    """
    PyTorch Dataset for TaikoNation.
    Uses lazy loading to be more memory-efficient. It prepares a map of
    data samples and only loads/processes one sample at a time in __getitem__.
    """
    def __init__(self, is_train=True):
        self.is_train = is_train
        self.samples = []
        self._prepare_samples()

    def _prepare_samples(self):
        """Creates a map of all song/chart pairs and their total length without loading data."""
        try:
            charts = os.listdir(path=INPUT_CHART_DIR)
            songs = os.listdir(path=INPUT_SONG_DIR)
        except FileNotFoundError as e:
            print(f"Error: Input directory not found - {e}. Ensure '{INPUT_CHART_DIR}' and '{INPUT_SONG_DIR}' exist.")
            return

        song_map = {s.split()[0]: s for s in songs}

        for i, chart_filename in enumerate(charts):
            is_for_this_set = (i not in TEST_DATA_INDICES) if self.is_train else (i in TEST_DATA_INDICES)
            if not is_for_this_set:
                continue

            try:
                id_number = chart_filename.split("_")[0]
                if id_number not in song_map:
                    continue
                song_filename = song_map[id_number]
            except IndexError:
                continue

            # We need to load the song just to know its length for indexing
            song_path = os.path.join(INPUT_SONG_DIR, song_filename)
            song_data = _load_song_data(song_path)
            if song_data is None or len(song_data) == 0:
                continue

            num_chunks = len(song_data)
            for j in range(num_chunks):
                self.samples.append({
                    "chart_path": os.path.join(INPUT_CHART_DIR, chart_filename),
                    "song_path": song_path,
                    "chunk_index": j,
                    "total_chunks": num_chunks
                })

        if not self.samples:
            print(f"Warning: No samples found for {'training' if self.is_train else 'testing'}.")
        else:
            print(f"Prepared {'training' if self.is_train else 'testing'} dataset with {len(self.samples)} total chunks.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]

        # Load the specific data needed for this one item
        song_data = _load_song_data(sample_info["song_path"])
        note_data = _load_note_data(sample_info["chart_path"], sample_info["total_chunks"])

        if song_data is None or note_data is None:
            # If data is bad, return a dummy tensor. This should be handled gracefully.
            return torch.zeros(1385), torch.zeros(4, 7)

        # Package just one chunk of data
        j = sample_info["chunk_index"]
        song_input, note_input, output_chunk_list = [], [], []

        for k in range(SONG_CHUNK_SIZE):
            is_padding = j - k < 0

            song_vec = np.zeros([SONG_VECTOR_SIZE]) if is_padding else song_data[j - k]
            song_input.append(song_vec)

            if k < NOTE_CHUNK_SIZE:
                note_vec = np.zeros([NOTE_VECTOR_SIZE]) if is_padding else note_data[j - k]
                note_input.append(note_vec)
            elif k != 15:
                note_input.append(np.ones([NOTE_VECTOR_SIZE]))

            if k > 11:
                output_vec = np.zeros([NOTE_VECTOR_SIZE]) if is_padding else note_data[j - k]
                output_chunk_list.append(output_vec)

        input_tensor = np.concatenate([np.array(song_input).flatten(), np.array(note_input).flatten()])
        output_tensor = np.reshape(np.concatenate(output_chunk_list), [4, NOTE_VECTOR_SIZE])

        return torch.from_numpy(input_tensor.astype(np.float32)), torch.from_numpy(output_tensor.astype(np.float32))


def get_data_loaders(batch_size=32):
    """Returns training and testing DataLoaders."""
    train_dataset = TaikoDataset(is_train=True)
    test_dataset = TaikoDataset(is_train=False)

    # num_workers=0 can be useful for debugging
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, test_loader

if __name__ == '__main__':
    # Example of how to use the data loaders
    print("Testing TaikoDataset and DataLoaders...")
    train_loader, test_loader = get_data_loaders(batch_size=4)

    print("\n--- Training Loader ---")
    for i, (inputs, outputs) in enumerate(train_loader):
        print(f"Batch {i+1}:")
        print("Input shape:", inputs.shape)   # Should be [batch_size, 1385]
        print("Output shape:", outputs.shape) # Should be [batch_size, 4, 7]
        if i == 0:
            break # Only show first batch

    print("\n--- Testing Loader ---")
    for i, (inputs, outputs) in enumerate(test_loader):
        print(f"Batch {i+1}:")
        print("Input shape:", inputs.shape)
        print("Output shape:", outputs.shape)
        if i == 0:
            break

    print("\nData loader test complete.")