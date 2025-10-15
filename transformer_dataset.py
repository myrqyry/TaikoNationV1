import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from tokenization import TaikoTokenizer
from audio_processing import get_audio_features

# --- Constants ---
INPUT_CHART_DIR = "input_charts_nr"
INPUT_SONG_DIR = "input_songs"
TEST_DATA_INDICES = [2, 5, 9, 82, 28, 22, 81, 43, 96, 97]

class TaikoTransformerDataset(Dataset):
    """
    A PyTorch Dataset for the Taiko Transformer model.
    It loads song-chart pairs, processes them into aligned audio features and
    token sequences, and prepares them for training a sequence-to-sequence model.
    """
    def __init__(self, is_train=True, max_sequence_length=512, time_quantization_ms=100, source_resolution_ms=23.2):
        self.is_train = is_train
        self.max_sequence_length = max_sequence_length
        self.tokenizer = TaikoTokenizer(time_quantization=time_quantization_ms, source_resolution=source_resolution_ms)
        self.samples = self._prepare_sample_map()

    def _prepare_sample_map(self):
        """Creates a list of all valid song-chart pairs."""
        samples = []
        try:
            charts = sorted(os.listdir(path=INPUT_CHART_DIR))
            songs = os.listdir(path=INPUT_SONG_DIR)
        except FileNotFoundError as e:
            print(f"Error: Input directory not found - {e}.")
            return []

        song_map = {s.split()[0]: s for s in songs}

        for i, chart_filename in enumerate(charts):
            is_for_this_set = (i not in TEST_DATA_INDICES) if self.is_train else (i in TEST_DATA_INDICES)
            if not is_for_this_set:
                continue

            try:
                id_number = chart_filename.split("_")[0]
                if id_number in song_map:
                    samples.append({
                        "chart_path": os.path.join(INPUT_CHART_DIR, chart_filename),
                        "song_path": os.path.join(INPUT_SONG_DIR, song_map[id_number])
                    })
            except IndexError:
                continue

        print(f"Found {len(samples)} samples for {'training' if self.is_train else 'testing'}.")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 1. Load and process audio features
        audio_features = get_audio_features(sample["song_path"])
        if audio_features is None:
            # Return dummy data if audio processing fails
            return self._get_dummy_item()

        # 2. Load and tokenize chart data
        token_ids = self.tokenizer.tokenize(sample["chart_path"])
        if not token_ids:
            return self._get_dummy_item()

        # 3. Align and pad/truncate sequences
        min_len = min(len(audio_features), len(token_ids))

        audio_features = audio_features[:min_len]
        token_ids = token_ids[:min_len]

        # Pad or truncate to max_sequence_length
        # Audio features padding: 0
        # Token padding: [PAD] token ID
        pad_token_id = self.tokenizer.vocab["[PAD]"]

        # Pad audio
        audio_padding_length = self.max_sequence_length - audio_features.shape[0]
        if audio_padding_length > 0:
            padding_array = np.zeros((audio_padding_length, audio_features.shape[1]))
            audio_features = np.vstack([audio_features, padding_array])
        else:
            audio_features = audio_features[:self.max_sequence_length]

        # Pad tokens
        token_padding_length = self.max_sequence_length - len(token_ids)
        if token_padding_length > 0:
            token_ids.extend([pad_token_id] * token_padding_length)
        else:
            token_ids = token_ids[:self.max_sequence_length]

        # 4. Prepare inputs for the transformer
        # Encoder input is the audio features
        encoder_input = torch.from_numpy(audio_features).float()

        # Decoder input is the token sequence, shifted right (starts with [CLS])
        cls_token_id = self.tokenizer.vocab["[CLS]"]
        decoder_input = torch.tensor([cls_token_id] + token_ids[:-1], dtype=torch.long)

        # Target is the original token sequence
        target = torch.tensor(token_ids, dtype=torch.long)

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "target": target
        }

    def _get_dummy_item(self):
        """Returns an empty/dummy item to be filtered out by the collate_fn."""
        return None


def collate_fn(batch):
    """Custom collate function to filter out None (failed) samples."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


def get_transformer_data_loaders(batch_size=16, max_sequence_length=512, time_quantization_ms=100, source_resolution_ms=23.2):
    """Returns training and testing DataLoaders for the transformer."""
    train_dataset = TaikoTransformerDataset(
        is_train=True,
        max_sequence_length=max_sequence_length,
        time_quantization_ms=time_quantization_ms,
        source_resolution_ms=source_resolution_ms
    )
    test_dataset = TaikoTransformerDataset(
        is_train=False,
        max_sequence_length=max_sequence_length,
        time_quantization_ms=time_quantization_ms,
        source_resolution_ms=source_resolution_ms
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn
    )
    # Return the tokenizer from one of the datasets (they are identical)
    return train_loader, test_loader, train_dataset.tokenizer
