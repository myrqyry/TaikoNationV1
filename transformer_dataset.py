import os
import torch
import numpy as np
import re
from torch.utils.data import Dataset, DataLoader
import json

from tokenization import PatternLevelTokenizer
from audio_processing import get_audio_features, augment_spectrogram

# --- Constants ---
INPUT_CHART_DIR = "input_charts_nr"
INPUT_SONG_DIR = "input_songs"
TEST_DATA_INDICES = [2, 5, 9, 82, 28, 22, 81, 43, 96, 97]

from sklearn.model_selection import KFold

class TaikoTransformerDataset(Dataset):
    """
    A PyTorch Dataset for the Taiko Transformer model.
    It loads song-chart pairs, processes them into aligned audio features and
    token sequences, and prepares them for training a sequence-to-sequence model.
    """
    def __init__(self, all_samples, indices, tokenizer, is_train=True, max_sequence_length=512):
        self.is_train = is_train
        self.max_sequence_length = max_sequence_length
        self.tokenizer = tokenizer
        self.samples = [all_samples[i] for i in indices]
        # Get these from the tokenizer to ensure consistency
        self.source_resolution_ms = self.tokenizer.source_resolution
        self.time_quantization_ms = self.tokenizer.time_quantization

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 1. Load and process audio features
        audio_features = get_audio_features(
            sample["song_path"],
            source_resolution_ms=self.source_resolution_ms,
            frame_duration_ms=self.time_quantization_ms
        )
        if audio_features is None:
            # Return dummy data if audio processing fails
            return self._get_dummy_item()

        # Apply augmentation only to the training set
        if self.is_train:
            audio_features = augment_spectrogram(audio_features)

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


def get_transformer_data_loaders(config, fold_idx=0):
    """
    Creates and returns training and testing DataLoaders for a specific
    cross-validation fold, using the PatternLevelTokenizer.
    """
    # --- Load Top Patterns for Tokenizer ---
    top_patterns = []
    pattern_config = config.get('patterns', {})
    if pattern_config.get('frequency_file'):
        try:
            with open(pattern_config['frequency_file'], 'r') as f:
                freqs = json.load(f)

            # Extract top N patterns from each n-gram length
            for ngram_key, patterns in freqs.items():
                sorted_patterns = sorted(patterns.items(), key=lambda item: item[1], reverse=True)
                top_n = pattern_config.get('top_n_patterns_per_length', 10)
                # Convert string representation of tuple back to tuple
                top_patterns.extend([eval(p[0]) for p in sorted_patterns[:top_n]])

            print(f"Loaded {len(top_patterns)} top patterns for the tokenizer.")
        except FileNotFoundError:
            print(f"Warning: Pattern frequency file not found. Proceeding with base tokenizer.")
        except Exception as e:
            print(f"Warning: Error loading pattern frequencies: {e}. Proceeding with base tokenizer.")

    # 1. Prepare the full list of samples (with improved difficulty parsing)
    all_samples = []
    try:
        charts = sorted(os.listdir(path=INPUT_CHART_DIR))
        songs = os.listdir(path=INPUT_SONG_DIR)
        song_map = {s.split()[0]: s for s in songs}

        for chart_filename in charts:
            try:
                id_number = chart_filename.split("_")[0]
                if id_number in song_map:
                    basename = os.path.splitext(chart_filename)[0]
                    difficulty = "unknown"
                    bracket_match = re.search(r'\[([^\]]+)\]$', basename)
                    if bracket_match:
                        difficulty = bracket_match.group(1)
                    else:
                        last_underscore_pos = basename.rfind('_')
                        if last_underscore_pos != -1:
                            potential_difficulty = basename[last_underscore_pos + 1:]
                            if 0 < len(potential_difficulty) <= 40:
                                difficulty = potential_difficulty
                    all_samples.append({
                        "chart_path": os.path.join(INPUT_CHART_DIR, chart_filename),
                        "song_path": os.path.join(INPUT_SONG_DIR, song_map[id_number]),
                        "difficulty": difficulty
                    })
            except IndexError:
                continue
    except FileNotFoundError as e:
        print(f"Error: Input directory not found - {e}.")
        return None, None, None

    print(f"Found a total of {len(all_samples)} samples.")

    # 2. Create K-Fold splits
    kf = KFold(n_splits=config['training']['k_folds'], shuffle=True, random_state=42)
    all_splits = list(kf.split(all_samples))
    train_indices, val_indices = all_splits[fold_idx]

    print(f"Fold {fold_idx + 1}/{config['training']['k_folds']}: {len(train_indices)} train, {len(val_indices)} validation samples.")

    # 3. Create Datasets with the PatternLevelTokenizer
    data_config = config['data']

    # Create a single tokenizer instance to be shared
    tokenizer = PatternLevelTokenizer(
        top_patterns=top_patterns,
        time_quantization=data_config['time_quantization_ms'],
        source_resolution=data_config['source_resolution_ms']
    )

    train_dataset = TaikoTransformerDataset(
        all_samples,
        train_indices,
        tokenizer,
        is_train=True,
        max_sequence_length=data_config['max_sequence_length'],
    )
    val_dataset = TaikoTransformerDataset(
        all_samples,
        val_indices,
        tokenizer,
        is_train=False,
        max_sequence_length=data_config['max_sequence_length'],
    )

    # 4. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn
    )
    # Return the tokenizer from one of the datasets (they are identical)
    return train_loader, val_loader, train_dataset.tokenizer
