import os
import torch
import numpy as np
import re
import json
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

from .tokenization import TaikoTokenizer
from .audio_processing import get_audio_features, augment_spectrogram

# --- Constants ---
INPUT_CHART_DIR = "input_charts_nr"
INPUT_SONG_DIR = "input_songs"
DIFFICULTY_MAP = {"easy": 0, "normal": 1, "hard": 2, "oni": 3, "ura": 4, "unknown": 1}

class TaikoTransformerDataset(Dataset):
    def __init__(self, all_samples, indices, tokenizer, genre_vocab, is_train=True, max_sequence_length=512, time_quantization_ms=100, source_resolution_ms=23.2, cache_dir='./cache', prefetch=True):
        self.is_train = is_train
        self.max_sequence_length = max_sequence_length
        self.tokenizer = tokenizer
        self.genre_vocab = genre_vocab
        self.samples = [all_samples[i] for i in indices]
        self.source_resolution_ms = source_resolution_ms
        self.time_quantization_ms = time_quantization_ms
        self.cache_dir = cache_dir
        self.prefetch = prefetch
        os.makedirs(cache_dir, exist_ok=True)
        self._preprocess_audio_async()

    def _preprocess_audio_async(self):
        """Asynchronously pre-compute spectrograms with progress bar"""
        print(f"Preprocessing audio features for {len(self.samples)} samples...")

        def process_sample(idx, sample):
            cache_path = os.path.join(self.cache_dir, f"spec_{idx}.npy")
            if not os.path.exists(cache_path):
                audio_features = get_audio_features(
                    sample["song_path"],
                    source_resolution_ms=self.source_resolution_ms,
                    frame_duration_ms=self.time_quantization_ms
                )
                if audio_features is not None:
                    temp_path = cache_path + '.tmp'
                    np.save(temp_path, audio_features)
                    os.replace(temp_path, cache_path)
                    return True
            return False

        num_workers = min(mp.cpu_count(), 8)
        processed_count = 0
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(process_sample, idx, sample): idx
                for idx, sample in enumerate(self.samples)
            }
            from tqdm import tqdm
            for future in tqdm(as_completed(futures), total=len(futures), desc="Preprocessing"):
                if future.result():
                    processed_count += 1

        print(f"Processed {processed_count} new audio files, {len(self.samples) - processed_count} from cache")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.prefetch and not hasattr(self, '_cache'):
            self._cache = {}

        cache_path = os.path.join(self.cache_dir, f"spec_{idx}.npy")

        if self.prefetch and idx in self._cache:
            audio_features = self._cache[idx]
        else:
            if not os.path.exists(cache_path):
                return None
            try:
                audio_features = np.load(cache_path)
                if self.prefetch:
                    self._cache[idx] = audio_features
            except Exception as e:
                print(f"Warning: Could not load {cache_path}: {e}")
                return None

        if audio_features.ndim != 2 or audio_features.shape[0] == 0:
            print(f"Warning: Corrupt audio features in {cache_path}, shape is {audio_features.shape}")
            return None

        if self.is_train:
            audio_features = augment_spectrogram(audio_features)

        try:
            token_ids = self.tokenizer.tokenize(sample["chart_path"])
        except Exception as e:
            print(f"Warning: Could not tokenize chart {sample['chart_path']}: {e}")
            return None
        if not token_ids:
            print(f"Warning: Tokenization resulted in empty sequence for {sample['chart_path']}")
            return None

        min_len = min(len(audio_features), len(token_ids))
        if min_len == 0:
            return None

        audio_features = audio_features[:min_len]
        token_ids = token_ids[:min_len]

        if audio_features.shape[0] < self.max_sequence_length:
            pad_width = self.max_sequence_length - audio_features.shape[0]
            padding = np.zeros((pad_width, audio_features.shape[1]))
            audio_features = np.vstack([audio_features, padding])
        else:
            audio_features = audio_features[:self.max_sequence_length]

        if len(token_ids) < self.max_sequence_length:
            token_ids.extend([self.tokenizer.vocab["[PAD]"]] * (self.max_sequence_length - len(token_ids)))
        else:
            token_ids = token_ids[:self.max_sequence_length]

        try:
            encoder_input = torch.from_numpy(audio_features).float()
            decoder_input = torch.tensor([self.tokenizer.vocab["[CLS]"]] + token_ids[:-1], dtype=torch.long)
            target = torch.tensor(token_ids, dtype=torch.long)

            difficulty_label = torch.tensor(DIFFICULTY_MAP.get(sample.get("difficulty", "unknown").lower(), 1), dtype=torch.long)
            genre_id = torch.tensor(self.genre_vocab.get(sample.get("genre", "unknown"), 0), dtype=torch.long)

            if encoder_input.shape[0] != self.max_sequence_length or decoder_input.shape[0] != self.max_sequence_length or target.shape[0] != self.max_sequence_length:
                print(f"Warning: Mismatched sequence lengths for sample {idx}. Skipping.")
                return None
        except Exception as e:
            print(f"Warning: Error creating tensors for sample {idx}: {e}")
            return None

        return {"encoder_input": encoder_input, "decoder_input": decoder_input, "target": target, "difficulty": difficulty_label, "genre_id": genre_id}

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch: return None
    return torch.utils.data.dataloader.default_collate(batch)

def get_transformer_data_loaders(config, fold_idx=0):
    genre_labels_path = "output/genre_labels.json"
    if os.path.exists(genre_labels_path):
        with open(genre_labels_path, 'r') as f:
            genre_labels = json.load(f)
    else:
        genre_labels = {}
        print("Warning: genre_labels.json not found. Genres will be 'unknown'.")

    all_samples, genres = [], set(["unknown"])
    try:
        charts = sorted(os.listdir(INPUT_CHART_DIR))
        songs = os.listdir(INPUT_SONG_DIR)
        song_map = {s.split()[0]: s for s in songs}

        for chart_filename in charts:
            try:
                id_number = chart_filename.split("_")[0]
                song_filename = song_map.get(id_number)
                if song_filename:
                    genre = genre_labels.get(song_filename, "unknown")
                    genres.add(genre)

                    difficulty_match = re.search(r'\[(.*?)\]', chart_filename)
                    difficulty = difficulty_match.group(1) if difficulty_match else "unknown"

                    all_samples.append({
                        "chart_path": os.path.join(INPUT_CHART_DIR, chart_filename),
                        "song_path": os.path.join(INPUT_SONG_DIR, song_filename),
                        "genre": genre,
                        "difficulty": difficulty
                    })
            except IndexError:
                continue
    except FileNotFoundError as e:
        print(f"Error: Input directory not found - {e}.")
        return None, None, None, None

    genre_vocab = {name: i for i, name in enumerate(sorted(list(genres)))}
    print(f"Found {len(all_samples)} samples and {len(genre_vocab)} genres.")

    kf = KFold(n_splits=config['training']['k_folds'], shuffle=True, random_state=42)
    train_indices, val_indices = list(kf.split(all_samples))[fold_idx]

    tokenizer = TaikoTokenizer()
    data_config = config['data']
    train_dataset = TaikoTransformerDataset(all_samples, train_indices, tokenizer, genre_vocab, is_train=True, max_sequence_length=data_config['max_sequence_length'], time_quantization_ms=data_config['time_quantization_ms'], source_resolution_ms=data_config['source_resolution_ms'])
    val_dataset = TaikoTransformerDataset(all_samples, val_indices, tokenizer, genre_vocab, is_train=False, max_sequence_length=data_config['max_sequence_length'], time_quantization_ms=data_config['time_quantization_ms'], source_resolution_ms=data_config['source_resolution_ms'])

    num_workers = min(mp.cpu_count(), config['training'].get('num_workers', 4))
    prefetch_factor = config['training'].get('prefetch_factor', 2)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=num_workers // 2,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        persistent_workers=True
    )

    return train_loader, val_loader, tokenizer, genre_vocab