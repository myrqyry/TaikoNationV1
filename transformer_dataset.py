import os
import torch
import numpy as np
import re
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold

from tokenization import TaikoTokenizer
from audio_processing import get_audio_features, augment_spectrogram

# --- Constants ---
INPUT_CHART_DIR = "input_charts_nr"
INPUT_SONG_DIR = "input_songs"
OSU_CHART_DIR = "eval/evaluation dataset/human_taiko_set" # Corrected path

DIFFICULTY_MAP = {"easy": 0, "normal": 1, "hard": 2, "oni": 3, "ura": 4, "unknown": 1}

def parse_osu_file(osu_path):
    metadata = {}
    try:
        with open(osu_path, 'r', encoding='utf-8') as f:
            in_metadata_section = False
            for line in f:
                line = line.strip()
                if line == "[Metadata]":
                    in_metadata_section = True
                elif line.startswith("["):
                    in_metadata_section = False
                if in_metadata_section and ':' in line:
                    key, value = line.split(':', 1)
                    metadata[key.strip()] = value.strip()
    except Exception as e:
        print(f"Warning: Could not parse {osu_path}: {e}")
    return metadata

class TaikoTransformerDataset(Dataset):
    def __init__(self, all_samples, indices, tokenizer, mapper_vocab, is_train=True, max_sequence_length=512):
        self.is_train = is_train
        self.max_sequence_length = max_sequence_length
        self.tokenizer = tokenizer
        self.mapper_vocab = mapper_vocab
        self.samples = [all_samples[i] for i in indices]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        audio_features = get_audio_features(sample["song_path"])
        if audio_features is None: return None

        if self.is_train:
            audio_features = augment_spectrogram(audio_features)

        token_ids = self.tokenizer.tokenize(sample["chart_path"])
        if not token_ids: return None

        min_len = min(len(audio_features), len(token_ids))
        audio_features = audio_features[:min_len]
        token_ids = token_ids[:min_len]

        # Pad or truncate audio and tokens
        if audio_features.shape[0] < self.max_sequence_length:
            audio_padding = torch.zeros(self.max_sequence_length - audio_features.shape[0], audio_features.shape[1])
            audio_features = torch.cat([torch.from_numpy(audio_features), audio_padding], dim=0)
        else:
            audio_features = torch.from_numpy(audio_features[:self.max_sequence_length])

        if len(token_ids) < self.max_sequence_length:
            token_ids.extend([self.tokenizer.vocab["[PAD]"]] * (self.max_sequence_length - len(token_ids)))
        else:
            token_ids = token_ids[:self.max_sequence_length]

        encoder_input = audio_features
        decoder_input = torch.tensor([self.tokenizer.vocab["[CLS]"]] + token_ids[:-1], dtype=torch.long)
        target = torch.tensor(token_ids, dtype=torch.long)

        difficulty_label = torch.tensor(DIFFICULTY_MAP.get(sample["difficulty"], 1), dtype=torch.long)
        mapper_id = torch.tensor(self.mapper_vocab.get(sample.get("mapper", "unknown"), 0), dtype=torch.long)

        return {
            "encoder_input": encoder_input, "decoder_input": decoder_input, "target": target,
            "difficulty": difficulty_label, "mapper_id": mapper_id
        }

def get_transformer_data_loaders(config, fold_idx=0):
    osu_map = {}
    if os.path.exists(OSU_CHART_DIR):
        for filename in os.listdir(OSU_CHART_DIR):
            if filename.endswith(".osu"):
                meta = parse_osu_file(os.path.join(OSU_CHART_DIR, filename))
                if 'Title' in meta:
                    # Normalize title for better matching
                    osu_map[meta['Title'].lower()] = {'path': os.path.join(OSU_CHART_DIR, filename), 'meta': meta}

    all_samples, mappers = [], set(["unknown"])
    charts = sorted(os.listdir(INPUT_CHART_DIR))
    song_map = {s.split()[0]: s for s in os.listdir(INPUT_SONG_DIR)}

    for chart_filename in charts:
        try:
            # Extract title from the npy filename more robustly
            parts = chart_filename.replace('_', ' ').split()
            title = parts[1]

            if parts[0] in song_map:
                sample = {
                    "chart_path": os.path.join(INPUT_CHART_DIR, chart_filename),
                    "song_path": os.path.join(INPUT_SONG_DIR, song_map[parts[0]]),
                    "difficulty": "unknown"
                }

                # Find corresponding .osu and get mapper
                if title.lower() in osu_map:
                    mapper = osu_map[title.lower()]['meta'].get('Creator', 'unknown')
                    sample['mapper'] = mapper
                    mappers.add(mapper)
                    # Try to get difficulty from .osu as well
                    sample['difficulty'] = osu_map[title.lower()]['meta'].get('Version', 'unknown').lower()


                all_samples.append(sample)
        except (ValueError, IndexError):
            continue

    mapper_vocab = {name: i for i, name in enumerate(sorted(list(mappers)))}
    print(f"Found {len(all_samples)} samples and {len(mapper_vocab)} mappers.")

    kf = KFold(n_splits=config['training']['k_folds'], shuffle=True, random_state=42)
    train_indices, val_indices = list(kf.split(all_samples))[fold_idx]

    tokenizer = TaikoTokenizer()
    train_dataset = TaikoTransformerDataset(all_samples, train_indices, tokenizer, mapper_vocab, is_train=True, max_sequence_length=config['data']['max_sequence_length'])
    val_dataset = TaikoTransformerDataset(all_samples, val_indices, tokenizer, mapper_vocab, is_train=False, max_sequence_length=config['data']['max_sequence_length'])

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, collate_fn=lambda b: torch.utils.data.dataloader.default_collate([x for x in b if x is not None]))
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, collate_fn=lambda b: torch.utils.data.dataloader.default_collate([x for x in b if x is not None]))

    return train_loader, val_loader, tokenizer, mapper_vocab