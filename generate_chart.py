import torch
import yaml
import os
import sys
import numpy as np
from tqdm import tqdm
import argparse

# Add project root to path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformer_model import MultiTaskTaikoTransformer
from tokenization import TaikoTokenizer
from audio_processing import get_audio_features

class TaikoChartGenerator:
    """
    A user-friendly interface for generating Taiko charts from audio files
    using a trained MultiTaskTaikoTransformer model.
    """
    def __init__(self, model_path, config_path='config/default.yaml'):
        print("--- Initializing Taiko Chart Generator ---")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.tokenizer = TaikoTokenizer(
            time_quantization=self.config['data']['time_quantization_ms'],
            source_resolution=self.config['data']['source_resolution_ms']
        )

        self.model = self._load_model(model_path)
        print(f"Generator initialized on {self.device}.")
        self.difficulty_map = {
            'easy': 0, 'normal': 1, 'hard': 2, 'oni': 3, 'ura': 4,
            'kantan': 0, 'futsuu': 1, 'muzukashii': 2, 'uraoni': 4,
            'inneroni': 4, 'expert': 3
        }

    def _load_model(self, model_path):
        """Loads a trained MultiTaskTaikoTransformer model."""
        print(f"Loading model from {model_path}...")
        model_config = self.config['model']

        model = MultiTaskTaikoTransformer(
            vocab_size=self.tokenizer.vocab_size,
            num_difficulty_classes=self.config['training']['multi_task']['num_difficulty_classes'],
            **model_config
        ).to(self.device)

        try:
            model.load_state_dict(torch.load(model_path, map_location=self.device))
        except FileNotFoundError:
            print(f"FATAL: Model file not found at {model_path}.")
            sys.exit(1)
        except Exception as e:
            print(f"FATAL: Error loading model state_dict: {e}")
            sys.exit(1)

        model.eval()
        print("Model loaded successfully.")
        return model

    def generate_from_audio(self, audio_path, target_difficulty='oni', max_len=2048):
        """
        Generates a sequence of tokens from an audio file, conditioned on a target difficulty.
        """
        print(f"Processing audio file: {os.path.basename(audio_path)}...")
        audio_features = get_audio_features(
            audio_path,
            source_resolution_ms=self.config['data']['source_resolution_ms'],
            frame_duration_ms=self.config['data']['time_quantization_ms']
        )
        if audio_features is None:
            print("Error: Could not process audio.")
            return []

        # Pad or truncate audio features
        seq_len = self.config['data']['max_sequence_length']
        if audio_features.shape[0] > seq_len:
            audio_features = audio_features[:seq_len]
        else:
            pad_len = seq_len - audio_features.shape[0]
            audio_features = np.pad(audio_features, ((0, pad_len), (0, 0)), 'constant')

        encoder_input = torch.from_numpy(audio_features).float().unsqueeze(0).to(self.device)

        # Initialize generation
        cls_token_id = self.tokenizer.vocab["[CLS]"]
        generated_tokens = [cls_token_id]

        # Convert target difficulty string to class index
        difficulty_class = self.difficulty_map.get(target_difficulty.lower(), 3) # Default to Oni
        difficulty_tensor = torch.tensor([difficulty_class], dtype=torch.long).to(self.device)

        print(f"Generating chart tokens for difficulty: {target_difficulty} ({difficulty_class})")
        # Autoregressive generation loop with progress bar
        for _ in tqdm(range(max_len), desc="Generating Tokens", unit="token"):
            if len(generated_tokens) >= self.config['data']['max_sequence_length']:
                print("\nMax sequence length reached. Stopping.")
                break

            decoder_input = torch.tensor([generated_tokens], dtype=torch.long).to(self.device)
            with torch.no_grad():
                predictions = self.model(src=encoder_input, tgt=decoder_input, target_difficulty=difficulty_tensor)

            token_logits = predictions['tokens']
            next_token_id = token_logits.argmax(dim=-1)[:, -1].item()

            if next_token_id in [self.tokenizer.vocab["[PAD]"], self.tokenizer.vocab["[SEP]"]]:
                print("\nEnd of sequence token generated. Stopping.")
                break

            generated_tokens.append(next_token_id)

        print(f"Generated a total of {len(generated_tokens) - 1} tokens.")
        return generated_tokens[1:] # Exclude the initial [CLS] token

    def export_to_osu(self, tokens, output_path, metadata=None):
        """Exports a sequence of tokens to the .osu file format."""
        pass

    def export_to_tja(self, tokens, output_path, metadata=None):
        """Exports a sequence of tokens to the .tja file format."""
        pass

def main():
    parser = argparse.ArgumentParser(description="Generate a Taiko chart from an audio file.")
    parser.add_argument('model_path', type=str, help='Path to the trained model checkpoint (.pth).')
    parser.add_argument('audio_path', type=str, help='Path to the input audio file (.npy).')
    parser.add_argument('output_path', type=str, help='Path to save the generated chart file.')
    parser.add_argument('--config', type=str, default='config/default.yaml', help='Path to the configuration file.')
    parser.add_argument('--difficulty', type=str, default='oni', help='Target difficulty for the generated chart.')

    args = parser.parse_args()

    print("--- Taiko Chart Generator CLI ---")

    generator = TaikoChartGenerator(args.model_path, args.config)

    tokens = generator.generate_from_audio(args.audio_path, target_difficulty=args.difficulty)

    if not tokens:
        print("Chart generation failed.")
        return

    generator.export_to_osu(tokens, args.output_path)

    print("--- Chart generation complete! ---")

if __name__ == "__main__":
    main()