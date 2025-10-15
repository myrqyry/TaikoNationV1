import torch
import yaml
import os
import sys
import numpy as np
from tqdm import tqdm
import argparse

# Add project root to path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformer_model import PatternAwareTransformer
from tokenization import TaikoTokenizer
from audio_processing import get_audio_features

class TaikoChartGenerator:
    """
    A user-friendly interface for generating Taiko charts from audio files
    using a trained PatternAwareTransformer model.
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

    def _load_model(self, model_path):
        """Loads a trained PatternAwareTransformer model."""
        print(f"Loading model from {model_path}...")
        model_config = self.config['model']

        model = PatternAwareTransformer(
            vocab_size=self.tokenizer.vocab_size,
            max_sequence_length=self.config['data']['max_sequence_length'],
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

    def generate_from_audio(self, audio_path, max_len=2048):
        """
        Generates a sequence of tokens from an audio file, showing a progress bar.
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

        print("Generating chart tokens...")
        # Autoregressive generation loop with progress bar
        for _ in tqdm(range(max_len), desc="Generating Tokens", unit="token"):
            if len(generated_tokens) >= self.config['data']['max_sequence_length']:
                print("\nMax sequence length reached. Stopping.")
                break

            decoder_input = torch.tensor([generated_tokens], dtype=torch.long).to(self.device)
            with torch.no_grad():
                token_logits = self.model(encoder_input, decoder_input)

            next_token_id = token_logits.argmax(dim=-1)[:, -1].item()

            if next_token_id in [self.tokenizer.vocab["[PAD]"], self.tokenizer.vocab["[SEP]"]]:
                print("\nEnd of sequence token generated. Stopping.")
                break

            generated_tokens.append(next_token_id)

        print(f"Generated a total of {len(generated_tokens) - 1} tokens.")
        return generated_tokens[1:] # Exclude the initial [CLS] token

    def export_to_osu(self, tokens, output_path, metadata=None):
        """Exports a sequence of tokens to the .osu file format."""
        if metadata is None:
            metadata = {'Title': 'AI Generated Chart', 'Artist': 'TaikoTransformer', 'Version': 'v1.0'}

        time_per_token = self.config['data']['time_quantization_ms']
        hit_objects = []
        current_time = 1000  # Start time for the first note

        for token_str in self.tokenizer.detokenize(tokens):
            if token_str in ["[PAD]", "[EMPTY]", "[CLS]", "[SEP]"]:
                current_time += time_per_token
                continue

            events = token_str.split(',')
            for event in events:
                x, y, hitsound, note_type = 128, 192, 0, 1
                if "big" in event: hitsound += 4
                if "finisher" in event: hitsound += 8
                if "ka" in event: hitsound += 2
                hit_objects.append(f"{x},{y},{current_time},{note_type},{hitsound},0:0:0:0:")
            current_time += time_per_token

        with open(output_path, 'w') as f:
            f.write("osu file format v14\n\n[General]\nMode: 1\n\n[Metadata]\n")
            f.write(f"Title:{metadata.get('Title', 'AI Chart')}\n")
            f.write(f"Artist:{metadata.get('Artist', 'AI')}\n")
            f.write(f"Version:{metadata.get('Version', 'Generated')}\n\n")
            f.write("[Difficulty]\nHPDrainRate:5\nCircleSize:5\nOverallDifficulty:5\nApproachRate:5\nSliderMultiplier:1.4\nSliderTickRate:1\n\n")
            f.write("[HitObjects]\n")
            f.write("\n".join(hit_objects))

        print(f"Successfully exported chart to {output_path}")

    def export_to_tja(self, tokens, output_path, metadata=None):
        """Exports a sequence of tokens to the .tja file format."""
        if metadata is None:
            metadata = {
                'TITLE': 'AI Generated Chart',
                'WAVE': 'song.ogg',
                'BPM': 150,
                'OFFSET': -1.5,
                'COURSE': 'Oni'
            }

        note_map = {
            "don": "1", "ka": "2", "big_don": "3", "big_ka": "4",
            "roll_start": "5", "roll_end": "8", "finisher": "7"
            # NOTE: "roll_cont" is not a standard TJA event, so we treat it as part of a roll.
        }

        chart_data = []
        notes_in_measure = 0
        beats_per_measure = 4 * 4 # Assuming 4/4 time signature, 16th notes

        detokenized = self.tokenizer.detokenize(tokens)
        for token_str in detokenized:
            if token_str in ["[PAD]", "[EMPTY]", "[CLS]", "[SEP]"]:
                chart_data.append("0")
            else:
                events = token_str.split(',')
                # For simplicity, we take the first valid event in a token
                note = "0"
                for event in events:
                    if event in note_map:
                        note = note_map[event]
                        break
                chart_data.append(note)

            notes_in_measure += 1
            if notes_in_measure >= beats_per_measure:
                chart_data.append(",")
                notes_in_measure = 0

        with open(output_path, 'w') as f:
            f.write(f"TITLE:{metadata.get('TITLE', 'AI Chart')}\n")
            f.write(f"WAVE:{metadata.get('WAVE', 'song.ogg')}\n")
            f.write(f"BPM:{metadata.get('BPM', 150)}\n")
            f.write(f"OFFSET:{metadata.get('OFFSET', -1.5)}\n\n")
            f.write("COURSE:Oni\n")
            f.write("LEVEL:10\n\n")
            f.write("#START\n")
            f.write("".join(chart_data))
            f.write("\n#END\n")

        print(f"Successfully exported chart to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate a Taiko chart from an audio file using a trained model.")
    parser.add_argument('model_path', type=str, help='Path to the trained model checkpoint (.pth).')
    parser.add_argument('audio_path', type=str, help='Path to the input audio file (.npy).')
    parser.add_argument('output_path', type=str, help='Path to save the generated chart file.')
    parser.add_argument('--config', type=str, default='config/default.yaml', help='Path to the configuration file.')
    parser.add_argument('--format', type=str, default='osu', choices=['osu', 'tja'], help='Output chart format.')

    args = parser.parse_args()

    print("--- Taiko Chart Generator CLI ---")

    # Initialize the generator
    generator = TaikoChartGenerator(args.model_path, args.config)

    # Generate tokens
    tokens = generator.generate_from_audio(args.audio_path)

    if not tokens:
        print("Chart generation failed.")
        return

    # Export to the desired format
    if args.format == 'osu':
        generator.export_to_osu(tokens, args.output_path)
    elif args.format == 'tja':
        generator.export_to_tja(tokens, args.output_path)

    print("--- Chart generation complete! ---")

if __name__ == "__main__":
    main()