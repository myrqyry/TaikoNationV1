import torch
import argparse
import os
import yaml
import numpy as np

# Add root to sys.path to allow imports from parent directory
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformer_model import TaikoTransformer
from tokenization import TaikoTokenizer
from audio_processing import get_audio_features
from tools.difficulty_scaler import DifficultyScaler

class TaikoChartGenerator:
    def __init__(self, model_path, config_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = self._load_config(config_path)
        self.tokenizer = TaikoTokenizer()
        self.model = self._load_model(model_path)
        self.model.eval()

    def _load_config(self, path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def _load_model(self, model_path):
        model = TaikoTransformer(
            vocab_size=self.tokenizer.vocab_size,
            **self.config['model']
        ).to(self.device)

        try:
            model.load_state_dict(torch.load(model_path, map_location=self.device))
        except FileNotFoundError:
            print(f"Warning: Model file not found at {model_path}. Using an untrained model.")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)
        return model

    def generate(self, audio_path, max_len=1024):
        audio_features = get_audio_features(audio_path)
        if audio_features is None:
            return None, "Failed to process audio."

        # Pad or truncate audio features
        max_seq_len = self.config['data']['max_sequence_length']
        if audio_features.shape[0] < max_seq_len:
            padding = np.zeros((max_seq_len - audio_features.shape[0], audio_features.shape[1]))
            audio_features = np.vstack([audio_features, padding])
        else:
            audio_features = audio_features[:max_seq_len]

        encoder_input = torch.from_numpy(audio_features).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output_sequences = self.model.generate(encoder_input, max_len=max_len)
        token_ids = output_sequences.squeeze(0).cpu().tolist()

        return token_ids, None

    def export_to_osu(self, tokens, output_path):
        with open(output_path, 'w') as f:
            f.write("osu file format v14\n\n[HitObjects]\n")
            for i, token_str in enumerate(tokens):
                if token_str not in ["[PAD]", "[EMPTY]"]:
                    time = (i + 1) * 100
                    f.write(f"256,192,{time},1,0,0:0:0:0:{token_str}\n")
        print(f"Chart exported to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and optionally scale a Taiko chart.")
    parser.add_argument('--model_path', type=str, default='output/taiko_transformer.pth')
    parser.add_argument('--audio_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, default='output/generated_chart.osu')
    parser.add_argument('--config', type=str, default='config/default.yaml')
    parser.add_argument('--max_gen_len', type=int, default=512)
    parser.add_argument('--scale_difficulty', type=float, default=0.0, help="Factor to scale difficulty by. Positive increases, negative decreases.")

    args = parser.parse_args()

    generator = TaikoChartGenerator(args.model_path, args.config)
    generated_ids, error = generator.generate(args.audio_path, max_len=args.max_gen_len)

    if error:
        print(f"Error: {error}")
    else:
        if args.scale_difficulty != 0.0:
            print(f"Scaling difficulty by a factor of {args.scale_difficulty}...")
            scaler = DifficultyScaler(generator.tokenizer)
            if args.scale_difficulty > 0:
                generated_ids = scaler.increase_difficulty(generated_ids, factor=args.scale_difficulty)
            else:
                generated_ids = scaler.decrease_difficulty(generated_ids, factor=abs(args.scale_difficulty))

        final_tokens = generator.tokenizer.detokenize(generated_ids)
        generator.export_to_osu(final_tokens, args.output_path)