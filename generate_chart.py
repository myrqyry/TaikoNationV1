# generate_chart.py
import argparse
import os
import torch
import yaml
import numpy as np

from transformer_model import TaikoTransformer
from tokenization import TaikoTokenizer
from audio_processing import get_audio_features
from transformer_dataset import DIFFICULTY_MAP

def load_config(config_path):
    """Loads a YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_model(checkpoint_path, config, device):
    """Loads a trained model from a checkpoint."""
    # We need to know the vocab size and number of genres/difficulties to init the model.
    # This info should ideally be saved in the checkpoint. For now, let's assume
    # we can derive them or use placeholders.
    # Placeholder values - these should be updated based on the actual training data/vocab
    vocab_size = TaikoTokenizer().vocab_size
    num_genres = 10 # Placeholder
    num_difficulties = len(DIFFICULTY_MAP)

    model = TaikoTransformer(
        vocab_size=vocab_size,
        num_genres=num_genres,
        num_difficulties=num_difficulties,
        d_model=config['model']['d_model'],
        nhead=config['model']['nhead'],
        num_encoder_layers=config['model']['num_encoder_layers'],
        num_decoder_layers=config['model']['num_decoder_layers'],
        dim_feedforward=config['model']['dim_feedforward'],
        dropout=config['model']['dropout'],
        audio_feature_size=config['model']['audio_feature_size'],
        max_sequence_length=config['data']['max_sequence_length']
    ).to(device)

    # Load the trained weights
    # Note: This will fail until a model is trained and saved.
    # For initial development, we can comment this out and use the randomly initialized model.
    # model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    model.eval()
    return model

def generate_chart(model, audio_features, tokenizer, difficulty_id, config, device):
    """Generates a chart token sequence from audio features."""
    print("Generating chart...")

    model.eval()

    # Prepare inputs
    encoder_input = torch.from_numpy(audio_features).float().unsqueeze(0).to(device)

    # Start with a CLS token
    decoder_input = torch.tensor([[tokenizer.vocab["[CLS]"]]], dtype=torch.long).to(device)

    generated_tokens = []

    with torch.no_grad():
        # The loop must not exceed the model's maximum sequence length.
        # We subtract 1 because the sequence starts with a [CLS] token.
        max_len = config['data']['max_sequence_length']
        for _ in range(max_len - 1):
            # For simplicity, we use a placeholder genre_id
            genre_id = torch.tensor([0], dtype=torch.long).to(device)
            difficulty_tensor = torch.tensor([difficulty_id], dtype=torch.long).to(device)

            # Get model output
            output_logits = model(encoder_input, decoder_input, genre_id, difficulty_tensor)

            # Greedy decoding: get the most likely next token
            next_token_logits = output_logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).item()

            generated_tokens.append(next_token_id)

            # Append the new token to the decoder input for the next step
            next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long).to(device)
            decoder_input = torch.cat([decoder_input, next_token_tensor], dim=1)

    return generated_tokens

def save_osu_chart(token_ids, tokenizer, output_path, audio_filename):
    """Saves the generated tokens to a basic .osu file."""
    print(f"Saving chart to {output_path}...")

    # This is a very basic .osu file header.
    # A real implementation would have more metadata.
    osu_header = f"""osu file format v14
[General]
AudioFilename: {os.path.basename(audio_filename)}
AudioLeadIn: 0
Mode: 1
[Metadata]
Title: Generated Chart
Artist: AI
Creator: TaikoNationV1
Version: Normal
[Difficulty]
HPDrainRate:5
CircleSize:5
OverallDifficulty:5
ApproachRate:5
SliderMultiplier:1.4
SliderTickRate:1
[HitObjects]
"""

    # Use the tokenizer to convert IDs back to human-readable token names
    token_names = tokenizer.detokenize(token_ids)

    with open(output_path, "w") as f:
        f.write(osu_header)

        # Simple conversion of tokens to hit objects
        # This assumes each token is a note at a fixed time interval.
        time_interval = 200 # ms
        current_time = 1000 # Start time

        for token_name in token_names:
            if token_name not in tokenizer.special_tokens and token_name != "[EMPTY]":
                # x,y,time,type,hitSound,objectParams,hitSample
                # For Taiko, x is always 256, y is always 192
                # type 1 = don, type 4 = finisher, type 8 = ka
                note_type = 1 # Default to 'don'
                if "ka" in token_name:
                    note_type = 8
                if "big" in token_name:
                    note_type = 4

                f.write(f"256,192,{current_time},{note_type},0,0:0:0:0:\n")
                current_time += time_interval

    print("Chart saved successfully.")


def main():
    parser = argparse.ArgumentParser(description="Generate a Taiko chart from an audio file.")
    parser.add_argument("model_path", help="Path to the trained model checkpoint (.pth).")
    parser.add_argument("audio_path", help="Path to the input audio features file (.npy).")
    parser.add_argument("output_path", help="Path to save the generated chart (.osu).")
    parser.add_argument("--difficulty", "-d", default="oni", help="Chart difficulty (e.g., easy, normal, hard, oni).")
    parser.add_argument("--config", default="config/default.yaml", help="Path to the configuration file.")

    args = parser.parse_args()

    # --- Setup ---
    if not os.path.exists(args.audio_path):
        print(f"Error: Audio file not found at {args.audio_path}")
        return

    # For now, we'll alert the user that the model file doesn't exist yet
    if not os.path.exists(args.model_path):
        print(f"Warning: Model checkpoint not found at {args.model_path}. The script will run with a randomly initialized model, producing random output.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Everything ---
    config = load_config(args.config)
    tokenizer = TaikoTokenizer()
    model = load_model(args.model_path, config, device)

    difficulty_str = args.difficulty.lower()
    if difficulty_str not in DIFFICULTY_MAP:
        print(f"Error: Invalid difficulty '{args.difficulty}'. Please choose from {list(DIFFICULTY_MAP.keys())}.")
        return
    difficulty_id = DIFFICULTY_MAP[difficulty_str]

    # For now, we assume the .npy file is already processed correctly.
    # A more robust implementation would run get_audio_features here.
    try:
        audio_features = np.load(args.audio_path)
        # Truncate the audio features to the model's max sequence length
        max_len = config['data']['max_sequence_length']
        if audio_features.shape[0] > max_len:
            print(f"Warning: Audio features longer than max sequence length ({max_len}). Truncating.")
            audio_features = audio_features[:max_len, :]
    except Exception as e:
        print(f"Error loading or processing audio features: {e}")
        return

    # --- Generate and Save ---
    generated_token_ids = generate_chart(model, audio_features, tokenizer, difficulty_id, config, device)
    save_osu_chart(generated_token_ids, tokenizer, args.output_path, args.audio_path)


if __name__ == "__main__":
    main()
