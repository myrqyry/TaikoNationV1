import sys
import torch
import numpy as np
import os
import yaml
import argparse

# Add the project root to the Python path to allow for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformer_model import PatternAwareTransformer
from tokenization import TaikoTokenizer
from audio_processing import get_audio_features
from eval.pattern_evaluation import PatternEvaluator

def load_model(model_path, config, tokenizer, device):
    """Loads a trained PatternAwareTransformer model from a .pth file."""
    print("Loading model...")
    model_config = config['model']
    model_config.pop('vocab_size', None)

    model = PatternAwareTransformer(
        vocab_size=tokenizer.vocab_size,
        **model_config
    ).to(device)

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        sys.exit(1)

    model.eval()
    print("Model loaded successfully.")
    return model

def generate_chart(model, audio_path, tokenizer, config, device, max_len=1024):
    """Generates a chart autoregressively from an audio file."""
    print(f"Generating chart for {os.path.basename(audio_path)}...")
    audio_features = get_audio_features(
        audio_path,
        source_resolution_ms=config['data']['source_resolution_ms'],
        frame_duration_ms=config['data']['time_quantization_ms']
    )
    if audio_features is None:
        return []

    if audio_features.shape[0] > config['data']['max_sequence_length']:
        audio_features = audio_features[:config['data']['max_sequence_length']]
    else:
        pad_len = config['data']['max_sequence_length'] - audio_features.shape[0]
        audio_features = np.pad(audio_features, ((0, pad_len), (0, 0)), 'constant')

    encoder_input = torch.from_numpy(audio_features).float().unsqueeze(0).to(device)
    cls_token_id = tokenizer.vocab["[CLS]"]
    generated_tokens = [cls_token_id]

    for _ in range(max_len):
        decoder_input = torch.tensor([generated_tokens], dtype=torch.long).to(device)
        with torch.no_grad():
            token_logits = model(encoder_input, decoder_input)
        next_token_id = token_logits.argmax(dim=-1)[:, -1].item()
        if next_token_id == tokenizer.vocab["[PAD]"] or next_token_id == tokenizer.vocab["[SEP]"]:
            break
        generated_tokens.append(next_token_id)

    print(f"Generated {len(generated_tokens)} tokens.")
    return generated_tokens[1:]

def export_to_osu(tokens, output_path, tokenizer, config, metadata=None):
    """Exports a sequence of tokens to a simplified .osu file format for Taiko."""
    if metadata is None:
        metadata = {'Title': 'AI Generated Chart', 'Artist': 'TaikoTransformer', 'Version': 'v1.0'}
    time_per_token = config['data']['time_quantization_ms']
    hit_objects = []
    current_time = 500
    detokenized = tokenizer.detokenize(tokens)
    for token_str in detokenized:
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
        f.write(f"Title:{metadata['Title']}\nArtist:{metadata['Artist']}\nVersion:{metadata['Version']}\n\n")
        f.write("[Difficulty]\nHPDrainRate:5\nCircleSize:5\nOverallDifficulty:5\nApproachRate:5\nSliderMultiplier:1.4\nSliderTickRate:1\n\n")
        f.write("[HitObjects]\n")
        for obj in hit_objects:
            f.write(f"{obj}\n")
    print(f"Successfully exported chart to {output_path}")

def evaluate_model(model, tokenizer, config, device, test_audio_dir, human_charts_dir="input_charts_nr", max_files=None):
    """Evaluates the model by generating charts, comparing them to human-created charts, and calculating pattern-based metrics."""
    print("\n--- Running Pattern Evaluation ---")
    evaluator = PatternEvaluator()
    human_patterns = []
    print(f"Loading human patterns from {human_charts_dir}...")
    human_chart_files = [f for f in os.listdir(human_charts_dir) if f.endswith('.npy')]
    for chart_filename in human_chart_files:
        tokens = tokenizer.tokenize(os.path.join(human_charts_dir, chart_filename))
        if tokens: human_patterns.append(tokens)
    if not human_patterns:
        print("Could not load any human patterns for evaluation. Aborting.")
        return
    print(f"Loaded {len(human_patterns)} human patterns.")
    test_audio_files = [f for f in os.listdir(test_audio_dir) if f.endswith('.npy')]
    if max_files: test_audio_files = test_audio_files[:max_files]
    all_metrics = []
    for audio_filename in test_audio_files:
        print(f"\nEvaluating chart for {audio_filename}...")
        audio_path = os.path.join(test_audio_dir, audio_filename)
        generated_tokens = generate_chart(model, audio_path, tokenizer, config, device)
        if not generated_tokens:
            print("Skipping evaluation for this file as no tokens were generated.")
            continue
        overlap = evaluator.calculate_pattern_overlap(generated_tokens, human_patterns)
        coverage = evaluator.calculate_pattern_space_coverage(generated_tokens)
        denden = evaluator.evaluate_denden_sequences(generated_tokens, tokenizer)
        metrics = {"file": audio_filename, "pattern_overlap": overlap, "pattern_coverage": coverage, "denden_count": denden['count']}
        all_metrics.append(metrics)
        print(f"  - Pattern Overlap (vs. human set): {overlap:.4f}")
        print(f"  - Pattern Coverage (variety): {coverage:.4f}")
        print(f"  - Denden Rolls: {denden['count']} (avg length: {denden['avg_length']:.2f})")
        output_filename = os.path.splitext(audio_filename)[0] + ".osu"
        output_path = os.path.join("output/generated_charts", output_filename)
        export_to_osu(generated_tokens, output_path, tokenizer, config, metadata={'Title': os.path.splitext(audio_filename)[0], 'Artist': 'AI', 'Version': 'Generated'})
    if all_metrics:
        avg_overlap = np.mean([m['pattern_overlap'] for m in all_metrics])
        avg_coverage = np.mean([m['pattern_coverage'] for m in all_metrics])
        avg_denden = np.mean([m['denden_count'] for m in all_metrics])
        print("\n--- Average Metrics ---")
        print(f"Average Pattern Overlap: {avg_overlap:.4f}")
        print(f"Average Pattern Coverage: {avg_coverage:.4f}")
        print(f"Average Denden Count: {avg_denden:.2f}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained Taiko Transformer model.")
    parser.add_argument('--model_path', type=str, default='output/taiko_transformer.pth_fold_1.pth', help='Path to the trained model .pth file.')
    parser.add_argument('--config', type=str, default='config/default.yaml', help='Path to the model configuration file.')
    parser.add_argument('--test_data', type=str, default='input_songs', help='Path to the directory of test audio files.')
    parser.add_argument('--output_dir', type=str, default='output/generated_charts', help='Directory to save generated charts.')
    args = parser.parse_args()
    print("--- Starting Model Evaluation ---")
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    tokenizer = TaikoTokenizer(
        time_quantization=config['data']['time_quantization_ms'],
        source_resolution=config['data']['source_resolution_ms']
    )
    model = load_model(args.model_path, config, tokenizer, device)
    evaluate_model(model, tokenizer, config, device, args.test_data, max_files=1)
    print("--- Evaluation Complete ---")

if __name__ == "__main__":
    main()