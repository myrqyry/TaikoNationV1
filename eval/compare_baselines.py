import sys
import torch
import numpy as np
import os
import yaml
import argparse

# Add the project root to the Python path to allow for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tokenization import TaikoTokenizer
from eval.pattern_evaluation import PatternEvaluator
from generate_chart import TaikoChartGenerator

def evaluate_model(generator, test_audio_dir, human_charts_dir="input_charts_nr", max_files=None):
    """Evaluates the model by generating charts, comparing them to human-created charts, and calculating pattern-based metrics."""
    print("\n--- Running Pattern Evaluation ---")
    evaluator = PatternEvaluator()
    human_patterns = []
    print(f"Loading human patterns from {human_charts_dir}...")
    human_chart_files = [f for f in os.listdir(human_charts_dir) if f.endswith('.npy')]
    for chart_filename in human_chart_files:
        tokens = generator.tokenizer.tokenize(os.path.join(human_charts_dir, chart_filename))
        if tokens: human_patterns.append(tokens)
    if not human_patterns:
        print("Could not load any human patterns for evaluation. Aborting.")
        return
    print(f"Loaded {len(human_patterns)} human patterns.")

    test_audio_files = [f for f in os.listdir(test_audio_dir) if f.endswith('.npy')]
    if max_files: test_audio_files = test_audio_files[:max_files]

    all_metrics = []
    for audio_filename in test_audio_files:
        print(f"\n--- Evaluating chart for {audio_filename} ---")
        audio_path = os.path.join(test_audio_dir, audio_filename)

        generated_tokens = generator.generate_from_audio(audio_path)
        if not generated_tokens:
            print("Skipping evaluation for this file as no tokens were generated.")
            continue

        # Calculate metrics
        overlap = evaluator.calculate_pattern_overlap(generated_tokens, human_patterns)
        coverage = evaluator.calculate_pattern_space_coverage(generated_tokens)
        denden = evaluator.evaluate_denden_sequences(generated_tokens, generator.tokenizer)

        metrics = {"file": audio_filename, "pattern_overlap": overlap, "pattern_coverage": coverage, "denden_count": denden['count']}
        all_metrics.append(metrics)
        print(f"  - Pattern Overlap (vs. human set): {overlap:.4f}")
        print(f"  - Pattern Coverage (variety): {coverage:.4f}")
        print(f"  - Denden Rolls: {denden['count']} (avg length: {denden['avg_length']:.2f})")

        # Export the generated chart
        output_filename = os.path.splitext(audio_filename)[0] + ".osu"
        output_path = os.path.join("output/generated_charts", output_filename)
        generator.export_to_osu(generated_tokens, output_path, metadata={'Title': os.path.splitext(audio_filename)[0]})

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
    parser.add_argument('model_path', type=str, help='Path to the trained model checkpoint (.pth).')
    parser.add_argument('test_data_dir', type=str, help='Path to the directory of test audio files.')
    parser.add_argument('--config', type=str, default='config/default.yaml', help='Path to the model configuration file.')
    parser.add_argument('--max_files', type=int, default=None, help='Maximum number of files to evaluate.')

    args = parser.parse_args()

    print("--- Starting Model Evaluation ---")

    # Initialize the generator, which loads the model and tokenizer
    generator = TaikoChartGenerator(args.model_path, args.config)

    # Run the evaluation
    evaluate_model(generator, args.test_data_dir, max_files=args.max_files)

    print("\n--- Evaluation Complete ---")

if __name__ == "__main__":
    main()