import os
import sys
import json
from collections import Counter
import argparse

# Add project root to path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tokenization import TaikoTokenizer
from tqdm import tqdm
import numpy as np

def analyze_patterns(chart_dir, output_path, max_n=5):
    """
    Analyzes the frequency of n-grams (patterns) in a directory of chart files.
    """
    print(f"Analyzing patterns in {chart_dir}...")

    tokenizer = TaikoTokenizer()
    all_tokens = []

    chart_files = [f for f in os.listdir(chart_dir) if f.endswith('.npy')]
    for chart_filename in tqdm(chart_files, desc="Loading and tokenizing charts"):
        chart_path = os.path.join(chart_dir, chart_filename)
        # The tokenizer's tokenize method correctly handles .npy files
        tokens = tokenizer.tokenize(chart_path)
        if tokens:
            all_tokens.extend(tokens)

    print(f"Total tokens analyzed: {len(all_tokens)}")

    all_ngram_freqs = {}
    for n in range(2, max_n + 1):
        print(f"Calculating frequencies for n-grams of length {n}...")
        ngrams = Counter(zip(*[all_tokens[i:] for i in range(n)]))
        # Convert tuple keys to strings for JSON compatibility
        all_ngram_freqs[f'{n}-grams'] = {str(k): v for k, v in ngrams.items()}

    print(f"Saving pattern frequencies to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(all_ngram_freqs, f, indent=2)

    print("Analysis complete.")

def main():
    parser = argparse.ArgumentParser(description="Analyze n-gram pattern frequencies in Taiko charts.")
    parser.add_argument('chart_dir', type=str, help='Directory containing the .npy chart files.')
    parser.add_argument('output_path', type=str, help='Path to save the output JSON file.')
    parser.add_argument('--max_n', type=int, default=5, help='Maximum n-gram length to analyze.')

    args = parser.parse_args()

    analyze_patterns(args.chart_dir, args.output_path, args.max_n)

if __name__ == "__main__":
    main()