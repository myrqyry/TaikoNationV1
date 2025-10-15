import torch
import yaml
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformer_model import PatternAwareTransformer
from tokenization import TaikoTokenizer
from audio_processing import get_audio_features
from generate_chart import TaikoChartGenerator

def generate_attention_map(generator, audio_path, output_path):
    """
    Generates and saves a heatmap of the model's pattern attention weights
    across the full generation of a chart.
    """
    print(f"Step 1: Generating chart to collect attention weights for {os.path.basename(audio_path)}...")
    # This call will populate generator.last_attention_weights
    generated_tokens = generator.generate_from_audio(audio_path)

    if generator.last_attention_weights is None:
        print("Error: Could not retrieve attention weights.")
        return

    # 2. Process the collected attention weights
    print("Step 2: Processing attention weights...")
    # Average the weights across all attention heads.
    # The shape is (num_steps, 1, num_heads, 1, pattern_memory_size), so we squeeze and average.
    attn_weights = generator.last_attention_weights.squeeze().mean(axis=1)

    # 3. Create and save the heatmap
    print(f"Step 3: Creating and saving heatmap to {output_path}...")
    plt.figure(figsize=(20, 12))
    sns.heatmap(attn_weights, cmap='viridis', cbar=True, xticklabels=False)
    plt.title(f"Pattern Attention Heatmap for {os.path.basename(audio_path)}", fontsize=16)
    plt.xlabel("Pattern Memory Index", fontsize=12)
    plt.ylabel("Generated Token Sequence", fontsize=12)

    try:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Attention map saved successfully.")
    except Exception as e:
        print(f"Error saving heatmap: {e}")

    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize the attention of a trained Taiko Transformer model.")
    parser.add_argument('model_path', type=str, help='Path to the trained model checkpoint (.pth).')
    parser.add_argument('audio_path', type=str, help='Path to the input audio file (.npy).')
    parser.add_argument('output_path', type=str, help='Path to save the attention heatmap image (.png).')
    parser.add_argument('--config', type=str, default='config/default.yaml', help='Path to the configuration file.')

    args = parser.parse_args()

    print("--- Starting Attention Visualization ---")

    # Use the generator to handle model loading and config
    generator = TaikoChartGenerator(args.model_path, args.config)

    # Generate and save the attention map
    generate_attention_map(generator, args.audio_path, args.output_path)

    print("--- Visualization Complete ---")

if __name__ == "__main__":
    main()