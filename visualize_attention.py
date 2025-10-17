import torch
import argparse
import os
import yaml
import matplotlib.pyplot as plt
import numpy as np

from transformer_model import TaikoTransformer
from tokenization import TaikoTokenizer
from audio_processing import get_audio_features

def load_config(path="config/default.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def plot_attention_heatmap(attention_map, output_path, type='cross'):
    """
    Generates and saves a heatmap for a given attention type.
    """
    attention_map = attention_map.cpu().numpy()

    fig, ax = plt.subplots(figsize=(15, 12))
    cax = ax.matshow(attention_map, cmap='viridis', aspect='auto')
    fig.colorbar(cax)

    if type == 'cross':
        ax.set_xlabel('Audio Time Steps')
        ax.set_ylabel('Generated Note Tokens (Final Step)')
        ax.set_title('Cross-Attention Heatmap (Token-to-Audio Focus)')
    else: # self-attention
        ax.set_xlabel('Generated Note Tokens (Keys)')
        ax.set_ylabel('Generated Note Tokens (Queries)')
        ax.set_title('Decoder Self-Attention Heatmap (Token-to-Token Focus)')

    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=15))
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=15))

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Heatmap for '{type}' attention saved to {output_path}")

def visualize(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(args.config)
    tokenizer = TaikoTokenizer()
    model = TaikoTransformer(vocab_size=tokenizer.vocab_size, **config['model']).to(device)

    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Model loaded from {args.model_path}")
    except FileNotFoundError:
        print(f"Warning: Model not found. Using a dummy model.")
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        torch.save(model.state_dict(), args.model_path)

    model.eval()

    print("Processing audio...")
    audio_features = get_audio_features(
        args.audio_path,
        source_resolution_ms=config['data']['source_resolution_ms'],
        frame_duration_ms=config['data']['time_quantization_ms']
    )
    if audio_features is None: return

    max_len = config['data']['max_sequence_length']
    if audio_features.shape[0] < max_len:
        padding = torch.zeros(max_len - audio_features.shape[0], audio_features.shape[1])
        audio_features = torch.cat([torch.from_numpy(audio_features), padding], dim=0)
    else:
        audio_features = torch.from_numpy(audio_features[:max_len])

    encoder_input = audio_features.unsqueeze(0).to(device)

    print("Generating chart and capturing attention...")
    with torch.no_grad():
        output = model.generate(
            encoder_input,
            max_len=args.max_gen_len,
            tokenizer=tokenizer,
            return_attention=True
        )

    attn_key = f"{args.attention_type}_attention"
    attention_map = output.get(attn_key)

    if attention_map is None:
        print(f"Error: Could not retrieve '{attn_key}' from model output.")
        return

    attention_to_plot = attention_map.squeeze(0).mean(dim=0).cpu()

    if attention_to_plot.ndim == 1:
        attention_to_plot = attention_to_plot.unsqueeze(0)

    if attention_to_plot.ndim != 2:
        print(f"Error: Final attention map is not 2D. Shape: {attention_to_plot.shape}")
        return

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    plot_attention_heatmap(attention_to_plot, args.output_path, type=args.attention_type)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize model attention.")
    parser.add_argument('--model_path', type=str, default='output/taiko_transformer.pth', help='Path to the trained model.')
    parser.add_argument('--audio_path', type=str, required=True, help='Path to the input audio .npy file.')
    parser.add_argument('--output_path', type=str, default='output/attention_heatmap.png', help='Path to save the heatmap image.')
    parser.add_argument('--config', type=str, default='config/default.yaml', help='Path to the config file.')
    parser.add_argument('--max_gen_len', type=int, default=128, help='Max length of the generated chart.')
    parser.add_argument('--attention_type', type=str, default='cross', choices=['cross', 'self'], help='Type of attention to visualize.')

    args = parser.parse_args()
    visualize(args)