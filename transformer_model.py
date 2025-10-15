import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Adds positional information to the input embeddings, allowing the transformer
    to understand the order of the sequence.
    """
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class TaikoTransformer(nn.Module):
    """
    A Transformer model for generating Taiko charts from audio features.
    It uses an encoder-decoder architecture to map a sequence of audio
    features to a sequence of note tokens.
    """
    def __init__(self, vocab_size, d_model=256, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 audio_feature_size=80, max_sequence_length=2048):
        super(TaikoTransformer, self).__init__()
        self.d_model = d_model

        # --- Layers ---
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_sequence_length)

        # A linear layer to project the audio features into the model's dimension (d_model)
        self.audio_input_projection = nn.Linear(audio_feature_size, d_model)

        # The core Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True # This simplifies tensor manipulation
        )

        # Final output layer
        self.fc_out = nn.Linear(d_model, vocab_size)

    def _generate_square_subsequent_mask(self, sz):
        """Generates a causal mask for the decoder."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward_features(self, src, tgt):
        """
        Processes inputs through the transformer encoder-decoder stack,
        returning the raw feature output before the final projection.
        """
        # --- Prepare Inputs ---
        src = self.audio_input_projection(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)

        tgt_emb = self.token_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)

        # --- Create Masks ---
        tgt_mask = self._generate_square_subsequent_mask(tgt.size(1)).to(src.device)

        # --- Transformer Pass ---
        output = self.transformer(src, tgt_emb, tgt_mask=tgt_mask)
        return output

    def forward(self, src, tgt):
        """
        Full forward pass including the final output projection.
        """
        features = self.forward_features(src, tgt)
        return self.fc_out(features)


class PatternAwareTransformer(TaikoTransformer):
    """
    Enhanced transformer with an explicit pattern memory module. It refines
    the standard transformer's output by attending to a learned set of
    meaningful musical patterns.
    """
    def __init__(self, *args, **kwargs):
        num_heads = kwargs.pop('nhead', 8)
        super().__init__(*args, **kwargs)

        self.pattern_memory = nn.Parameter(torch.randn(512, self.d_model))
        self.pattern_attention = nn.MultiheadAttention(
            self.d_model,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )

    def forward(self, src, tgt, return_attention=False):
        """
        Forward pass that incorporates the pattern-aware refinement step.

        Args:
            src (torch.Tensor): The source audio features.
            tgt (torch.Tensor): The target token sequence.
            return_attention (bool): If True, returns attention weights alongside the output.

        Returns:
            torch.Tensor or (torch.Tensor, torch.Tensor): The output logits, and optionally
                                                          the pattern attention weights.
        """
        base_output = self.forward_features(src, tgt)

        # The key difference is capturing the attention weights (attn_weights)
        pattern_context, attn_weights = self.pattern_attention(
            query=base_output,
            key=self.pattern_memory.unsqueeze(0).repeat(base_output.size(0), 1, 1),
            value=self.pattern_memory.unsqueeze(0).repeat(base_output.size(0), 1, 1)
        )

        refined_output = base_output + pattern_context
        output_logits = self.fc_out(refined_output)

        if return_attention:
            return output_logits, attn_weights
        else:
            return output_logits