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


class MultiTaskTaikoTransformer(PatternAwareTransformer):
    """
    Extends PatternAwareTransformer to use a difficulty-aware pattern memory.
    """
    def __init__(self, num_difficulty_classes=5, patterns_per_diff=200, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.difficulty_embedding = nn.Embedding(num_difficulty_classes, self.d_model)
        self.difficulty_head = nn.Linear(self.d_model, num_difficulty_classes)

        # Replace the single pattern memory with a difficulty-specific one
        self.pattern_memory = nn.Parameter(
            torch.randn(num_difficulty_classes, patterns_per_diff, self.d_model)
        )

    def forward(self, src, tgt, target_difficulty=None, return_attention=False):
        """
        Forward pass for the multi-task model.
        """
        base_output = self.forward_features(src, tgt)

        # Add difficulty conditioning
        if target_difficulty is not None:
            difficulty_emb = self.difficulty_embedding(target_difficulty)
            base_output = base_output + difficulty_emb.unsqueeze(1)

        # Select the appropriate pattern memory based on the target difficulty
        if target_difficulty is not None:
            # Select the pattern memory for each item in the batch
            relevant_patterns = torch.index_select(self.pattern_memory, 0, target_difficulty)
        else:
            # Default to the first difficulty bank if none is specified
            relevant_patterns = self.pattern_memory[0].unsqueeze(0).repeat(base_output.size(0), 1, 1)

        pattern_context, attn_weights = self.pattern_attention(
            query=base_output,
            key=relevant_patterns,
            value=relevant_patterns
        )

        refined_output = base_output + pattern_context
        token_logits = self.fc_out(refined_output)

        pooled_output = refined_output.mean(dim=1)
        difficulty_logits = self.difficulty_head(pooled_output)

        outputs = {
            'tokens': token_logits,
            'difficulty': difficulty_logits
        }

        if return_attention:
            return outputs, attn_weights
        else:
            return outputs