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
                 audio_feature_size=80):
        super(TaikoTransformer, self).__init__()
        self.d_model = d_model

        # --- Layers ---
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

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
        # Pop 'num_heads' from kwargs for pattern_attention, if it exists,
        # or default to 8. This avoids passing it to TaikoTransformer's __init__.
        num_heads = kwargs.pop('nhead', 8)
        super().__init__(*args, **kwargs)

        # --- Pattern Modeling Components ---
        # A learnable memory bank of common patterns
        self.pattern_memory = nn.Parameter(torch.randn(512, self.d_model))

        # Attention mechanism to query the pattern memory
        self.pattern_attention = nn.MultiheadAttention(
            self.d_model,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )

    def forward(self, src, tgt):
        """
        Forward pass that incorporates the pattern-aware refinement step.
        """
        # 1. Get the base features from the standard transformer layers
        base_output = self.forward_features(src, tgt)

        # 2. Refine features with pattern memory
        # Query: The transformer's output (what the model is currently thinking)
        # Key/Value: The learnable pattern memory (the library of patterns)
        pattern_context, _ = self.pattern_attention(
            query=base_output,
            key=self.pattern_memory.unsqueeze(0).repeat(base_output.size(0), 1, 1),
            value=self.pattern_memory.unsqueeze(0).repeat(base_output.size(0), 1, 1)
        )

        # 3. Add the pattern context back to the base output
        refined_output = base_output + pattern_context

        # 4. Project the refined features to the vocabulary space
        return self.fc_out(refined_output)


class MultiTaskTaikoTransformer(PatternAwareTransformer):
    """
    Extends the PatternAwareTransformer to perform multi-task learning.
    In addition to generating note sequences, it predicts auxiliary targets
    like tempo and difficulty, which helps the model build a richer internal
    representation of the music.
    """
    def __init__(self, num_difficulty_classes=5, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # --- Auxiliary Task Heads ---
        # A head for tempo regression (predicting a single continuous value)
        self.tempo_head = nn.Linear(self.d_model, 1)

        # A head for difficulty classification (predicting logits for each class)
        self.difficulty_head = nn.Linear(self.d_model, num_difficulty_classes)

    def forward(self, src, tgt):
        """
        Forward pass that returns a dictionary of outputs for each task.
        """
        # 1. Get the base features from the standard transformer layers
        base_output = self.forward_features(src, tgt)

        # 2. Refine features with pattern memory
        pattern_context, _ = self.pattern_attention(
            query=base_output,
            key=self.pattern_memory.unsqueeze(0).repeat(base_output.size(0), 1, 1),
            value=self.pattern_memory.unsqueeze(0).repeat(base_output.size(0), 1, 1)
        )
        refined_output = base_output + pattern_context

        # 3. Main task: Project features to vocabulary space for token prediction
        token_logits = self.fc_out(refined_output)

        # 4. Auxiliary tasks: Use a pooled representation of the features.
        # We take the mean of the features across the sequence length to get a
        # single vector representing the entire sequence.
        pooled_output = refined_output.mean(dim=1)
        tempo_pred = self.tempo_head(pooled_output)
        difficulty_pred = self.difficulty_head(pooled_output)

        # 5. Return all outputs in a dictionary for flexible loss calculation
        return {
            'tokens': token_logits,
            'tempo': tempo_pred,
            'difficulty': difficulty_pred
        }
