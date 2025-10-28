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
    def __init__(self, vocab_size, num_genres, num_difficulties, d_model=256, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 audio_feature_size=80, max_sequence_length=512):
        super(TaikoTransformer, self).__init__()
        self.d_model = d_model

        # --- Layers ---
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.genre_embedding = nn.Embedding(num_genres, d_model)
        self.difficulty_embedding = nn.Embedding(num_difficulties, d_model)
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

        self._compiled = False

    def compile_model_if_needed(self):
        """Compile model once for optimization"""
        if not self._compiled:
            self.forward = torch.compile(self.forward, mode="max-autotune")
            self._compiled = True

    def _generate_square_subsequent_mask(self, sz):
        """Generates a causal mask for the decoder."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tgt, genre_id, difficulty_id):
        """
        Forward pass of the model.
        Args:
            src (torch.Tensor): The audio features (encoder input).
            tgt (torch.Tensor): The note tokens (decoder input).
            genre_id (torch.Tensor): The ID of the genre for style conditioning.
            difficulty_id (torch.Tensor): The ID of the difficulty.
        Returns:
            torch.Tensor: The output logits over the vocabulary.
        """
        # --- Prepare Inputs ---
        # Expected shapes (batch_first=True):
        #  src: (batch, seq_len_src, audio_feature_size)
        #  tgt: (batch, seq_len_tgt)
        #  genre_id, difficulty_id: (batch,)
        # We add light assertions to help catch shape mismatches during development.
        if src.dim() != 3:
            raise ValueError(f"Expected src to be 3D (batch,seq,feat); got shape {tuple(src.shape)}")
        if tgt.dim() != 2:
            raise ValueError(f"Expected tgt to be 2D (batch,seq); got shape {tuple(tgt.shape)}")
        if genre_id.dim() not in (1,):
            # allow scalar genre_id in eval but prefer vector in training
            raise ValueError(f"Expected genre_id to be 1D (batch,); got shape {tuple(genre_id.shape)}")

        src = self.audio_input_projection(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)

        # Embed tokens, add genre and difficulty styles, and add positional encoding
        tgt_embed = self.token_embedding(tgt) * math.sqrt(self.d_model)
        # Expand genre/difficulty embeddings to match tgt token embeddings
        genre_embed = self.genre_embedding(genre_id).unsqueeze(1).expand(-1, tgt_embed.size(1), -1)
        difficulty_embed = self.difficulty_embedding(difficulty_id).unsqueeze(1).expand(-1, tgt_embed.size(1), -1)
        tgt = self.pos_encoder(tgt_embed + genre_embed + difficulty_embed)

        # --- Create Masks ---
        # The decoder needs a causal mask to prevent it from seeing future tokens.
        tgt_mask = self._generate_square_subsequent_mask(tgt.size(1)).to(src.device)

        # The encoder and decoder also need padding masks if we were to support variable length sequences.
        # For now, since our dataset pads everything to max_sequence_length, we can omit them for simplicity.
        # src_key_padding_mask and tgt_key_padding_mask would go here.

        # --- Transformer Pass ---
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
            output = self.transformer(src, tgt, tgt_mask=tgt_mask)

        # --- Final Output ---
        return self.fc_out(output)
