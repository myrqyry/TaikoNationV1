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

    def forward(self, src, tgt):
        """
        Forward pass of the model.

        Args:
            src (torch.Tensor): The audio features (encoder input).
                                Shape: [batch_size, seq_len, audio_feature_size]
            tgt (torch.Tensor): The note tokens (decoder input).
                                Shape: [batch_size, seq_len]

        Returns:
            torch.Tensor: The output logits over the vocabulary.
                          Shape: [batch_size, seq_len, vocab_size]
        """
        # --- Prepare Inputs ---
        # Project audio features to the model dimension
        src = self.audio_input_projection(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)

        # Embed and positionally encode the target tokens
        tgt = self.token_embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)

        # --- Create Masks ---
        # The decoder needs a causal mask to prevent it from seeing future tokens.
        tgt_mask = self._generate_square_subsequent_mask(tgt.size(1)).to(src.device)

        # The encoder and decoder also need padding masks if we were to support variable length sequences.
        # For now, since our dataset pads everything to max_sequence_length, we can omit them for simplicity.
        # src_key_padding_mask and tgt_key_padding_mask would go here.

        # --- Transformer Pass ---
        output = self.transformer(src, tgt, tgt_mask=tgt_mask)

        # --- Final Output ---
        return self.fc_out(output)
