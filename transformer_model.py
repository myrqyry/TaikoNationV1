import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class CustomTransformerDecoderLayer(nn.Module):
    """A custom decoder layer that returns both self-attention and cross-attention weights."""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, batch_first=True):
        super(CustomTransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None):
        # Self-attention block
        tgt2, self_attn_weights = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, need_weights=True)
        tgt = self.norm1(tgt + self.dropout1(tgt2))

        # Cross-attention block
        tgt2, cross_attn_weights = self.multihead_attn(tgt, memory, memory, need_weights=True)
        tgt = self.norm2(tgt + self.dropout2(tgt2))

        # Feed-forward block
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout3(tgt2))

        return tgt, self_attn_weights, cross_attn_weights

class PositionalEncoding(nn.Module):
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
        return x + self.pe[:, :x.size(1)]

class TaikoTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 audio_feature_size=80):
        super(TaikoTransformer, self).__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.audio_input_projection = nn.Linear(audio_feature_size, d_model)

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = CustomTransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.decoder = nn.ModuleList([decoder_layer for _ in range(num_decoder_layers)])

        self.fc_out = nn.Linear(d_model, vocab_size)
        self.value_head = nn.Linear(d_model, 1)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tgt, return_attention=False):
        src_embed = self.pos_encoder(self.audio_input_projection(src) * math.sqrt(self.d_model))
        memory = self.encoder(src_embed)

        tgt_embed = self.pos_encoder(self.token_embedding(tgt) * math.sqrt(self.d_model))

        output = tgt_embed
        self_attns, cross_attns = [], []

        for layer in self.decoder:
            output, self_attn, cross_attn = layer(output, memory, tgt_mask=self._generate_square_subsequent_mask(tgt.size(1)).to(src.device))
            if return_attention:
                self_attns.append(self_attn)
                cross_attns.append(cross_attn)

        result = {
            "logits": self.fc_out(output),
            "value": self.value_head(output)
        }
        if return_attention:
            result["self_attention"] = self_attns[-1]
            result["cross_attention"] = cross_attns[-1]

        return result

    def generate(self, src, max_len, tokenizer, return_attention=False):
        self.eval()
        device = src.device
        batch_size = src.size(0)
        memory = self.encoder(self.pos_encoder(self.audio_input_projection(src) * math.sqrt(self.d_model)))
        sequences = torch.full((batch_size, 1), tokenizer.vocab["[CLS]"], dtype=torch.long, device=device)

        final_self_attn, final_cross_attn = None, None

        for i in range(max_len):
            tgt_embedded = self.pos_encoder(self.token_embedding(sequences) * math.sqrt(self.d_model))
            output = tgt_embedded
            for layer in self.decoder:
                output, self_attn, cross_attn = layer(output, memory, tgt_mask=self._generate_square_subsequent_mask(sequences.size(1)).to(device))

            if i == max_len - 1 and return_attention:
                final_self_attn = self_attn
                final_cross_attn = cross_attn

            last_token_logits = self.fc_out(output[:, -1, :])
            next_token = torch.multinomial(F.softmax(last_token_logits, dim=-1), num_samples=1)
            sequences = torch.cat([sequences, next_token], dim=1)

        result = {"sequences": sequences[:, 1:]}
        if return_attention:
            result["self_attention"] = final_self_attn
            result["cross_attention"] = final_cross_attn

        return result