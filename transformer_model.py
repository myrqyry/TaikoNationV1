import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class CustomTransformerDecoderLayer(nn.Module):
    """
    A custom TransformerDecoderLayer that returns the cross-attention weights.
    """
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

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = self.norm1(tgt + self.dropout1(tgt2))
        tgt2, attn_weights = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                                 key_padding_mask=memory_key_padding_mask,
                                                 need_weights=True)
        tgt = self.norm2(tgt + self.dropout2(tgt2))
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout3(tgt2))
        return tgt, attn_weights

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

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tgt, return_attention=False):
        src_proj = self.audio_input_projection(src) * math.sqrt(self.d_model)
        src_embed = self.pos_encoder(src_proj)
        memory = self.encoder(src_embed)

        tgt_embed = self.token_embedding(tgt) * math.sqrt(self.d_model)
        tgt_embed = self.pos_encoder(tgt_embed)

        output = tgt_embed
        attention_maps = []

        for layer in self.decoder:
            output, attn = layer(output, memory, tgt_mask=self._generate_square_subsequent_mask(tgt.size(1)).to(src.device))
            if return_attention:
                attention_maps.append(attn)

        output = self.fc_out(output)
        if return_attention:
            return output, attention_maps[-1] # Return logits and last layer's attention
        return output

    def generate(self, src, max_len, tokenizer, temperature=1.0, return_attention=False):
        self.eval()
        device = src.device
        batch_size = src.size(0)

        src_proj = self.audio_input_projection(src) * math.sqrt(self.d_model)
        memory = self.encoder(self.pos_encoder(src_proj))

        start_token_id = tokenizer.vocab["[CLS]"]
        sequences = torch.full((batch_size, 1), start_token_id, dtype=torch.long, device=device)

        final_attention = None

        for i in range(max_len):
            tgt_embedded = self.pos_encoder(self.token_embedding(sequences) * math.sqrt(self.d_model))

            output = tgt_embedded
            for layer in self.decoder:
                output, attn = layer(output, memory, tgt_mask=self._generate_square_subsequent_mask(sequences.size(1)).to(device))

            # We only need the attention from the final step for this visualization
            if i == max_len - 1 and return_attention:
                final_attention = attn

            last_token_logits = self.fc_out(output[:, -1, :])
            probs = F.softmax(last_token_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            sequences = torch.cat([sequences, next_token], dim=1)

        result = {"sequences": sequences[:, 1:]}
        if return_attention:
            result["attention"] = final_attention

        return result