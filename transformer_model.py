import torch
import torch.nn as nn
import math
import torch.nn.functional as F

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
        x = x + self.pe[:, :x.size(1)]
        return x

class TaikoTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 audio_feature_size=80):
        super(TaikoTransformer, self).__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.audio_input_projection = nn.Linear(audio_feature_size, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tgt):
        src_proj = self.audio_input_projection(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src_proj)
        tgt_embed = self.token_embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt_embed)
        tgt_mask = self._generate_square_subsequent_mask(tgt.size(1)).to(src.device)
        output = self.transformer(src, tgt, tgt_mask=tgt_mask)
        return {"logits": self.fc_out(output), "hidden_states": output}

    def generate(self, src, max_len, tokenizer, temperature=1.0):
        self.eval()
        device = src.device
        batch_size = src.size(0)

        src_proj = self.audio_input_projection(src) * math.sqrt(self.d_model)
        memory = self.transformer.encoder(self.pos_encoder(src_proj))

        start_token_id = tokenizer.vocab["[CLS]"]
        sequences = torch.full((batch_size, 1), start_token_id, dtype=torch.long, device=device)
        log_probs = torch.zeros(batch_size, 0, device=device) # Start with empty log_probs
        hidden_states_list = []

        for i in range(max_len):
            tgt_embedded = self.pos_encoder(self.token_embedding(sequences) * math.sqrt(self.d_model))
            tgt_mask = self._generate_square_subsequent_mask(sequences.size(1)).to(device)

            output = self.transformer.decoder(tgt_embedded, memory, tgt_mask=tgt_mask)
            hidden_states_list.append(output[:, -1, :].unsqueeze(1))

            last_token_logits = self.fc_out(output[:, -1, :])
            probs = F.softmax(last_token_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            token_log_prob = F.log_softmax(last_token_logits, dim=-1)
            next_log_prob = torch.gather(token_log_prob, 1, next_token)

            sequences = torch.cat([sequences, next_token], dim=1)
            log_probs = torch.cat([log_probs, next_log_prob], dim=1)

        # The final generated sequence, excluding the start token
        final_sequences = sequences[:, 1:]
        # Concatenate all hidden states
        final_hidden_states = torch.cat(hidden_states_list, dim=1)

        return {
            "sequences": final_sequences,
            "log_probs": log_probs,
            "hidden_states": final_hidden_states
        }