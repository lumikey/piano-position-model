"""
Hand position prediction Transformer model.

Input: 21 tokens x 4 features
  - Previous hand (6 tokens): wrist + 5 fingertips (x, y, part_id, token_type)
  - Current notes (5 tokens): one per finger slot (midi_norm, black_key, is_active, token_type)
  - Lookahead (10 tokens): next notes with fingering (midi_norm, black_key, finger_norm, time_until)

Output: 12 values (wrist x,y + 5 fingertips x,y)
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class HandPositionTransformer(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=3, dim_feedforward=128, dropout=0.1):
        super().__init__()

        # Input: 4 features per token
        self.input_proj = nn.Linear(4, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output heads: one for wrist, one for each fingertip
        self.wrist_head = nn.Linear(d_model, 2)
        self.finger_heads = nn.ModuleList([nn.Linear(d_model, 2) for _ in range(5)])

        self.d_model = d_model

    def forward(self, tokens, mask=None):
        """
        tokens: (batch, 21, 4)
        Returns: (batch, 12) - wrist(2) + fingertips(10)
        """
        x = self.input_proj(tokens)
        x = self.pos_encoder(x)
        x = self.transformer(x, src_key_padding_mask=mask)

        # Use the hand position tokens (0-5) to predict output
        wrist_out = self.wrist_head(x[:, 0, :])  # (batch, 2)

        finger_outs = []
        for i, head in enumerate(self.finger_heads):
            finger_outs.append(head(x[:, 1 + i, :]))  # (batch, 2)

        # Concatenate: wrist(2) + 5 fingers(10) = 12
        output = torch.cat([wrist_out] + finger_outs, dim=1)
        return output
