"""GrokkingTransformer model for modular arithmetic experiments."""

import warnings

import torch
import torch.nn as nn


class GrokkingTransformer(nn.Module):
    def __init__(self, p, d_model, nhead, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(p, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(2, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, norm_first=True,
        )
        # norm_first=True intentionally disables nested tensor fast path
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*enable_nested_tensor is True.*")
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, p)

    def forward(self, x):
        emb = self.embedding(x) + self.pos_embedding
        out = self.transformer(emb)
        out = out.mean(dim=1)
        logits = self.decoder(out)
        return logits
