"""Shared fixtures and helpers for strategy tests."""

import torch
import torch.nn as nn
from torch.optim import SGD

from framework.trainers.components import BackpropComponents


def make_backprop_components(model, loss_fn=None, data=None, auxiliary_models=None):
    """Build BackpropComponents for a given model."""
    optimizer = SGD(model.parameters(), lr=0.01)
    if loss_fn is None:
        loss_fn = lambda m, b: m(b).sum()
    return BackpropComponents(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        criterion=None,
        loss_fn=loss_fn,
        strategy=None,
        data=data,
        auxiliary_models=auxiliary_models,
    )


class TinyVAE(nn.Module):
    """Minimal VAE: encoder -> (mu, logvar), reparameterize, decode."""

    def __init__(self, in_dim=4, latent_dim=2):
        super().__init__()
        self.encoder = nn.Linear(in_dim, latent_dim * 2)
        self.decoder = nn.Linear(latent_dim, in_dim)

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=-1)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        recon = self.decoder(z)
        return recon, mu, logvar


class TinyDenoiser(nn.Module):
    """Accepts (x, timestep) and predicts noise."""

    def __init__(self, dim=4, num_timesteps=1000):
        super().__init__()
        self.time_embed = nn.Embedding(num_timesteps, dim)
        self.net = nn.Linear(dim * 2, dim)

    def forward(self, x, t):
        t_emb = self.time_embed(t)
        return self.net(torch.cat([x, t_emb], dim=-1))
