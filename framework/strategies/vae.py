"""Variational autoencoder training strategy.

ELBO loss = reconstruction_loss + beta * KL_divergence, with optional
beta annealing (beta-VAE). The model architecture is the runner's
responsibility; the strategy handles loss decomposition and KL computation.

Uses existing BackpropComponents. The runner provides a recon_loss_fn via
kwargs and optionally a vae_forward_fn to unpack model output.
"""

from __future__ import annotations

from typing import Any

import torch

from .strategy_runner import StrategyRunner, StepResult


class VAETrainStep(StrategyRunner):
    """VAE training with ELBO loss and optional beta scheduling.

    Setup expects:
        - kwargs['recon_loss_fn']: fn(recon, batch) -> reconstruction loss
        - kwargs['beta']: KL weight (default 1.0, set < 1 for beta-VAE)
        - kwargs['beta_schedule_fn']: optional fn(step) -> beta_t for annealing
        - kwargs['vae_forward_fn']: optional fn(model, batch) -> (recon, mu, logvar).
            When None, calls model(batch) and expects a 3-tuple.
    """

    def setup(self, *, components, config, device, **kwargs) -> None:
        self.model = components.model
        self.optimizer = components.optimizer
        self.device = device
        self._components = components
        self._data_iter = None
        self.data = components.data

        # Reconstruction loss (required)
        self.recon_loss_fn = kwargs.get('recon_loss_fn')
        if self.recon_loss_fn is None:
            raise ValueError(
                "VAETrainStep requires a recon_loss_fn. "
                "Provide via kwargs['recon_loss_fn'] as fn(recon, batch) -> loss."
            )

        # VAE hyperparameters
        self.beta = kwargs.get('beta', 1.0)
        self.beta_schedule_fn = kwargs.get('beta_schedule_fn')
        self.vae_forward_fn = kwargs.get('vae_forward_fn')

        # Accumulation + AMP
        self._accumulation_steps = getattr(config, 'accumulation_steps', 1)
        self._accum_count = 0
        self._scaler = components.grad_scaler
        self._use_amp = self._scaler is not None
        self._autocast_device = device.type if device.type in ('cuda', 'cpu') else 'cpu'

    def _forward(self, batch):
        """Forward pass through VAE, returning (recon, mu, logvar)."""
        if self.vae_forward_fn is not None:
            return self.vae_forward_fn(self.model, batch)
        output = self.model(batch)
        if isinstance(output, tuple) and len(output) == 3:
            return output
        raise ValueError(
            "VAETrainStep: model(batch) must return a 3-tuple (recon, mu, logvar), "
            "or provide a vae_forward_fn."
        )

    @staticmethod
    def _kl_divergence(mu, logvar):
        """Closed-form KL divergence for diagonal Gaussian vs N(0,I)."""
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    def train_step(self, step: int, batch: Any = None) -> StepResult:
        if batch is None:
            if self.data is None:
                raise ValueError("VAETrainStep: no batch provided and no data source set")
            if self._data_iter is None:
                self._data_iter = iter(self.data)
            try:
                batch = next(self._data_iter)
            except StopIteration:
                return StepResult(metrics={'loss': 0.0}, trained=False, should_stop=True)

        if self._accum_count == 0:
            self.optimizer.zero_grad()

        with torch.autocast(device_type=self._autocast_device, enabled=self._use_amp):
            recon, mu, logvar = self._forward(batch)
            recon_loss = self.recon_loss_fn(recon, batch)
            kl_loss = self._kl_divergence(mu, logvar)
            current_beta = self.beta_schedule_fn(step) if self.beta_schedule_fn else self.beta
            loss = recon_loss + current_beta * kl_loss

        scaled_loss = loss / self._accumulation_steps
        if self._scaler:
            self._scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        self._accum_count += 1
        if self._accum_count >= self._accumulation_steps:
            if self._scaler:
                self._scaler.unscale_(self.optimizer)
            self._components.clip_gradients()
            if self._scaler:
                self._scaler.step(self.optimizer)
                self._scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()
            self._accum_count = 0
            trained = True
        else:
            trained = False

        return StepResult(
            metrics={
                'loss': loss.detach(),
                'recon_loss': recon_loss.detach(),
                'kl_loss': kl_loss.detach(),
                'beta': current_beta,
            },
            trained=trained,
            batch_data=batch,
        )

    @property
    def name(self) -> str:
        return "VAETrainStep"
