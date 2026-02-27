"""Adversarial training strategies (GANs).

Alternating generator/discriminator training with configurable D steps
per G step. Loss functions are experiment-provided.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.autograd as autograd

from .strategy_runner import StrategyRunner, StepResult
from ..capabilities import TrainingParadigm


class AdversarialTrainStep(StrategyRunner):
    """Standard GAN training: alternating D and G updates.

    Each step:
    1. Train discriminator for d_steps_per_g_step batches
    2. Train generator for 1 batch

    Setup expects:
        - components: AdversarialComponents
        - components.g_loss_fn(G, D, real_batch) -> generator loss
        - components.d_loss_fn(G, D, real_batch) -> discriminator loss
    """

    paradigm = TrainingParadigm.ADVERSARIAL

    def setup(self, *, components, config, device, **kwargs) -> None:
        self.generator = components.generator
        self.discriminator = components.discriminator
        self.g_optimizer = components.g_optimizer
        self.d_optimizer = components.d_optimizer
        self._components = components
        self.device = device
        self.data = components.data
        self._data_iter = None

        self.g_loss_fn = components.g_loss_fn
        self.d_loss_fn = components.d_loss_fn
        if self.g_loss_fn is None or self.d_loss_fn is None:
            raise ValueError(
                "AdversarialTrainStep requires both g_loss_fn and d_loss_fn."
            )

        self.d_steps_per_g_step = components.d_steps_per_g_step

        # AMP
        self._scaler = components.grad_scaler
        self._use_amp = self._scaler is not None
        self._autocast_device = device.type if device.type in ('cuda', 'cpu') else 'cpu'

    def _get_batch(self):
        """Get next batch from data iterator."""
        if self._data_iter is None:
            self._data_iter = iter(self.data)
        try:
            return next(self._data_iter)
        except StopIteration:
            self._data_iter = iter(self.data)
            return next(self._data_iter)

    def _train_discriminator(self, batch) -> torch.Tensor:
        """One discriminator training step."""
        self.d_optimizer.zero_grad()
        with torch.autocast(device_type=self._autocast_device, enabled=self._use_amp):
            d_loss = self.d_loss_fn(self.generator, self.discriminator, batch)
        if self._scaler:
            self._scaler.scale(d_loss).backward()
            self._scaler.unscale_(self.d_optimizer)
            if self._components.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.discriminator.parameters(), self._components.max_grad_norm
                )
            self._scaler.step(self.d_optimizer)
            self._scaler.update()
        else:
            d_loss.backward()
            if self._components.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.discriminator.parameters(), self._components.max_grad_norm
                )
            self.d_optimizer.step()
        return d_loss.detach()

    def _train_generator(self, batch) -> torch.Tensor:
        """One generator training step."""
        self.g_optimizer.zero_grad()
        with torch.autocast(device_type=self._autocast_device, enabled=self._use_amp):
            g_loss = self.g_loss_fn(self.generator, self.discriminator, batch)
        if self._scaler:
            self._scaler.scale(g_loss).backward()
            self._scaler.unscale_(self.g_optimizer)
            if self._components.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.generator.parameters(), self._components.max_grad_norm
                )
            self._scaler.step(self.g_optimizer)
            self._scaler.update()
        else:
            g_loss.backward()
            if self._components.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.generator.parameters(), self._components.max_grad_norm
                )
            self.g_optimizer.step()
        return g_loss.detach()

    def train_step(self, step: int, batch: Any = None) -> StepResult:
        if batch is None:
            batch = self._get_batch()

        # Discriminator steps
        d_loss = None
        for _ in range(self.d_steps_per_g_step):
            d_batch = self._get_batch() if self.d_steps_per_g_step > 1 else batch
            d_loss = self._train_discriminator(d_batch)

        # Generator step
        g_loss = self._train_generator(batch)

        return StepResult(
            metrics={
                'loss': g_loss,
                'g_loss': g_loss,
                'd_loss': d_loss,
            },
            batch_data=batch,
        )

    @property
    def name(self) -> str:
        return "AdversarialTrainStep"


class WGANGPTrainStep(AdversarialTrainStep):
    """Wasserstein GAN with gradient penalty.

    Adds gradient penalty to discriminator loss. The penalty enforces
    the Lipschitz constraint by penalizing gradients of the discriminator
    with respect to interpolated inputs.

    Setup expects (in addition to AdversarialTrainStep):
        - kwargs['gp_weight']: gradient penalty coefficient (default 10.0)
        - kwargs['interpolate_fn']: callable(real, fake) -> interpolated
          (default: random interpolation between real and fake)
    """

    def setup(self, *, components, config, device, **kwargs) -> None:
        super().setup(components=components, config=config, device=device, **kwargs)
        self.gp_weight = kwargs.get('gp_weight', 10.0)
        self.interpolate_fn = kwargs.get('interpolate_fn', self._default_interpolate)

    @staticmethod
    def _default_interpolate(real, fake):
        """Random interpolation between real and fake samples."""
        alpha = torch.rand(real.size(0), *([1] * (real.dim() - 1)),
                           device=real.device, dtype=real.dtype)
        return (alpha * real + (1 - alpha) * fake).requires_grad_(True)

    def _gradient_penalty(self, real, fake):
        """Compute gradient penalty on interpolated samples."""
        interpolated = self.interpolate_fn(real, fake)
        d_interpolated = self.discriminator(interpolated)

        if isinstance(d_interpolated, tuple):
            d_interpolated = d_interpolated[0]

        gradients = autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_norm = gradients.norm(2, dim=1)
        return ((gradient_norm - 1) ** 2).mean()

    def _train_discriminator(self, batch) -> torch.Tensor:
        """Discriminator step with gradient penalty."""
        self.d_optimizer.zero_grad()
        with torch.autocast(device_type=self._autocast_device, enabled=self._use_amp):
            d_loss = self.d_loss_fn(self.generator, self.discriminator, batch)

        # GP must be computed in fp32 for stability
        with torch.no_grad():
            noise = torch.randn(batch.size(0), self.generator.latent_dim
                                if hasattr(self.generator, 'latent_dim')
                                else batch.size(-1),
                                device=self.device)
            fake = self.generator(noise)
        gp = self._gradient_penalty(batch, fake.detach())
        total_loss = d_loss + self.gp_weight * gp

        if self._scaler:
            self._scaler.scale(total_loss).backward()
            self._scaler.unscale_(self.d_optimizer)
            if self._components.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.discriminator.parameters(), self._components.max_grad_norm
                )
            self._scaler.step(self.d_optimizer)
            self._scaler.update()
        else:
            total_loss.backward()
            if self._components.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.discriminator.parameters(), self._components.max_grad_norm
                )
            self.d_optimizer.step()

        return d_loss.detach()

    @property
    def name(self) -> str:
        return "WGANGPTrainStep"
