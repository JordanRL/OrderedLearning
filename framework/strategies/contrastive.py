"""Contrastive and self-supervised learning strategies.

Provides SimCLR-style contrastive learning (NT-Xent loss) and momentum
encoder variants (BYOL/MoCo pattern). Both use standard BackpropComponents
with experiment-provided augmentation functions.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .strategy_runner import StrategyRunner, StepResult


class ContrastiveTrainStep(StrategyRunner):
    """SimCLR-style contrastive learning with NT-Xent loss.

    Each step:
    1. Apply two augmentations to each sample → (view1, view2)
    2. Forward both views through encoder + projection head
    3. Compute NT-Xent loss across the batch
    4. Backward and step

    Setup expects:
        - components.model: encoder (backbone)
        - kwargs['projection_head']: projection MLP (nn.Module)
        - kwargs['augment_fn']: callable(batch) -> (view1, view2)
        - kwargs['temperature']: NT-Xent temperature (default 0.07)
    """

    def setup(self, *, components, config, device, **kwargs) -> None:
        self.model = components.model       # encoder
        self.optimizer = components.optimizer
        self.device = device
        self._components = components
        self._data_iter = None
        self.data = components.data

        self.projection_head = kwargs.get('projection_head')
        if self.projection_head is None:
            raise ValueError(
                "ContrastiveTrainStep requires a projection_head. "
                "Provide via kwargs['projection_head']."
            )

        self.augment_fn = kwargs.get('augment_fn')
        if self.augment_fn is None:
            raise ValueError(
                "ContrastiveTrainStep requires an augment_fn. "
                "Provide via kwargs['augment_fn']."
            )

        self.temperature = kwargs.get('temperature', 0.07)

        # Accumulation + AMP
        self._accumulation_steps = getattr(config, 'accumulation_steps', 1)
        self._accum_count = 0
        self._scaler = components.grad_scaler
        self._use_amp = self._scaler is not None
        self._autocast_device = device.type if device.type in ('cuda', 'cpu') else 'cpu'

    def _encode(self, x):
        """Forward through encoder + projection head."""
        h = self.model(x)
        # Handle models that return objects with a specific attribute
        if hasattr(h, 'last_hidden_state'):
            h = h.last_hidden_state[:, 0]  # CLS token
        elif hasattr(h, 'logits'):
            h = h.logits
        z = self.projection_head(h)
        return F.normalize(z, dim=-1)

    def _nt_xent_loss(self, z1, z2):
        """NT-Xent (normalized temperature-scaled cross entropy) loss."""
        batch_size = z1.shape[0]
        z = torch.cat([z1, z2], dim=0)  # (2N, D)
        sim = torch.mm(z, z.t()) / self.temperature  # (2N, 2N)

        # Mask out self-similarity
        mask = ~torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(~mask, float('-inf'))

        # Positive pairs: (i, i+N) and (i+N, i)
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=z.device),
            torch.arange(0, batch_size, device=z.device),
        ])

        return F.cross_entropy(sim, labels)

    def train_step(self, step: int, batch: Any = None) -> StepResult:
        if batch is None:
            if self.data is None:
                raise ValueError("ContrastiveTrainStep: no batch and no data source")
            if self._data_iter is None:
                self._data_iter = iter(self.data)
            try:
                batch = next(self._data_iter)
            except StopIteration:
                return StepResult(metrics={'loss': 0.0}, trained=False, should_stop=True)

        if self._accum_count == 0:
            self.optimizer.zero_grad()

        with torch.autocast(device_type=self._autocast_device, enabled=self._use_amp):
            view1, view2 = self.augment_fn(batch)
            z1 = self._encode(view1)
            z2 = self._encode(view2)
            loss = self._nt_xent_loss(z1, z2)

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
            metrics={'loss': loss.detach()},
            trained=trained,
            batch_data=batch,
        )

    @property
    def name(self) -> str:
        return "ContrastiveTrainStep"


class MomentumContrastiveTrainStep(ContrastiveTrainStep):
    """Momentum encoder variant (BYOL/MoCo pattern).

    Extends ContrastiveTrainStep with an EMA-updated momentum encoder.
    The online encoder is trained with gradients; the momentum encoder
    is updated via exponential moving average after each optimizer step.

    Setup expects (in addition to ContrastiveTrainStep requirements):
        - components.auxiliary_models['momentum_encoder']: EMA copy of encoder
        - kwargs['momentum_projection_head']: projection head for momentum encoder
        - kwargs['ema_decay']: EMA decay rate (default 0.996)
    """

    def setup(self, *, components, config, device, **kwargs) -> None:
        super().setup(components=components, config=config, device=device, **kwargs)

        self.momentum_encoder = components.auxiliary_models.get('momentum_encoder')
        if self.momentum_encoder is None:
            raise ValueError(
                "MomentumContrastiveTrainStep requires auxiliary_models['momentum_encoder']."
            )
        self.momentum_encoder.eval()

        self.momentum_projection_head = kwargs.get('momentum_projection_head')
        if self.momentum_projection_head is None:
            raise ValueError(
                "MomentumContrastiveTrainStep requires kwargs['momentum_projection_head']."
            )
        self.momentum_projection_head.eval()

        self.ema_decay = kwargs.get('ema_decay', 0.996)

    def _encode_momentum(self, x):
        """Forward through momentum encoder + momentum projection head (no grad)."""
        with torch.no_grad():
            h = self.momentum_encoder(x)
            if hasattr(h, 'last_hidden_state'):
                h = h.last_hidden_state[:, 0]
            elif hasattr(h, 'logits'):
                h = h.logits
            z = self.momentum_projection_head(h)
            return F.normalize(z, dim=-1)

    @torch.no_grad()
    def _update_momentum(self):
        """EMA update: momentum_param = decay * momentum_param + (1-decay) * online_param."""
        for online_p, momentum_p in zip(
            self.model.parameters(), self.momentum_encoder.parameters()
        ):
            momentum_p.data.mul_(self.ema_decay).add_(
                online_p.data, alpha=1.0 - self.ema_decay
            )
        for online_p, momentum_p in zip(
            self.projection_head.parameters(),
            self.momentum_projection_head.parameters(),
        ):
            momentum_p.data.mul_(self.ema_decay).add_(
                online_p.data, alpha=1.0 - self.ema_decay
            )

    def train_step(self, step: int, batch: Any = None) -> StepResult:
        if batch is None:
            if self.data is None:
                raise ValueError("MomentumContrastiveTrainStep: no batch and no data source")
            if self._data_iter is None:
                self._data_iter = iter(self.data)
            try:
                batch = next(self._data_iter)
            except StopIteration:
                return StepResult(metrics={'loss': 0.0}, trained=False, should_stop=True)

        if self._accum_count == 0:
            self.optimizer.zero_grad()

        with torch.autocast(device_type=self._autocast_device, enabled=self._use_amp):
            view1, view2 = self.augment_fn(batch)
            # Online encoder (with grad)
            z1 = self._encode(view1)
            # Momentum encoder (no grad)
            z2 = self._encode_momentum(view2)
            loss = self._nt_xent_loss(z1, z2)

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
            # EMA update after optimizer step
            self._update_momentum()
            trained = True
        else:
            trained = False

        return StepResult(
            metrics={'loss': loss.detach()},
            trained=trained,
            batch_data=batch,
        )

    @property
    def name(self) -> str:
        return "MomentumContrastiveTrainStep"
