"""Non-contrastive self-supervised learning strategy.

BYOL / DINO / VICReg pattern: online encoder with predictor, EMA teacher
without predictor, regression or similarity loss between representations.

Follows the MomentumContrastiveTrainStep EMA pattern closely. The key
difference is the loss function: regression/similarity instead of NT-Xent.

Uses existing BackpropComponents with auxiliary_models={'ema_encoder': ...}.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .strategy_runner import StrategyRunner, StepResult


class SelfDistillationTrainStep(StrategyRunner):
    """Self-distillation training (BYOL / DINO / VICReg).

    Each step:
    1. Augment batch into two views
    2. Online path: encoder -> projection -> predictor (with grad)
    3. Target path: EMA encoder -> EMA projection (no grad)
    4. Compute loss between online and target representations
    5. Optionally symmetrize (swap views, compute both directions)
    6. Backward and step
    7. EMA update of target network

    Setup expects:
        - kwargs['projection_head']: nn.Module — online projection head
        - kwargs['predictor']: nn.Module — online predictor (BYOL pattern)
        - kwargs['augment_fn']: fn(batch) -> (view1, view2)
        - kwargs['ema_projection_head']: nn.Module — EMA projection head
        - components.auxiliary_models['ema_encoder']: EMA copy of encoder
        - kwargs['ema_decay']: EMA decay rate (default 0.996)
        - kwargs['loss_type']: 'cosine' | 'mse' | 'vicreg' (default 'cosine')
        - kwargs['symmetrize']: compute both directions (default True)
        - VICReg weights: vicreg_sim_weight (25.0), vicreg_var_weight (25.0),
            vicreg_cov_weight (1.0)
    """

    def setup(self, *, components, config, device, **kwargs) -> None:
        self.model = components.model  # online encoder
        self.optimizer = components.optimizer
        self.device = device
        self._components = components
        self._data_iter = None
        self.data = components.data

        # Required kwargs
        self.projection_head = kwargs.get('projection_head')
        if self.projection_head is None:
            raise ValueError(
                "SelfDistillationTrainStep requires a projection_head. "
                "Provide via kwargs['projection_head']."
            )

        self.predictor = kwargs.get('predictor')
        if self.predictor is None:
            raise ValueError(
                "SelfDistillationTrainStep requires a predictor. "
                "Provide via kwargs['predictor']."
            )

        self.augment_fn = kwargs.get('augment_fn')
        if self.augment_fn is None:
            raise ValueError(
                "SelfDistillationTrainStep requires an augment_fn. "
                "Provide via kwargs['augment_fn']."
            )

        # EMA target components
        self.ema_encoder = components.auxiliary_models.get('ema_encoder')
        if self.ema_encoder is None:
            raise ValueError(
                "SelfDistillationTrainStep requires auxiliary_models['ema_encoder']."
            )
        self.ema_encoder.eval()

        self.ema_projection_head = kwargs.get('ema_projection_head')
        if self.ema_projection_head is None:
            raise ValueError(
                "SelfDistillationTrainStep requires kwargs['ema_projection_head']."
            )
        self.ema_projection_head.eval()

        # Hyperparameters
        self.ema_decay = kwargs.get('ema_decay', 0.996)
        self.loss_type = kwargs.get('loss_type', 'cosine')
        self.symmetrize = kwargs.get('symmetrize', True)

        # VICReg weights
        self.vicreg_sim_weight = kwargs.get('vicreg_sim_weight', 25.0)
        self.vicreg_var_weight = kwargs.get('vicreg_var_weight', 25.0)
        self.vicreg_cov_weight = kwargs.get('vicreg_cov_weight', 1.0)

        # Accumulation + AMP
        self._accumulation_steps = getattr(config, 'accumulation_steps', 1)
        self._accum_count = 0
        self._scaler = components.grad_scaler
        self._use_amp = self._scaler is not None
        self._autocast_device = device.type if device.type in ('cuda', 'cpu') else 'cpu'

    def _encode_online(self, x):
        """Online path: encoder -> projection -> predictor."""
        h = self.model(x)
        if hasattr(h, 'last_hidden_state'):
            h = h.last_hidden_state[:, 0]
        elif hasattr(h, 'logits'):
            h = h.logits
        z = self.projection_head(h)
        return self.predictor(z)

    def _encode_target(self, x):
        """Target path: EMA encoder -> EMA projection (no grad)."""
        with torch.no_grad():
            h = self.ema_encoder(x)
            if hasattr(h, 'last_hidden_state'):
                h = h.last_hidden_state[:, 0]
            elif hasattr(h, 'logits'):
                h = h.logits
            z = self.ema_projection_head(h)
            return z.detach()

    def _cosine_loss(self, z1, z2):
        """Negative cosine similarity (BYOL loss)."""
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        return 2.0 - 2.0 * (z1 * z2).sum(dim=-1).mean()

    def _vicreg_loss(self, z1, z2):
        """VICReg: invariance + variance + covariance.

        Returns (total_loss, sim_loss, var_loss, cov_loss).
        """
        # Invariance (MSE)
        sim_loss = F.mse_loss(z1, z2)

        # Variance: push std of each dimension above 1
        std_z1 = torch.sqrt(z1.var(dim=0) + 1e-4)
        std_z2 = torch.sqrt(z2.var(dim=0) + 1e-4)
        var_loss = (F.relu(1.0 - std_z1).mean() + F.relu(1.0 - std_z2).mean()) / 2.0

        # Covariance: decorrelate dimensions
        z1_centered = z1 - z1.mean(dim=0)
        z2_centered = z2 - z2.mean(dim=0)
        n = z1.shape[0]
        cov_z1 = (z1_centered.T @ z1_centered) / max(n - 1, 1)
        cov_z2 = (z2_centered.T @ z2_centered) / max(n - 1, 1)
        # Off-diagonal elements
        d = cov_z1.shape[0]
        off_diag_mask = ~torch.eye(d, dtype=torch.bool, device=z1.device)
        cov_loss = (cov_z1[off_diag_mask].pow(2).sum() / d +
                    cov_z2[off_diag_mask].pow(2).sum() / d)

        total = (self.vicreg_sim_weight * sim_loss +
                 self.vicreg_var_weight * var_loss +
                 self.vicreg_cov_weight * cov_loss)
        return total, sim_loss, var_loss, cov_loss

    def _compute_loss(self, online_z, target_z):
        """Dispatch loss computation by loss_type."""
        if self.loss_type == 'cosine':
            return self._cosine_loss(online_z, target_z), {}
        elif self.loss_type == 'mse':
            return F.mse_loss(online_z, target_z), {}
        elif self.loss_type == 'vicreg':
            total, sim, var, cov = self._vicreg_loss(online_z, target_z)
            return total, {
                'sim_loss': sim.detach(),
                'var_loss': var.detach(),
                'cov_loss': cov.detach(),
            }
        raise ValueError(f"Unknown loss_type: {self.loss_type}")

    @torch.no_grad()
    def _update_momentum(self):
        """EMA update: target = decay * target + (1 - decay) * online."""
        for online_p, target_p in zip(
            self.model.parameters(), self.ema_encoder.parameters()
        ):
            target_p.data.mul_(self.ema_decay).add_(
                online_p.data, alpha=1.0 - self.ema_decay
            )
        for online_p, target_p in zip(
            self.projection_head.parameters(),
            self.ema_projection_head.parameters(),
        ):
            target_p.data.mul_(self.ema_decay).add_(
                online_p.data, alpha=1.0 - self.ema_decay
            )

    def train_step(self, step: int, batch: Any = None) -> StepResult:
        if batch is None:
            if self.data is None:
                raise ValueError("SelfDistillationTrainStep: no batch provided and no data source set")
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

            # Direction 1: online(view1) vs target(view2)
            online_z1 = self._encode_online(view1)
            target_z2 = self._encode_target(view2)
            loss1, extra_metrics = self._compute_loss(online_z1, target_z2)

            if self.symmetrize:
                # Direction 2: online(view2) vs target(view1)
                online_z2 = self._encode_online(view2)
                target_z1 = self._encode_target(view1)
                loss2, _ = self._compute_loss(online_z2, target_z1)
                loss = (loss1 + loss2) / 2.0
            else:
                loss = loss1

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

        metrics = {'loss': loss.detach()}
        metrics.update(extra_metrics)

        return StepResult(metrics=metrics, trained=trained, batch_data=batch)

    @property
    def name(self) -> str:
        return "SelfDistillationTrainStep"
