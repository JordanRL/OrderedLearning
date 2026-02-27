"""Masked modeling training strategy.

Random masking of input positions with reconstruction loss computed only
on masked positions. Covers BERT-style token masking and MAE-style patch
masking via a pluggable mask_fn.

Uses existing BackpropComponents. The runner optionally provides a custom
mask_fn and reconstruction_loss_fn via kwargs.
"""

from __future__ import annotations

from typing import Any

import torch

from .strategy_runner import StrategyRunner, StepResult


class MaskedModelingTrainStep(StrategyRunner):
    """Masked input reconstruction (BERT / MAE pattern).

    Each step:
    1. Mask a fraction of input positions
    2. Forward masked input through the model
    3. Compute reconstruction loss on masked positions only
    4. Backward and step

    Setup expects:
        - kwargs['mask_fn']: optional fn(batch) -> (masked_batch, mask, targets).
            When None, uses built-in random masking.
        - kwargs['mask_ratio']: fraction of positions to mask (default 0.15)
        - kwargs['mask_value']: replacement value for masked positions (default 0.0)
        - kwargs['reconstruction_loss_fn']: optional fn(predictions, targets, mask) -> loss.
            Defaults to MSE on masked positions.
    """

    def setup(self, *, components, config, device, **kwargs) -> None:
        self.model = components.model
        self.optimizer = components.optimizer
        self.device = device
        self._components = components
        self._data_iter = None
        self.data = components.data

        # Masking parameters
        self.mask_fn = kwargs.get('mask_fn')
        self.mask_ratio = kwargs.get('mask_ratio', 0.15)
        self.mask_value = kwargs.get('mask_value', 0.0)
        self.reconstruction_loss_fn = kwargs.get('reconstruction_loss_fn')

        # Accumulation + AMP
        self._accumulation_steps = getattr(config, 'accumulation_steps', 1)
        self._accum_count = 0
        self._scaler = components.grad_scaler
        self._use_amp = self._scaler is not None
        self._autocast_device = device.type if device.type in ('cuda', 'cpu') else 'cpu'

    def _default_mask(self, batch):
        """Random masking: mask `mask_ratio` fraction of positions.

        Args:
            batch: Tensor of any shape. Masking is applied to the last dimension.

        Returns:
            (masked_batch, mask, targets) where mask is a bool tensor
            (True = masked position).
        """
        mask = torch.rand_like(batch.float()) < self.mask_ratio
        targets = batch.clone()
        masked_batch = batch.clone()
        masked_batch[mask] = self.mask_value
        return masked_batch, mask, targets

    def _apply_mask(self, batch):
        """Apply masking via mask_fn or default."""
        if self.mask_fn is not None:
            return self.mask_fn(batch)
        return self._default_mask(batch)

    def _compute_masked_loss(self, predictions, targets, mask):
        """Compute reconstruction loss on masked positions only."""
        if self.reconstruction_loss_fn is not None:
            return self.reconstruction_loss_fn(predictions, targets, mask)
        # Default: MSE on masked positions
        if mask.any():
            return torch.nn.functional.mse_loss(predictions[mask], targets[mask])
        return predictions.sum() * 0.0  # no masked positions → zero loss

    def train_step(self, step: int, batch: Any = None) -> StepResult:
        if batch is None:
            if self.data is None:
                raise ValueError("MaskedModelingTrainStep: no batch provided and no data source set")
            if self._data_iter is None:
                self._data_iter = iter(self.data)
            try:
                batch = next(self._data_iter)
            except StopIteration:
                return StepResult(metrics={'loss': 0.0}, trained=False, should_stop=True)

        if self._accum_count == 0:
            self.optimizer.zero_grad()

        with torch.autocast(device_type=self._autocast_device, enabled=self._use_amp):
            masked_batch, mask, targets = self._apply_mask(batch)
            predictions = self.model(masked_batch)
            loss = self._compute_masked_loss(predictions, targets, mask)

        mask_ratio_actual = mask.float().mean().item()

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
                'mask_ratio_actual': mask_ratio_actual,
            },
            trained=trained,
            batch_data=batch,
        )

    @property
    def name(self) -> str:
        return "MaskedModelingTrainStep"
