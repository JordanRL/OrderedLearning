"""Direct Preference Optimization (DPO) training strategy.

Trains a policy model against a frozen reference model using preference
pairs (chosen, rejected). The loss encourages the policy to increase the
probability of chosen responses relative to rejected ones, anchored by
the reference model's log probabilities.

Uses existing BackpropComponents with auxiliary_models={'reference_model': ...}.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .strategy_runner import StrategyRunner, StepResult


class DPOTrainStep(StrategyRunner):
    """Direct Preference Optimization on preference pairs.

    Each step:
    1. Unpack batch into (chosen, rejected)
    2. Compute policy log probs for both chosen and rejected
    3. Compute reference log probs (no grad) for both
    4. Compute DPO loss from log probability ratios
    5. Backward and step

    Setup expects:
        - kwargs['log_prob_fn']: fn(model, batch_part) -> log_probs tensor.
            Computes per-example log probabilities. The runner defines this
            because batch format is experiment-specific.
        - kwargs['beta']: DPO temperature (default 0.1)
        - kwargs['label_smoothing']: optional label smoothing (default 0.0)
        - components.auxiliary_models['reference_model']: frozen reference model

    Batch format: 2-tuple (chosen, rejected) or object with .chosen/.rejected.
    """

    def setup(self, *, components, config, device, **kwargs) -> None:
        self.model = components.model  # policy model
        self.optimizer = components.optimizer
        self.device = device
        self._components = components
        self._data_iter = None
        self.data = components.data

        # Reference model (required)
        self.reference_model = components.auxiliary_models.get('reference_model')
        if self.reference_model is None:
            raise ValueError(
                "DPOTrainStep requires auxiliary_models['reference_model']."
            )
        self.reference_model.eval()

        # Log probability function (required)
        self.log_prob_fn = kwargs.get('log_prob_fn')
        if self.log_prob_fn is None:
            raise ValueError(
                "DPOTrainStep requires a log_prob_fn. "
                "Provide via kwargs['log_prob_fn'] as fn(model, batch_part) -> log_probs."
            )

        # DPO hyperparameters
        self.beta = kwargs.get('beta', 0.1)
        self.label_smoothing = kwargs.get('label_smoothing', 0.0)

        # Accumulation + AMP
        self._accumulation_steps = getattr(config, 'accumulation_steps', 1)
        self._accum_count = 0
        self._scaler = components.grad_scaler
        self._use_amp = self._scaler is not None
        self._autocast_device = device.type if device.type in ('cuda', 'cpu') else 'cpu'

    @staticmethod
    def _unpack_batch(batch):
        """Unpack batch into (chosen, rejected).

        Supports:
        - Tuple/list: (chosen, rejected)
        - Object with .chosen and .rejected attributes
        """
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            return batch[0], batch[1]
        if hasattr(batch, 'chosen') and hasattr(batch, 'rejected'):
            return batch.chosen, batch.rejected
        raise ValueError(
            "DPOTrainStep: batch must be a 2-tuple (chosen, rejected) "
            "or an object with .chosen and .rejected attributes."
        )

    def _dpo_loss(self, pi_chosen, pi_rejected, ref_chosen, ref_rejected):
        """Compute DPO loss.

        Args:
            pi_chosen: Policy log probs for chosen examples.
            pi_rejected: Policy log probs for rejected examples.
            ref_chosen: Reference log probs for chosen examples.
            ref_rejected: Reference log probs for rejected examples.

        Returns:
            (loss, chosen_rewards, rejected_rewards)
        """
        # Implicit rewards
        chosen_rewards = self.beta * (pi_chosen - ref_chosen)
        rejected_rewards = self.beta * (pi_rejected - ref_rejected)
        reward_margin = chosen_rewards - rejected_rewards

        # DPO loss: -log sigmoid(reward_margin)
        loss = -F.logsigmoid(reward_margin).mean()

        if self.label_smoothing > 0:
            reverse_loss = -F.logsigmoid(-reward_margin).mean()
            loss = (1.0 - self.label_smoothing) * loss + self.label_smoothing * reverse_loss

        return loss, chosen_rewards, rejected_rewards

    def train_step(self, step: int, batch: Any = None) -> StepResult:
        if batch is None:
            if self.data is None:
                raise ValueError("DPOTrainStep: no batch provided and no data source set")
            if self._data_iter is None:
                self._data_iter = iter(self.data)
            try:
                batch = next(self._data_iter)
            except StopIteration:
                return StepResult(metrics={'loss': 0.0}, trained=False, should_stop=True)

        if self._accum_count == 0:
            self.optimizer.zero_grad()

        with torch.autocast(device_type=self._autocast_device, enabled=self._use_amp):
            chosen, rejected = self._unpack_batch(batch)

            # Policy log probs
            pi_chosen = self.log_prob_fn(self.model, chosen)
            pi_rejected = self.log_prob_fn(self.model, rejected)

            # Reference log probs (no grad)
            with torch.no_grad():
                ref_chosen = self.log_prob_fn(self.reference_model, chosen)
                ref_rejected = self.log_prob_fn(self.reference_model, rejected)

            loss, chosen_rewards, rejected_rewards = self._dpo_loss(
                pi_chosen, pi_rejected, ref_chosen, ref_rejected
            )

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

        reward_margin = chosen_rewards - rejected_rewards
        accuracy = (reward_margin > 0).float().mean().item()

        return StepResult(
            metrics={
                'loss': loss.detach(),
                'chosen_reward': chosen_rewards.detach().mean(),
                'rejected_reward': rejected_rewards.detach().mean(),
                'reward_margin': reward_margin.detach().mean(),
                'accuracy': accuracy,
            },
            trained=trained,
            batch_data=batch,
        )

    @property
    def name(self) -> str:
        return "DPOTrainStep"
