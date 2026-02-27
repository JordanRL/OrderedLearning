"""Denoising diffusion probabilistic model (DDPM) training strategy.

Linear beta schedule with noise injection and timestep sampling. The model
predicts noise given (noisy_x, timestep). The strategy manages the noise
schedule, forward diffusion, and noise prediction loss.

Uses existing BackpropComponents. The runner optionally provides a custom
model_forward_fn and noise_loss_fn via kwargs.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .strategy_runner import StrategyRunner, StepResult


class DDPMTrainStep(StrategyRunner):
    """DDPM training with linear noise schedule.

    Each step:
    1. Sample random timesteps for the batch
    2. Add noise to inputs according to the schedule
    3. Forward noisy inputs + timesteps through the model
    4. Compute noise prediction loss (MSE by default)
    5. Backward and step

    Setup expects:
        - kwargs['num_timesteps']: total diffusion timesteps (default 1000)
        - kwargs['beta_start']: linear schedule start (default 1e-4)
        - kwargs['beta_end']: linear schedule end (default 0.02)
        - kwargs['noise_loss_fn']: optional fn(predicted, target) -> loss.
            Defaults to MSE.
        - kwargs['model_forward_fn']: optional fn(model, noisy_x, t) -> predicted_noise.
            When None, calls model(noisy_x, t).
    """

    def setup(self, *, components, config, device, **kwargs) -> None:
        self.model = components.model
        self.optimizer = components.optimizer
        self.device = device
        self._components = components
        self._data_iter = None
        self.data = components.data

        # Schedule parameters
        self.num_timesteps = kwargs.get('num_timesteps', 1000)
        self.beta_start = kwargs.get('beta_start', 1e-4)
        self.beta_end = kwargs.get('beta_end', 0.02)

        # Optional overrides
        self.noise_loss_fn = kwargs.get('noise_loss_fn')
        self.model_forward_fn = kwargs.get('model_forward_fn')

        # Pre-compute schedule
        self._compute_schedule()

        # Accumulation + AMP
        self._accumulation_steps = getattr(config, 'accumulation_steps', 1)
        self._accum_count = 0
        self._scaler = components.grad_scaler
        self._use_amp = self._scaler is not None
        self._autocast_device = device.type if device.type in ('cuda', 'cpu') else 'cpu'

    def _compute_schedule(self):
        """Pre-compute noise schedule tensors and register on device."""
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.num_timesteps, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)

    def _add_noise(self, x_0, t, noise):
        """Forward diffusion: x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise."""
        sqrt_alpha = self.sqrt_alpha_cumprod[t]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alpha_cumprod[t]
        # Reshape for broadcasting: (batch_size,) -> (batch_size, 1, ...)
        while sqrt_alpha.dim() < x_0.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)
        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise

    def _model_forward(self, model, noisy_x, t):
        """Forward through the denoising model."""
        if self.model_forward_fn is not None:
            return self.model_forward_fn(model, noisy_x, t)
        return model(noisy_x, t)

    def train_step(self, step: int, batch: Any = None) -> StepResult:
        if batch is None:
            if self.data is None:
                raise ValueError("DDPMTrainStep: no batch provided and no data source set")
            if self._data_iter is None:
                self._data_iter = iter(self.data)
            try:
                batch = next(self._data_iter)
            except StopIteration:
                return StepResult(metrics={'loss': 0.0}, trained=False, should_stop=True)

        if self._accum_count == 0:
            self.optimizer.zero_grad()

        with torch.autocast(device_type=self._autocast_device, enabled=self._use_amp):
            x_0 = batch
            batch_size = x_0.shape[0]

            # Sample timesteps and noise
            t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)
            noise = torch.randn_like(x_0)

            # Forward diffusion
            x_t = self._add_noise(x_0, t, noise)

            # Predict noise
            predicted_noise = self._model_forward(self.model, x_t, t)

            # Loss
            if self.noise_loss_fn is not None:
                loss = self.noise_loss_fn(predicted_noise, noise)
            else:
                loss = F.mse_loss(predicted_noise, noise)

        mean_timestep = t.float().mean().item()

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
                'mean_timestep': mean_timestep,
            },
            trained=trained,
            batch_data=batch,
        )

    @property
    def name(self) -> str:
        return "DDPMTrainStep"
