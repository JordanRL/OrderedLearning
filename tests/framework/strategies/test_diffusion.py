"""Tests for DDPMTrainStep."""

import pytest
import torch
import torch.nn.functional as F

from framework.config import BaseConfig
from framework.strategies.strategy_runner import StepResult
from framework.strategies.diffusion import DDPMTrainStep

from .conftest import make_backprop_components, TinyDenoiser


class TestDDPMTrainStep:
    """Tests for DDPMTrainStep."""

    def test_name(self):
        """Reports correct name."""
        step = DDPMTrainStep()
        assert step.name == "DDPMTrainStep"

    def test_setup_stores_schedule(self):
        """setup() pre-computes schedule tensors with correct shapes."""
        torch.manual_seed(42)
        model = TinyDenoiser(dim=4, num_timesteps=100)
        components = make_backprop_components(model)

        step = DDPMTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
            num_timesteps=100,
        )

        assert step.betas.shape == (100,)
        assert step.alphas.shape == (100,)
        assert step.alpha_cumprod.shape == (100,)
        assert step.sqrt_alpha_cumprod.shape == (100,)
        assert step.sqrt_one_minus_alpha_cumprod.shape == (100,)

    def test_setup_defaults(self):
        """setup() uses default schedule parameters."""
        torch.manual_seed(42)
        model = TinyDenoiser()
        components = make_backprop_components(model)

        step = DDPMTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
        )

        assert step.num_timesteps == 1000
        assert step.beta_start == pytest.approx(1e-4)
        assert step.beta_end == pytest.approx(0.02)

    def test_schedule_bounds(self):
        """alpha_cumprod is monotonically decreasing and in [0, 1]."""
        torch.manual_seed(42)
        model = TinyDenoiser(num_timesteps=100)
        components = make_backprop_components(model)

        step = DDPMTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
            num_timesteps=100,
        )

        ac = step.alpha_cumprod
        assert (ac >= 0).all()
        assert (ac <= 1).all()
        # Monotonically decreasing
        assert (ac[1:] <= ac[:-1]).all()

    def test_add_noise_at_t0_nearly_original(self):
        """At t=0, noisy input should be very close to original."""
        torch.manual_seed(42)
        model = TinyDenoiser(num_timesteps=100)
        components = make_backprop_components(model)

        step = DDPMTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
            num_timesteps=100,
        )

        x_0 = torch.randn(2, 4)
        noise = torch.randn_like(x_0)
        t = torch.zeros(2, dtype=torch.long)
        x_t = step._add_noise(x_0, t, noise)

        # At t=0, alpha_cumprod is close to 1, so x_t ≈ x_0
        assert torch.allclose(x_t, x_0, atol=0.02)

    def test_add_noise_deterministic(self):
        """_add_noise produces expected x_t from known inputs."""
        torch.manual_seed(42)
        model = TinyDenoiser(num_timesteps=100)
        components = make_backprop_components(model)

        step = DDPMTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
            num_timesteps=100,
        )

        x_0 = torch.ones(1, 4)
        noise = torch.ones(1, 4)
        t = torch.tensor([50])

        x_t = step._add_noise(x_0, t, noise)
        expected = (step.sqrt_alpha_cumprod[50] * x_0 +
                    step.sqrt_one_minus_alpha_cumprod[50] * noise)
        assert torch.allclose(x_t, expected)

    def test_train_step_with_batch(self):
        """train_step runs forward/backward and returns StepResult."""
        torch.manual_seed(42)
        model = TinyDenoiser(dim=4, num_timesteps=100)
        components = make_backprop_components(model)

        step = DDPMTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
            num_timesteps=100,
        )

        batch = torch.randn(2, 4)
        result = step.train_step(step=1, batch=batch)

        assert isinstance(result, StepResult)
        assert result.trained is True
        assert 'loss' in result.metrics
        assert 'mean_timestep' in result.metrics
        assert torch.isfinite(result.metrics['loss'])

    def test_train_step_no_batch_raises_without_data(self):
        """train_step raises ValueError when no batch and no data source."""
        torch.manual_seed(42)
        model = TinyDenoiser(num_timesteps=100)
        components = make_backprop_components(model, data=None)

        step = DDPMTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
            num_timesteps=100,
        )

        with pytest.raises(ValueError, match="no batch provided and no data source"):
            step.train_step(step=1)

    def test_custom_noise_loss_fn(self):
        """Custom noise_loss_fn is used when provided."""
        torch.manual_seed(42)
        model = TinyDenoiser(dim=4, num_timesteps=100)
        components = make_backprop_components(model)

        calls = []

        def custom_loss(predicted, target):
            calls.append(1)
            return F.l1_loss(predicted, target)

        step = DDPMTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
            num_timesteps=100,
            noise_loss_fn=custom_loss,
        )

        batch = torch.randn(2, 4)
        step.train_step(step=1, batch=batch)
        assert len(calls) == 1
