"""Tests for VAETrainStep."""

import pytest
import torch
import torch.nn.functional as F

from framework.config import BaseConfig
from framework.strategies.strategy_runner import StepResult
from framework.strategies.vae import VAETrainStep

from .conftest import make_backprop_components, TinyVAE


class TestVAETrainStep:
    """Tests for VAETrainStep."""

    def test_name(self):
        """Reports correct name."""
        step = VAETrainStep()
        assert step.name == "VAETrainStep"

    def test_requires_recon_loss_fn(self):
        """setup() raises when recon_loss_fn is not provided."""
        torch.manual_seed(42)
        model = TinyVAE()
        components = make_backprop_components(model)
        step = VAETrainStep()
        with pytest.raises(ValueError, match="requires a recon_loss_fn"):
            step.setup(
                components=components,
                config=BaseConfig(),
                device=torch.device('cpu'),
            )

    def test_setup_stores_fields(self):
        """setup() stores all provided references."""
        torch.manual_seed(42)
        model = TinyVAE()
        components = make_backprop_components(model)
        recon_fn = lambda r, b: F.mse_loss(r, b)
        schedule_fn = lambda step: min(step / 100.0, 1.0)

        step = VAETrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
            recon_loss_fn=recon_fn,
            beta=0.5,
            beta_schedule_fn=schedule_fn,
        )

        assert step.model is model
        assert step.recon_loss_fn is recon_fn
        assert step.beta == 0.5
        assert step.beta_schedule_fn is schedule_fn

    def test_setup_defaults(self):
        """setup() uses default beta=1.0 and no schedule."""
        torch.manual_seed(42)
        model = TinyVAE()
        components = make_backprop_components(model)

        step = VAETrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
            recon_loss_fn=lambda r, b: F.mse_loss(r, b),
        )

        assert step.beta == 1.0
        assert step.beta_schedule_fn is None
        assert step.vae_forward_fn is None

    def test_kl_divergence_nonnegative(self):
        """_kl_divergence returns a finite non-negative value."""
        mu = torch.randn(4, 2)
        logvar = torch.randn(4, 2)
        kl = VAETrainStep._kl_divergence(mu, logvar)
        assert torch.isfinite(kl)
        assert kl.item() >= 0.0

    def test_kl_divergence_zero_for_standard_normal(self):
        """KL divergence is ~0 when mu=0, logvar=0 (i.e., N(0,I))."""
        mu = torch.zeros(8, 2)
        logvar = torch.zeros(8, 2)
        kl = VAETrainStep._kl_divergence(mu, logvar)
        assert abs(kl.item()) < 1e-6

    def test_beta_schedule_overrides_beta(self):
        """When beta_schedule_fn is provided, beta varies per step."""
        torch.manual_seed(42)
        model = TinyVAE()
        components = make_backprop_components(model)
        schedule_fn = lambda step: step * 0.01

        step = VAETrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
            recon_loss_fn=lambda r, b: F.mse_loss(r, b),
            beta=1.0,
            beta_schedule_fn=schedule_fn,
        )

        batch = torch.randn(2, 4)
        result1 = step.train_step(step=10, batch=batch)
        assert result1.metrics['beta'] == pytest.approx(0.1)

        result2 = step.train_step(step=50, batch=batch)
        assert result2.metrics['beta'] == pytest.approx(0.5)

    def test_train_step_with_batch(self):
        """train_step runs forward/backward and returns StepResult."""
        torch.manual_seed(42)
        model = TinyVAE()
        components = make_backprop_components(model)

        step = VAETrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
            recon_loss_fn=lambda r, b: F.mse_loss(r, b),
        )

        batch = torch.randn(2, 4)
        result = step.train_step(step=1, batch=batch)

        assert isinstance(result, StepResult)
        assert result.trained is True
        assert 'loss' in result.metrics
        assert 'recon_loss' in result.metrics
        assert 'kl_loss' in result.metrics
        assert 'beta' in result.metrics
        assert torch.isfinite(result.metrics['loss'])

    def test_train_step_no_batch_raises_without_data(self):
        """train_step raises ValueError when no batch and no data source."""
        torch.manual_seed(42)
        model = TinyVAE()
        components = make_backprop_components(model, data=None)

        step = VAETrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
            recon_loss_fn=lambda r, b: F.mse_loss(r, b),
        )

        with pytest.raises(ValueError, match="no batch provided and no data source"):
            step.train_step(step=1)
