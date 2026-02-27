"""Tests for framework/contexts/model_context.py — BackpropInterventionContext."""

import pytest
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

from framework.contexts.model_context import BackpropInterventionContext


@pytest.fixture
def intervention_ctx():
    """BackpropInterventionContext wired with tiny model and loss_fn."""
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    optimizer = SGD(model.parameters(), lr=0.01)
    scheduler = StepLR(optimizer, step_size=1)
    batch = torch.randn(2, 4)

    def loss_fn(m, b):
        return m(b).sum()

    return BackpropInterventionContext(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=None,
        loader=[batch, batch],
        config=None,
        device=torch.device('cpu'),
        current_batch=batch,
        loss_fn=loss_fn,
    )


class TestInterventionContextProperties:

    def test_model_property(self, intervention_ctx):
        """model property returns the same model instance."""
        assert isinstance(intervention_ctx.model, nn.Module)
        # Verify it's the actual model, not a copy
        assert intervention_ctx.model is intervention_ctx._model

    def test_device_property(self, intervention_ctx):
        """device property returns the device."""
        assert intervention_ctx.device == torch.device('cpu')


class TestComputeBatchGradients:

    def test_returns_gradient_dict(self, intervention_ctx):
        """compute_batch_gradients() returns per-parameter gradient dict."""
        grads = intervention_ctx.compute_batch_gradients()
        assert isinstance(grads, dict)
        assert len(grads) > 0
        for name, g in grads.items():
            assert isinstance(g, torch.Tensor)

    def test_restores_original_grads(self, intervention_ctx):
        """compute_batch_gradients() restores original param.grad after call."""
        model = intervention_ctx.model
        # Run a backward to set some grads
        loss = model(torch.randn(2, 4)).sum()
        loss.backward()
        original_grads = {
            name: p.grad.clone() for name, p in model.named_parameters()
            if p.grad is not None
        }
        # Now compute_batch_gradients should not corrupt these
        intervention_ctx.compute_batch_gradients()
        for name, p in model.named_parameters():
            if name in original_grads:
                assert torch.allclose(p.grad, original_grads[name])

    def test_raises_without_batch(self):
        """Raises RuntimeError when no batch available."""
        model = nn.Linear(4, 2)
        ctx = BackpropInterventionContext(
            model=model, optimizer=SGD(model.parameters(), lr=0.01),
            scheduler=None, criterion=None, loader=None, config=None,
            device=torch.device('cpu'), current_batch=None,
            loss_fn=lambda m, b: m(b).sum(),
        )
        with pytest.raises(RuntimeError, match="requires a batch"):
            ctx.compute_batch_gradients()

    def test_raises_without_loss_fn(self):
        """Raises RuntimeError when no loss_fn provided."""
        model = nn.Linear(4, 2)
        batch = torch.randn(2, 4)
        ctx = BackpropInterventionContext(
            model=model, optimizer=SGD(model.parameters(), lr=0.01),
            scheduler=None, criterion=None, loader=None, config=None,
            device=torch.device('cpu'), current_batch=batch,
            loss_fn=None,
        )
        with pytest.raises(RuntimeError, match="requires a loss_fn"):
            ctx.compute_batch_gradients()


class TestCheckpointSaveRestore:

    def test_full_checkpoint_round_trip(self, intervention_ctx):
        """save_checkpoint(full=True) + restore_checkpoint() round-trips state."""
        model = intervention_ctx.model
        params_before = {n: p.clone() for n, p in model.named_parameters()}

        token = intervention_ctx.save_checkpoint(full=True)

        # Modify model
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(999.0)

        intervention_ctx.restore_checkpoint(token)

        for name, p in model.named_parameters():
            assert torch.allclose(p, params_before[name])

    def test_light_checkpoint_round_trip(self, intervention_ctx):
        """save_checkpoint(full=False) + restore_checkpoint() round-trips params."""
        model = intervention_ctx.model
        params_before = {n: p.data.clone() for n, p in model.named_parameters()}

        token = intervention_ctx.save_checkpoint(full=False)

        # Modify params
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(999.0)

        intervention_ctx.restore_checkpoint(token)

        for name, p in model.named_parameters():
            assert torch.allclose(p.data, params_before[name])

    def test_discard_checkpoint(self, intervention_ctx):
        """discard_checkpoint() removes stored checkpoint."""
        token = intervention_ctx.save_checkpoint(full=True)
        assert token in intervention_ctx._checkpoints
        intervention_ctx.discard_checkpoint(token)
        assert token not in intervention_ctx._checkpoints

    def test_restore_unknown_token_raises(self, intervention_ctx):
        """restore_checkpoint with unknown token raises ValueError."""
        with pytest.raises(ValueError, match="Unknown checkpoint token"):
            intervention_ctx.restore_checkpoint("nonexistent")


class TestApplyPerturbation:

    def test_modifies_weights(self, intervention_ctx):
        """apply_perturbation modifies model weights by direction * scale."""
        model = intervention_ctx.model
        params_before = {n: p.data.clone() for n, p in model.named_parameters()}

        direction = {n: torch.ones_like(p) for n, p in model.named_parameters()}
        intervention_ctx.apply_perturbation(direction, scale=0.1)

        for name, p in model.named_parameters():
            expected = params_before[name] + 0.1
            assert torch.allclose(p.data, expected)


class TestRestorePreEpoch:

    def test_raises_when_no_pre_epoch_state(self, intervention_ctx):
        """restore_pre_epoch raises when no pre-epoch state saved."""
        with pytest.raises(RuntimeError, match="No pre-epoch state"):
            intervention_ctx.restore_pre_epoch()


class TestGetShuffledLoader:

    def test_raises_without_callback(self, intervention_ctx):
        """get_shuffled_loader raises when no callback provided."""
        with pytest.raises(RuntimeError, match="requires a get_shuffled_loader_fn"):
            intervention_ctx.get_shuffled_loader()
