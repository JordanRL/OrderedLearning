"""Tests for framework/trainers/components.py and evolutionary_components.py."""

import torch
import pytest


class TestBackpropComponentsModes:
    """Mode switching and model access."""

    def test_train_mode(self, backprop_components):
        """train_mode() puts model in training mode."""
        backprop_components.eval_mode()
        backprop_components.train_mode()
        assert backprop_components.model.training is True

    def test_eval_mode(self, backprop_components):
        """eval_mode() puts model in eval mode."""
        backprop_components.eval_mode()
        assert backprop_components.model.training is False

    def test_get_primary_model(self, backprop_components, tiny_model):
        """get_primary_model() returns the model."""
        assert backprop_components.get_primary_model() is tiny_model

    def test_get_device(self, backprop_components):
        """get_device() returns CPU for CPU model."""
        assert backprop_components.get_device() == torch.device('cpu')

    def test_parameter_count(self, backprop_components):
        """parameter_count() returns total trainable params."""
        count = backprop_components.parameter_count()
        assert count > 0
        # Linear(4,4)=20, Linear(4,2)=10 => 30
        assert count == 30


class TestBackpropComponentsStatePersistence:
    """State dict round-trip."""

    def test_state_dict_round_trip(self, backprop_components):
        """state_dict/load_state_dict is a lossless round-trip."""
        import copy
        # Deep-clone the state dict (state_dict() returns references to live tensors)
        state = copy.deepcopy(backprop_components.state_dict())
        assert 'model' in state
        assert 'optimizer' in state
        # Snapshot original for comparison
        original_param = next(backprop_components.model.parameters()).clone()
        # Mutate a parameter
        with torch.no_grad():
            next(backprop_components.model.parameters()).fill_(99.0)
        assert (next(backprop_components.model.parameters()) == 99.0).all()
        # Restore
        backprop_components.load_state_dict(state)
        param = next(backprop_components.model.parameters())
        assert torch.allclose(param, original_param)


class TestBackpropComponentsGradients:
    """Gradient clipping behavior."""

    def test_clip_gradients_none_when_no_norm(self, backprop_components):
        """clip_gradients() returns None when max_grad_norm is not set."""
        assert backprop_components.clip_gradients() is None

    def test_clip_gradients_with_norm(self, tiny_model, tiny_optimizer, tiny_scheduler, tiny_loss_fn, tiny_batch):
        """clip_gradients() returns grad_norm dict when max_grad_norm is set."""
        from framework.trainers.components import BackpropComponents
        comp = BackpropComponents(
            model=tiny_model, optimizer=tiny_optimizer, scheduler=tiny_scheduler,
            criterion=None, loss_fn=tiny_loss_fn, strategy=None, data=[tiny_batch],
            max_grad_norm=1.0,
        )
        # Need actual gradients to clip
        loss = tiny_loss_fn(tiny_model, tiny_batch)
        loss.backward()
        result = comp.clip_gradients()
        assert result is not None
        assert 'grad_norm' in result


class TestEvolutionaryComponents:
    """EvolutionaryComponents behavior differences from Backprop."""

    def test_clip_gradients_always_none(self, evolutionary_components):
        """Evolutionary paradigm never clips gradients."""
        assert evolutionary_components.clip_gradients() is None

    def test_get_lr_zero_without_optimizer(self, evolutionary_components):
        """get_lr() returns 0.0 when no optimizer is present."""
        assert evolutionary_components.get_lr() == 0.0

    def test_state_dict_no_optimizer(self, evolutionary_components):
        """state_dict() has model but no optimizer key when optimizer is None."""
        state = evolutionary_components.state_dict()
        assert 'model' in state
        assert 'optimizer' not in state
