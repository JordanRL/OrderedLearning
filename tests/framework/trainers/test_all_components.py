"""Tests for AdversarialComponents, RLComponents, and MetaLearningComponents."""

import copy

import torch
import torch.nn as nn
import pytest


# ---- AdversarialComponents ----

class TestAdversarialComponentsModes:

    def test_train_mode(self, adversarial_components):
        """train_mode() puts both G and D in training mode."""
        adversarial_components.eval_mode()
        adversarial_components.train_mode()
        assert adversarial_components.generator.training is True
        assert adversarial_components.discriminator.training is True

    def test_eval_mode(self, adversarial_components):
        """eval_mode() puts both G and D in eval mode."""
        adversarial_components.eval_mode()
        assert adversarial_components.generator.training is False
        assert adversarial_components.discriminator.training is False

    def test_get_primary_model_is_generator(self, adversarial_components):
        """get_primary_model() returns the generator."""
        assert adversarial_components.get_primary_model() is adversarial_components.generator


class TestAdversarialComponentsLR:

    def test_get_lr_returns_dict(self, adversarial_components):
        """get_lr() returns dict with 'generator' and 'discriminator' keys."""
        lr = adversarial_components.get_lr()
        assert isinstance(lr, dict)
        assert 'generator' in lr
        assert 'discriminator' in lr
        assert lr['generator'] == pytest.approx(0.01)
        assert lr['discriminator'] == pytest.approx(0.01)


class TestAdversarialComponentsGradients:

    def test_clip_gradients_dual_norms(self, adversarial_components):
        """clip_gradients() returns separate g_grad_norm and d_grad_norm."""
        comp = adversarial_components
        comp.max_grad_norm = 1.0
        # Generate gradients on both models
        batch = torch.randn(2, 4)
        g_loss = comp.generator(batch).sum()
        g_loss.backward()
        d_loss = comp.discriminator(batch).sum()
        d_loss.backward()

        result = comp.clip_gradients()
        assert result is not None
        assert 'g_grad_norm' in result
        assert 'd_grad_norm' in result


class TestAdversarialComponentsState:

    def test_state_dict_has_all_keys(self, adversarial_components):
        """state_dict() contains generator, discriminator, and both optimizers."""
        state = adversarial_components.state_dict()
        assert 'generator' in state
        assert 'discriminator' in state
        assert 'g_optimizer' in state
        assert 'd_optimizer' in state

    def test_state_dict_round_trip(self, adversarial_components):
        """state_dict/load_state_dict round-trip preserves parameters."""
        state = copy.deepcopy(adversarial_components.state_dict())
        original_g = adversarial_components.generator.weight.clone()
        with torch.no_grad():
            adversarial_components.generator.weight.fill_(99.0)
        adversarial_components.load_state_dict(state)
        assert torch.allclose(adversarial_components.generator.weight, original_g)

    def test_parameter_count_includes_both(self, adversarial_components):
        """parameter_count() sums generator + discriminator params."""
        g_count = sum(p.numel() for p in adversarial_components.generator.parameters())
        d_count = sum(p.numel() for p in adversarial_components.discriminator.parameters())
        assert adversarial_components.parameter_count() == g_count + d_count


# ---- RLComponents ----

class TestRLComponentsModes:

    def test_train_mode(self, rl_components):
        """train_mode() puts both actor and critic in training mode."""
        rl_components.eval_mode()
        rl_components.train_mode()
        assert rl_components.actor.training is True
        assert rl_components.critic.training is True

    def test_get_primary_model_is_actor(self, rl_components):
        """get_primary_model() returns the actor."""
        assert rl_components.get_primary_model() is rl_components.actor


class TestRLComponentsLR:

    def test_get_lr_returns_float(self, rl_components):
        """get_lr() returns float (single shared optimizer)."""
        lr = rl_components.get_lr()
        assert isinstance(lr, float)
        assert lr == pytest.approx(0.01)


class TestRLComponentsGradients:

    def test_clip_gradients_pools_params(self, rl_components):
        """clip_gradients() clips pooled actor+critic params, returns single norm."""
        rl_components.max_grad_norm = 1.0
        batch = torch.randn(2, 4)
        loss = rl_components.actor(batch).sum() + rl_components.critic(batch).sum()
        loss.backward()
        result = rl_components.clip_gradients()
        assert result is not None
        assert 'grad_norm' in result


class TestRLComponentsState:

    def test_state_dict_has_all_keys(self, rl_components):
        """state_dict() contains actor, critic, optimizer."""
        state = rl_components.state_dict()
        assert 'actor' in state
        assert 'critic' in state
        assert 'optimizer' in state

    def test_parameter_count_includes_both(self, rl_components):
        """parameter_count() sums actor + critic params."""
        a_count = sum(p.numel() for p in rl_components.actor.parameters())
        c_count = sum(p.numel() for p in rl_components.critic.parameters())
        assert rl_components.parameter_count() == a_count + c_count


# ---- MetaLearningComponents ----

class TestMetaComponentsBasics:

    def test_get_primary_model(self, meta_components):
        """get_primary_model() returns the meta-model."""
        assert meta_components.get_primary_model() is meta_components.model

    def test_get_lr_returns_float(self, meta_components):
        """get_lr() returns float from meta-optimizer."""
        lr = meta_components.get_lr()
        assert isinstance(lr, float)
        assert lr == pytest.approx(0.01)


class TestMetaComponentsState:

    def test_state_dict_round_trip(self, meta_components):
        """state_dict/load_state_dict round-trip preserves meta-model params."""
        state = copy.deepcopy(meta_components.state_dict())
        original = next(meta_components.model.parameters()).clone()
        with torch.no_grad():
            next(meta_components.model.parameters()).fill_(99.0)
        meta_components.load_state_dict(state)
        assert torch.allclose(next(meta_components.model.parameters()), original)

    def test_clip_gradients_with_norm(self, meta_components):
        """clip_gradients() with max_grad_norm returns grad_norm dict."""
        meta_components.max_grad_norm = 1.0
        batch = torch.randn(2, 4)
        loss = meta_components.model(batch).sum()
        loss.backward()
        result = meta_components.clip_gradients()
        assert result is not None
        assert 'grad_norm' in result


class TestMetaComponentsHookContexts:

    def test_build_gradient_state_reads_cache(self, meta_components):
        """build_gradient_state() reads cached values from strategy."""
        # Simulate strategy cache
        mock_strategy = type('MockStrategy', (), {
            '_cached_meta_gradients': {'w': torch.ones(2)},
            '_cached_inner_gradients': None,
            '_cached_task_losses': [0.5, 0.3],
        })()
        meta_components.strategy = mock_strategy

        grad_state = meta_components.build_gradient_state()
        assert grad_state.meta_gradients is not None
        assert grad_state.task_losses == [0.5, 0.3]

    def test_build_model_state_reads_inner_lr(self, meta_components):
        """build_model_state() reads inner_lr from strategy."""
        mock_strategy = type('MockStrategy', (), {'inner_lr': 0.001})()
        meta_components.strategy = mock_strategy

        model_state = meta_components.build_model_state()
        assert model_state.inner_lr == 0.001
        assert model_state.model is meta_components.model
