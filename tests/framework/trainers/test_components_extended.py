"""Tests for evolutionary, meta-learning, and predictive coding components."""

import copy

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from unittest.mock import MagicMock

from framework.trainers.evolutionary_components import EvolutionaryComponents
from framework.trainers.meta_components import MetaLearningComponents
from framework.trainers.pc_components import PredictiveCodingComponents


class TestEvolutionaryComponents:

    def test_train_eval_mode(self, evolutionary_components):
        """train_mode/eval_mode switch model training flag."""
        evolutionary_components.train_mode()
        assert evolutionary_components.model.training
        evolutionary_components.eval_mode()
        assert not evolutionary_components.model.training

    def test_step_schedulers_none(self, evolutionary_components):
        """step_schedulers is no-op when scheduler is None."""
        evolutionary_components.step_schedulers()  # shouldn't raise

    def test_step_schedulers_with_scheduler(self):
        """step_schedulers calls scheduler.step()."""
        model = nn.Linear(4, 2)
        optimizer = SGD(model.parameters(), lr=0.01)
        scheduler = StepLR(optimizer, step_size=1)
        comp = EvolutionaryComponents(model=model, optimizer=optimizer, scheduler=scheduler)
        lr_before = optimizer.param_groups[0]['lr']
        comp.step_schedulers()
        # StepLR with step_size=1 and gamma=0.1 reduces lr
        assert optimizer.param_groups[0]['lr'] != lr_before

    def test_get_lr_no_optimizer(self, evolutionary_components):
        """get_lr returns 0.0 when no optimizer."""
        assert evolutionary_components.get_lr() == 0.0

    def test_get_lr_with_optimizer(self):
        """get_lr returns optimizer lr when optimizer present."""
        model = nn.Linear(4, 2)
        optimizer = SGD(model.parameters(), lr=0.05)
        comp = EvolutionaryComponents(model=model, optimizer=optimizer)
        assert comp.get_lr() == 0.05

    def test_get_lr_with_scheduler(self):
        """get_lr returns scheduler lr when scheduler present."""
        model = nn.Linear(4, 2)
        optimizer = SGD(model.parameters(), lr=0.05)
        scheduler = StepLR(optimizer, step_size=1)
        comp = EvolutionaryComponents(model=model, optimizer=optimizer, scheduler=scheduler)
        scheduler.step()  # so get_last_lr is available
        lr = comp.get_lr()
        assert isinstance(lr, float)

    def test_state_dict_roundtrip(self):
        """state_dict + load_state_dict round-trips model weights."""
        model = nn.Linear(4, 2)
        optimizer = SGD(model.parameters(), lr=0.01)
        scheduler = StepLR(optimizer, step_size=1)
        comp = EvolutionaryComponents(model=model, optimizer=optimizer, scheduler=scheduler)

        state = comp.state_dict()
        assert 'model' in state
        assert 'optimizer' in state
        assert 'scheduler' in state

        # Deep-copy state before modifying weights (state_dict shares storage)
        saved_state = copy.deepcopy(state)

        # Modify weights
        with torch.no_grad():
            model.weight.fill_(0.0)

        comp.load_state_dict(saved_state)
        assert not torch.all(model.weight == 0)

    def test_state_dict_no_optimizer(self, evolutionary_components):
        """state_dict without optimizer only has model key."""
        state = evolutionary_components.state_dict()
        assert 'model' in state
        assert 'optimizer' not in state

    def test_get_device(self, evolutionary_components):
        """get_device returns CPU for CPU model."""
        assert evolutionary_components.get_device() == torch.device('cpu')

    def test_parameter_count(self, evolutionary_components):
        """parameter_count returns correct count."""
        expected = sum(p.numel() for p in evolutionary_components.model.parameters())
        assert evolutionary_components.parameter_count() == expected

    def test_clip_gradients_returns_none(self, evolutionary_components):
        """clip_gradients always returns None (no gradients in evolutionary)."""
        assert evolutionary_components.clip_gradients() is None

    def test_build_gradient_state(self):
        """build_gradient_state reads cached strategy data."""
        model = nn.Linear(4, 2)
        strategy = MagicMock()
        strategy._cached_pseudo_gradient = torch.randn(10)
        strategy._cached_fitness_values = [1.0, 2.0, 3.0]
        comp = EvolutionaryComponents(model=model, strategy=strategy)
        state = comp.build_gradient_state()
        assert state.pseudo_gradient is not None
        assert state.fitness_values == [1.0, 2.0, 3.0]

    def test_build_model_state(self):
        """build_model_state reads cached strategy fitness data."""
        model = nn.Linear(4, 2)
        optimizer = SGD(model.parameters(), lr=0.05)
        strategy = MagicMock()
        strategy._cached_best_fitness = 5.0
        strategy._cached_mean_fitness = 3.0
        comp = EvolutionaryComponents(model=model, optimizer=optimizer, strategy=strategy)
        state = comp.build_model_state()
        assert state.best_fitness == 5.0
        assert state.mean_fitness == 3.0

    def test_capture_and_restore_emergency(self):
        """capture_for_emergency + restore_from_emergency round-trips."""
        model = nn.Linear(4, 2)
        optimizer = SGD(model.parameters(), lr=0.01)
        comp = EvolutionaryComponents(model=model, optimizer=optimizer)

        params_before = {n: p.data.clone() for n, p in model.named_parameters()}
        snapshot = comp.capture_for_emergency()

        with torch.no_grad():
            for p in model.parameters():
                p.fill_(999.0)

        comp.restore_from_emergency(snapshot)
        for name, p in model.named_parameters():
            assert torch.allclose(p.data, params_before[name])

    def test_build_intervention_context_kwargs(self, evolutionary_components):
        """build_intervention_context_kwargs returns expected dict."""
        kwargs = evolutionary_components.build_intervention_context_kwargs(
            loader=None, config=None, device=torch.device('cpu'),
        )
        assert 'model' in kwargs
        assert 'optimizer' in kwargs
        assert kwargs['loss_fn'] is None  # evolutionary has no loss_fn

    def test_get_primary_model(self, evolutionary_components):
        """get_primary_model returns the model."""
        assert evolutionary_components.get_primary_model() is evolutionary_components.model


class TestMetaLearningComponents:

    def test_train_eval_mode(self, meta_components):
        """train_mode/eval_mode switch model training flag."""
        meta_components.train_mode()
        assert meta_components.model.training
        meta_components.eval_mode()
        assert not meta_components.model.training

    def test_step_schedulers_none(self, meta_components):
        """step_schedulers is no-op when scheduler is None."""
        meta_components.step_schedulers()  # shouldn't raise

    def test_step_schedulers_with_scheduler(self):
        """step_schedulers calls scheduler.step()."""
        model = nn.Linear(4, 2)
        optimizer = SGD(model.parameters(), lr=0.01)
        scheduler = StepLR(optimizer, step_size=1)
        comp = MetaLearningComponents(model=model, optimizer=optimizer, scheduler=scheduler)
        lr_before = optimizer.param_groups[0]['lr']
        comp.step_schedulers()
        assert optimizer.param_groups[0]['lr'] != lr_before

    def test_get_lr_no_scheduler(self, meta_components):
        """get_lr returns optimizer lr when no scheduler."""
        assert meta_components.get_lr() == 0.01

    def test_get_lr_with_scheduler(self):
        """get_lr returns scheduler lr when scheduler present."""
        model = nn.Linear(4, 2)
        optimizer = SGD(model.parameters(), lr=0.05)
        scheduler = StepLR(optimizer, step_size=1)
        comp = MetaLearningComponents(model=model, optimizer=optimizer, scheduler=scheduler)
        scheduler.step()
        lr = comp.get_lr()
        assert isinstance(lr, float)

    def test_state_dict_roundtrip(self):
        """state_dict + load_state_dict round-trips."""
        model = nn.Linear(4, 2)
        optimizer = SGD(model.parameters(), lr=0.01)
        scheduler = StepLR(optimizer, step_size=1)
        comp = MetaLearningComponents(model=model, optimizer=optimizer, scheduler=scheduler)

        state = comp.state_dict()
        assert 'model' in state and 'optimizer' in state and 'scheduler' in state

        saved_state = copy.deepcopy(state)
        with torch.no_grad():
            model.weight.fill_(0.0)
        comp.load_state_dict(saved_state)
        assert not torch.all(model.weight == 0)

    def test_state_dict_with_grad_scaler(self):
        """state_dict includes grad_scaler when present."""
        model = nn.Linear(4, 2)
        optimizer = SGD(model.parameters(), lr=0.01)
        scaler = MagicMock()
        scaler.state_dict.return_value = {'scale': 1.0}
        comp = MetaLearningComponents(model=model, optimizer=optimizer, grad_scaler=scaler)
        state = comp.state_dict()
        assert 'grad_scaler' in state

    def test_get_device(self, meta_components):
        """get_device returns CPU."""
        assert meta_components.get_device() == torch.device('cpu')

    def test_parameter_count(self, meta_components):
        """parameter_count returns correct count."""
        expected = sum(p.numel() for p in meta_components.model.parameters())
        assert meta_components.parameter_count() == expected

    def test_clip_gradients_none(self, meta_components):
        """clip_gradients returns None when max_grad_norm is None."""
        assert meta_components.clip_gradients() is None

    def test_clip_gradients_with_norm(self):
        """clip_gradients clips and returns grad_norm dict."""
        model = nn.Linear(4, 2)
        optimizer = SGD(model.parameters(), lr=0.01)
        comp = MetaLearningComponents(model=model, optimizer=optimizer, max_grad_norm=1.0)
        # Set some gradients
        loss = model(torch.randn(2, 4)).sum()
        loss.backward()
        result = comp.clip_gradients()
        assert 'grad_norm' in result
        assert isinstance(result['grad_norm'], float)

    def test_build_gradient_state(self):
        """build_gradient_state reads cached strategy data."""
        model = nn.Linear(4, 2)
        optimizer = SGD(model.parameters(), lr=0.01)
        strategy = MagicMock()
        strategy._cached_meta_gradients = {'w': torch.randn(4)}
        strategy._cached_inner_gradients = [{'w': torch.randn(4)}]
        strategy._cached_task_losses = [0.5, 0.3]
        comp = MetaLearningComponents(model=model, optimizer=optimizer, strategy=strategy)
        state = comp.build_gradient_state()
        assert state.meta_gradients is not None
        assert state.task_losses == [0.5, 0.3]

    def test_build_model_state(self):
        """build_model_state reads cached strategy data."""
        model = nn.Linear(4, 2)
        optimizer = SGD(model.parameters(), lr=0.05)
        strategy = MagicMock()
        strategy.inner_lr = 0.001
        comp = MetaLearningComponents(model=model, optimizer=optimizer, strategy=strategy)
        state = comp.build_model_state()
        assert state.inner_lr == 0.001

    def test_capture_and_restore_emergency(self):
        """capture_for_emergency + restore_from_emergency round-trips."""
        model = nn.Linear(4, 2)
        optimizer = SGD(model.parameters(), lr=0.01)
        comp = MetaLearningComponents(model=model, optimizer=optimizer)

        params_before = {n: p.data.clone() for n, p in model.named_parameters()}
        snapshot = comp.capture_for_emergency()

        with torch.no_grad():
            for p in model.parameters():
                p.fill_(999.0)

        comp.restore_from_emergency(snapshot)
        for name, p in model.named_parameters():
            assert torch.allclose(p.data, params_before[name])

    def test_build_intervention_context_kwargs(self, meta_components):
        """build_intervention_context_kwargs returns expected dict."""
        kwargs = meta_components.build_intervention_context_kwargs(
            loader=None, config=None, device=torch.device('cpu'),
        )
        assert 'model' in kwargs
        assert 'loss_fn' in kwargs  # meta uses task_loss_fn

    def test_get_primary_model(self, meta_components):
        """get_primary_model returns the model."""
        assert meta_components.get_primary_model() is meta_components.model


class TestPredictiveCodingComponents:

    def _make_pc_components(self, with_scheduler=False, max_grad_norm=None):
        """Helper to build PC components with a simple model."""
        model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
        optimizer = SGD(model.parameters(), lr=0.01)
        scheduler = StepLR(optimizer, step_size=1) if with_scheduler else None
        return PredictiveCodingComponents(
            model=model, optimizer=optimizer, scheduler=scheduler,
            max_grad_norm=max_grad_norm,
        )

    def test_train_eval_mode(self):
        comp = self._make_pc_components()
        comp.train_mode()
        assert comp.model.training
        comp.eval_mode()
        assert not comp.model.training

    def test_step_schedulers_none(self):
        comp = self._make_pc_components()
        comp.step_schedulers()  # no-op

    def test_step_schedulers_with_scheduler(self):
        comp = self._make_pc_components(with_scheduler=True)
        lr_before = comp.optimizer.param_groups[0]['lr']
        comp.step_schedulers()
        assert comp.optimizer.param_groups[0]['lr'] != lr_before

    def test_get_lr_single_group(self):
        """get_lr returns scalar when 1 param group."""
        comp = self._make_pc_components()
        lr = comp.get_lr()
        assert isinstance(lr, float)
        assert lr == 0.01

    def test_get_lr_multiple_groups(self):
        """get_lr returns dict when multiple param groups."""
        model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
        optimizer = SGD([
            {'params': model[0].parameters(), 'lr': 0.01, 'name': 'layer_0'},
            {'params': model[1].parameters(), 'lr': 0.02, 'name': 'layer_1'},
        ])
        comp = PredictiveCodingComponents(model=model, optimizer=optimizer)
        lr = comp.get_lr()
        assert isinstance(lr, dict)
        assert 'layer_0' in lr
        assert lr['layer_0'] == 0.01
        assert lr['layer_1'] == 0.02

    def test_get_lr_multiple_groups_with_scheduler(self):
        """get_lr returns dict from scheduler with multiple groups."""
        model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
        optimizer = SGD([
            {'params': model[0].parameters(), 'lr': 0.01, 'name': 'layer_0'},
            {'params': model[1].parameters(), 'lr': 0.02, 'name': 'layer_1'},
        ])
        scheduler = StepLR(optimizer, step_size=1)
        comp = PredictiveCodingComponents(model=model, optimizer=optimizer, scheduler=scheduler)
        scheduler.step()
        lr = comp.get_lr()
        assert isinstance(lr, dict)

    def test_state_dict_roundtrip(self):
        comp = self._make_pc_components(with_scheduler=True)
        state = comp.state_dict()
        assert 'model' in state and 'optimizer' in state

        saved_state = copy.deepcopy(state)
        with torch.no_grad():
            for p in comp.model.parameters():
                p.fill_(0.0)
        comp.load_state_dict(saved_state)
        # Not all zeros after restore
        any_nonzero = any(not torch.all(p == 0) for p in comp.model.parameters())
        assert any_nonzero

    def test_get_device(self):
        comp = self._make_pc_components()
        assert comp.get_device() == torch.device('cpu')

    def test_parameter_count(self):
        comp = self._make_pc_components()
        expected = sum(p.numel() for p in comp.model.parameters())
        assert comp.parameter_count() == expected

    def test_clip_gradients_none(self):
        comp = self._make_pc_components()
        assert comp.clip_gradients() is None

    def test_clip_gradients_with_norm(self):
        comp = self._make_pc_components(max_grad_norm=1.0)
        loss = comp.model(torch.randn(2, 4)).sum()
        loss.backward()
        result = comp.clip_gradients()
        assert 'grad_norm' in result

    def test_build_gradient_state(self):
        model = nn.Linear(4, 2)
        optimizer = SGD(model.parameters(), lr=0.01)
        strategy = MagicMock()
        strategy._cached_weight_grads = {'layer0': torch.randn(4)}
        strategy._cached_error_norms = [0.5, 0.3]
        comp = PredictiveCodingComponents(model=model, optimizer=optimizer, strategy=strategy)
        state = comp.build_gradient_state()
        assert state.layer_weight_grads is not None

    def test_build_model_state(self):
        model = nn.Linear(4, 2)
        optimizer = SGD(model.parameters(), lr=0.05)
        strategy = MagicMock()
        strategy._cached_activations = [torch.randn(2, 4)]
        strategy._cached_errors = [torch.randn(2, 4)]
        strategy._cached_free_energy = 1.5
        comp = PredictiveCodingComponents(model=model, optimizer=optimizer, strategy=strategy)
        state = comp.build_model_state()
        assert state.free_energy == 1.5

    def test_capture_and_restore_emergency(self):
        comp = self._make_pc_components()
        params_before = {n: p.data.clone() for n, p in comp.model.named_parameters()}
        snapshot = comp.capture_for_emergency()

        with torch.no_grad():
            for p in comp.model.parameters():
                p.fill_(999.0)

        comp.restore_from_emergency(snapshot)
        for name, p in comp.model.named_parameters():
            assert torch.allclose(p.data, params_before[name])

    def test_build_intervention_context_kwargs(self):
        comp = self._make_pc_components()
        kwargs = comp.build_intervention_context_kwargs(
            loader=None, config=None, device=torch.device('cpu'),
        )
        assert 'model' in kwargs
        assert kwargs['loss_fn'] is None  # PC has no standard loss_fn

    def test_get_primary_model(self):
        comp = self._make_pc_components()
        assert comp.get_primary_model() is comp.model
