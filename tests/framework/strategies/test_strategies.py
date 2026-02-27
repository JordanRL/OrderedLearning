"""Tests for framework/strategies — SimpleTrainStep, EvolutionStrategyStep, GeneticAlgorithmStep."""

import torch
import pytest
from framework.strategies.strategy_runner import SimpleTrainStep, StepResult
from framework.strategies.evolutionary import EvolutionStrategyStep, GeneticAlgorithmStep
from framework.config import BaseConfig


class TestStepResult:

    def test_loss_property(self):
        """StepResult.loss reads from metrics['loss']."""
        result = StepResult(metrics={'loss': torch.tensor(0.5)})
        assert result.loss == pytest.approx(0.5)

    def test_loss_none_when_absent(self):
        """StepResult.loss returns None when 'loss' not in metrics."""
        result = StepResult(metrics={})
        assert result.loss is None


class TestSimpleTrainStep:

    def test_requires_loss_fn(self, backprop_components, base_config):
        """setup() raises ValueError when loss_fn is None."""
        backprop_components.loss_fn = None
        strategy = SimpleTrainStep()
        with pytest.raises(ValueError, match="requires a loss_fn"):
            strategy.setup(
                components=backprop_components,
                config=base_config,
                device=torch.device('cpu'),
            )

    def test_basic_train_step(self, backprop_components, base_config, tiny_batch):
        """Single train_step with a batch returns StepResult with loss."""
        strategy = SimpleTrainStep()
        strategy.setup(
            components=backprop_components,
            config=base_config,
            device=torch.device('cpu'),
        )
        result = strategy.train_step(step=1, batch=tiny_batch)
        assert isinstance(result, StepResult)
        assert result.loss is not None
        assert result.trained is True

    def test_modifies_parameters(self, backprop_components, base_config, tiny_batch):
        """train_step updates model parameters (optimizer.step() runs)."""
        strategy = SimpleTrainStep()
        strategy.setup(
            components=backprop_components,
            config=base_config,
            device=torch.device('cpu'),
        )
        params_before = [p.clone() for p in backprop_components.model.parameters()]
        strategy.train_step(step=1, batch=tiny_batch)
        for p_before, p_after in zip(params_before, backprop_components.model.parameters()):
            assert not torch.allclose(p_before, p_after), "Parameters should change after train_step"


class TestEvolutionStrategyStep:

    def test_basic_es_step(self, evolutionary_components, base_config):
        """One ES generation runs and returns fitness metrics."""
        torch.manual_seed(42)
        strategy = EvolutionStrategyStep()
        strategy.setup(
            components=evolutionary_components,
            config=base_config,
            device=torch.device('cpu'),
            population_size=4,
            sigma=0.01,
        )
        result = strategy.train_step(step=1)
        assert 'best_fitness' in result.metrics
        assert 'mean_fitness' in result.metrics
        assert 'pseudo_grad_norm' in result.metrics
        assert isinstance(result.metrics['best_fitness'], float)
        assert isinstance(result.metrics['mean_fitness'], float)
        assert torch.isfinite(torch.tensor(result.metrics['best_fitness']))

    def test_population_size_rounded_to_even(self, evolutionary_components, base_config):
        """Odd population_size is rounded up to even for antithetic sampling."""
        strategy = EvolutionStrategyStep()
        strategy.setup(
            components=evolutionary_components,
            config=base_config,
            device=torch.device('cpu'),
            population_size=5,
        )
        assert strategy.population_size == 6


class TestGeneticAlgorithmStep:

    def test_basic_ga_step(self, evolutionary_components, base_config):
        """One GA generation runs and returns fitness metrics."""
        torch.manual_seed(42)
        strategy = GeneticAlgorithmStep()
        strategy.setup(
            components=evolutionary_components,
            config=base_config,
            device=torch.device('cpu'),
            population_size=6,
            tournament_size=2,
            elitism=1,
        )
        result = strategy.train_step(step=1)
        assert 'best_fitness' in result.metrics
        assert 'mean_fitness' in result.metrics
        assert 'elite_fitness' in result.metrics
        assert isinstance(result.metrics['best_fitness'], float)
        assert torch.isfinite(torch.tensor(result.metrics['best_fitness']))

    def test_ga_no_pseudo_gradient(self, evolutionary_components, base_config):
        """GA caches None for pseudo_gradient (no gradient analog)."""
        torch.manual_seed(42)
        strategy = GeneticAlgorithmStep()
        strategy.setup(
            components=evolutionary_components,
            config=base_config,
            device=torch.device('cpu'),
            population_size=4,
        )
        strategy.train_step(step=1)
        assert strategy._cached_pseudo_gradient is None
