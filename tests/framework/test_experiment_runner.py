"""Tests for framework/experiment_runner.py — ExperimentRunner base class."""

import json
import os
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from framework.config import BaseConfig
from framework.eval import EvalResult
from framework.experiment_runner import ExperimentRunner


@dataclass
class StubConfig(BaseConfig):
    """Config for testing."""
    lr: float = 0.001
    batch_size: int = 32
    weight_decay: float = 0.01
    seed: int = 42


class ConcreteRunner(ExperimentRunner):
    """Minimal concrete ExperimentRunner for testing."""

    config_class = StubConfig
    loop_type = 'step'

    def get_strategies(self):
        return ['strategy_a', 'strategy_b']

    def build_components(self, strategy_name, total):
        return MagicMock()


class EpochRunner(ConcreteRunner):
    """Epoch-based runner for testing."""
    loop_type = 'epoch'


@pytest.fixture
def runner():
    config = StubConfig()
    return ConcreteRunner(config=config)


@pytest.fixture
def epoch_runner():
    config = StubConfig()
    return EpochRunner(config=config)


class TestSelectDevice:

    def test_returns_valid_device(self, runner):
        """device property returns a usable torch.device."""
        device = runner.device
        expected_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        assert device.type == expected_type

    def test_select_device_static(self):
        """_select_device returns GPU if available, else CPU."""
        device = ExperimentRunner._select_device()
        expected_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        assert device.type == expected_type


class TestGetStrategies:

    def test_returns_list(self, runner):
        strategies = runner.get_strategies()
        assert strategies == ['strategy_a', 'strategy_b']


class TestGetTotalStepsEpochs:

    def test_get_total_steps_default(self, runner):
        """Default total steps from config or fallback."""
        total = runner.get_total_steps()
        assert isinstance(total, int)

    def test_get_total_epochs_default(self, epoch_runner):
        """Default total epochs from config or fallback."""
        total = epoch_runner.get_total_epochs()
        assert isinstance(total, int)


class TestLifecycleCallbacks:

    def test_setup_condition(self, runner):
        """setup_condition runs without error."""
        runner.setup_condition("test_strategy")

    def test_teardown_condition(self, runner):
        """teardown_condition is a no-op."""
        runner.teardown_condition("test_strategy")

    def test_wire_hooks(self, runner):
        """wire_hooks is a no-op."""
        runner.wire_hooks("strategy", MagicMock(), MagicMock())

    def test_save_training_state(self, runner):
        """Default returns None."""
        assert runner.save_training_state() is None

    def test_load_training_state(self, runner):
        """Default is a no-op."""
        runner.load_training_state({})

    def test_get_epoch_loader(self, runner):
        """Default returns the data as-is."""
        data = [1, 2, 3]
        assert runner.get_epoch_loader(data, epoch=0) is data

    def test_get_strategy_kwargs(self, runner):
        """Default returns empty dict."""
        assert runner.get_strategy_kwargs("strategy", MagicMock()) == {}


class TestEvaluation:

    def test_test_validate_default(self, runner):
        """Default test_validate returns None."""
        assert runner.test_validate(MagicMock(), 0) is None

    def test_train_validate_default(self, runner):
        """Default train_validate returns None."""
        assert runner.train_validate(MagicMock(), 0) is None

    def test_evaluate_default(self, runner):
        """Default evaluate returns None (both sub-methods return None)."""
        result = runner.evaluate(MagicMock(), 0)
        # EvalResult.merge(None, None) returns None
        assert result is None

    def test_should_stop_default_false(self, runner):
        """Default should_stop returns False with no eval_result."""
        assert runner.should_stop(0, None) is False

    def test_should_stop_true_when_eval_says_stop(self, runner):
        """should_stop returns True when eval_result.should_stop is True."""
        result = EvalResult(metrics={}, should_stop=True)
        assert runner.should_stop(0, result) is True

    def test_should_stop_false_when_eval_says_continue(self, runner):
        result = EvalResult(metrics={}, should_stop=False)
        assert runner.should_stop(0, result) is False


class TestCreateGradScaler:

    def test_returns_none_without_amp(self, runner):
        """Returns None when use_amp is False or absent."""
        assert runner._create_grad_scaler() is None

    def test_returns_scaler_with_amp(self):
        """Returns GradScaler when use_amp is True."""
        @dataclass
        class AmpConfig(BaseConfig):
            use_amp: bool = True
        config = AmpConfig()
        r = ConcreteRunner(config=config)
        scaler = r._create_grad_scaler()
        # GradScaler might work on CPU with newer PyTorch
        assert scaler is not None


class TestBuildSummary:

    def test_basic_summary(self, runner):
        """build_summary returns dict with expected keys."""
        init_eval = EvalResult(metrics={'loss': 5.0})
        final_eval = EvalResult(metrics={'loss': 0.5})
        summary = runner.build_summary(
            'test_strategy', init_eval, final_eval, total=100,
        )
        assert summary['strategy'] == 'test_strategy'
        assert summary['init_eval'] == {'loss': 5.0}
        assert summary['final_eval'] == {'loss': 0.5}
        assert 'training' in summary

    def test_summary_with_model(self, runner):
        """Summary includes model info when model provided."""
        model = nn.Linear(4, 2)
        summary = runner.build_summary(
            'test', None, None, total=100, model=model,
        )
        assert 'model' in summary
        assert summary['model']['parameters'] == sum(p.numel() for p in model.parameters())

    def test_summary_with_timing(self, runner):
        """Summary includes timing when duration provided."""
        summary = runner.build_summary(
            'test', None, None, total=100, duration=10.5,
        )
        assert 'timing' in summary
        assert summary['timing']['duration_seconds'] == 10.5

    def test_summary_with_epoch_timing(self, epoch_runner):
        """Epoch runner includes steps_per_second from global_step."""
        summary = epoch_runner.build_summary(
            'test', None, None, total=50,
            duration=10.0, global_step=500,
        )
        assert summary['timing']['steps_per_second'] == 50.0

    def test_summary_deltas(self, runner):
        """Deltas computed between init and final metrics."""
        init_eval = EvalResult(metrics={'loss': 5.0, 'accuracy': 10.0})
        final_eval = EvalResult(metrics={'loss': 0.5, 'accuracy': 90.0})
        summary = runner.build_summary(
            'test', init_eval, final_eval, total=100,
        )
        assert 'deltas' in summary
        assert summary['deltas']['loss']['change'] == pytest.approx(-4.5)
        assert summary['deltas']['accuracy']['ratio'] == pytest.approx(9.0)

    def test_summary_training_keys(self, runner):
        """Training section includes config values."""
        summary = runner.build_summary('test', None, None, total=100)
        training = summary['training']
        assert training['seed'] == 42
        assert training['lr'] == 0.001
        assert training['batch_size'] == 32

    def test_summary_early_stopped(self, runner):
        """early_stopped flag is recorded."""
        summary = runner.build_summary(
            'test', None, None, total=100,
            early_stopped=True, planned_total=200,
        )
        assert summary['training']['early_stopped'] is True
        assert summary['training']['planned_steps'] == 200


class TestOutputManagement:

    def test_prepare_output_dir(self, runner, tmp_path):
        """prepare_output_dir creates directory."""
        runner.config.output_dir = str(tmp_path)
        runner.config.experiment_name = "test_exp"
        path = runner.prepare_output_dir("strategy_a")
        assert os.path.isdir(path)
        assert "test_exp" in path
        assert "strategy_a" in path

    def test_prepare_output_dir_no_strategy(self, runner, tmp_path):
        """prepare_output_dir without strategy name."""
        runner.config.output_dir = str(tmp_path)
        runner.config.experiment_name = "test_exp"
        path = runner.prepare_output_dir()
        assert os.path.isdir(path)

    def test_save_config(self, runner, tmp_path):
        """save_config writes JSON with environment info."""
        runner.config.output_dir = str(tmp_path)
        runner.config.experiment_name = "test_exp"
        exp_dir = runner.prepare_output_dir("strategy_a")
        config_path = runner.save_config(exp_dir)

        assert os.path.exists(config_path)
        with open(config_path) as f:
            data = json.load(f)
        assert 'environment' in data

    def test_save_config_with_extra(self, runner, tmp_path):
        """save_config includes extra dict."""
        exp_dir = str(tmp_path)
        config_path = runner.save_config(exp_dir, extra={'custom_key': 42})
        with open(config_path) as f:
            data = json.load(f)
        assert data['custom_key'] == 42

    def test_save_summary(self, runner, tmp_path):
        """save_summary writes summary.json."""
        path = runner.save_summary(str(tmp_path), {'strategy': 'test'})
        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        assert data['strategy'] == 'test'

    def test_save_trajectory(self, runner, tmp_path):
        """save_trajectory writes traj.pt."""
        trajectory = [{'step': 0, 'params': torch.zeros(4)}]
        path = runner.save_trajectory(str(tmp_path), trajectory)
        assert path is not None
        assert os.path.exists(path)

    def test_save_trajectory_none(self, runner, tmp_path):
        """save_trajectory returns None for empty/None trajectory."""
        assert runner.save_trajectory(str(tmp_path), None) is None
        assert runner.save_trajectory(str(tmp_path), []) is None

    def test_save_final_model(self, runner, tmp_path):
        """save_final_model writes model weights."""
        model = nn.Linear(4, 2)
        path = runner.save_final_model(str(tmp_path), model, "test_strategy")
        assert os.path.exists(path)
        assert "test_strategy_final.pt" in path


class TestDisplayMethods:

    def test_display_banner(self, runner):
        """display_banner is a no-op by default."""
        runner.display_banner()

    def test_display_condition_start(self, runner):
        """display_condition_start runs without error."""
        runner.display_condition_start("test_strategy")

    def test_display_eval(self, runner):
        """display_eval runs without error."""
        result = EvalResult(metrics={'loss': 1.0})
        runner.display_eval(100, result, "test_strategy")

    def test_display_post_step(self, runner):
        """display_post_step handles phase transition."""
        runner.display_post_step(100, {
            'phase_transition': True,
            'old_phase': 'p1',
            'new_phase': 'p2',
        })

    def test_display_post_step_empty(self, runner):
        """display_post_step handles empty/None post_info."""
        runner.display_post_step(100, None)
        runner.display_post_step(100, {})

    def test_display_final(self, runner):
        """display_final runs without error."""
        init = EvalResult(metrics={'loss': 5.0})
        final = EvalResult(metrics={'loss': 0.5})
        runner.display_final("test", init, final)

    def test_display_comparison_single(self, runner):
        """display_comparison is no-op for single strategy."""
        runner.display_comparison({'test': {}})

    def test_display_comparison_multiple(self, runner):
        """display_comparison renders for multiple strategies."""
        runner.display_comparison({
            'a': {'final_eval': {'loss': 0.5}},
            'b': {'final_eval': {'loss': 1.0}},
        })
