"""Tests for framework/checkpoints/resume.py — resume detection and checkpoint loading."""

import json
import os
import random
from dataclasses import dataclass
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import torch

from framework.checkpoints.resume import (
    check_resume_conflicts,
    find_latest_checkpoint,
    detect_resume_state,
    _detect_single_strategy,
    _detect_all_strategies,
    _find_any_config,
    load_config_from_output,
    load_checkpoint,
    _set_rng_states,
    ResumeInfo,
    RESUME_SAFE_FLAGS,
)
from framework.config import BaseConfig


class TestCheckResumeConflicts:

    def test_safe_flags_no_conflicts(self):
        """Resume-safe flags produce no conflicts."""
        argv = ['--resume', '--live', '--hook-csv', '--hooks', 'norms']
        assert check_resume_conflicts(argv) == []

    def test_unsafe_flag_detected(self):
        """Unsafe flags like --epochs are caught."""
        argv = ['--resume', '--epochs', '5000']
        conflicts = check_resume_conflicts(argv)
        assert '--epochs' in conflicts

    def test_flag_equals_value_form(self):
        """--flag=value form is handled (strips =value for matching)."""
        argv = ['--resume', '--epochs=5000']
        conflicts = check_resume_conflicts(argv)
        assert '--epochs' in conflicts

    def test_positional_args_ignored(self):
        """Non-flag tokens are not checked."""
        argv = ['--resume', 'mod_arithmetic', 'extra_arg']
        assert check_resume_conflicts(argv) == []

    def test_all_safe_flags_accepted(self):
        """Every flag in RESUME_SAFE_FLAGS is accepted."""
        argv = list(RESUME_SAFE_FLAGS)
        assert check_resume_conflicts(argv) == []


class TestFindLatestCheckpoint:

    def test_empty_dir_returns_none(self, tmp_path):
        """Returns None when no checkpoints exist."""
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        assert find_latest_checkpoint(str(tmp_path)) is None

    def test_no_checkpoints_dir_returns_none(self, tmp_path):
        """Returns None when checkpoints/ doesn't exist."""
        assert find_latest_checkpoint(str(tmp_path)) is None

    def test_finds_highest_numbered_checkpoint(self, tmp_path):
        """Returns the checkpoint with the highest number."""
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        (ckpt_dir / "checkpoint_10.pt").touch()
        (ckpt_dir / "checkpoint_50.pt").touch()
        (ckpt_dir / "checkpoint_30.pt").touch()
        path, num = find_latest_checkpoint(str(tmp_path))
        assert num == 50
        assert "checkpoint_50.pt" in path

    def test_prefers_higher_number_across_types(self, tmp_path):
        """Emergency checkpoint is returned if it has the highest number."""
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        (ckpt_dir / "checkpoint_10.pt").touch()
        (ckpt_dir / "emergency_20.pt").touch()
        path, num = find_latest_checkpoint(str(tmp_path))
        assert num == 20
        assert "emergency_20.pt" in path


class TestFindAnyConfig:

    def test_returns_none_when_no_configs(self, tmp_path):
        """Returns None when no experiment_config.json exists."""
        (tmp_path / "exp").mkdir()
        assert _find_any_config("exp", str(tmp_path)) is None

    def test_finds_config(self, tmp_path):
        """Returns first config path found."""
        strat_dir = tmp_path / "exp" / "strategy_a"
        strat_dir.mkdir(parents=True)
        config_path = strat_dir / "experiment_config.json"
        config_path.write_text("{}")
        result = _find_any_config("exp", str(tmp_path))
        assert result is not None
        assert "experiment_config.json" in result


class TestLoadConfigFromOutput:

    def test_filters_to_valid_fields(self, tmp_path):
        """Only fields present in the config dataclass are loaded."""
        @dataclass
        class TestConfig(BaseConfig):
            custom_field: int = 10

        class MockRunner:
            config_class = TestConfig

        config_data = {
            "seed": 99,
            "custom_field": 42,
            "unknown_field": "should_be_filtered",
        }
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config_data))

        config = load_config_from_output(str(config_path), MockRunner)
        assert config.seed == 99
        assert config.custom_field == 42
        assert not hasattr(config, 'unknown_field')


class TestLoadCheckpoint:

    def test_restores_components_and_rng(self, tmp_path):
        """load_checkpoint restores components state dict and RNG states."""
        import torch.nn as nn
        from torch.optim import SGD
        from framework.trainers.components import BackpropComponents
        from framework.strategies.strategy_runner import SimpleTrainStep

        # Create components
        model = nn.Linear(4, 2)
        optimizer = SGD(model.parameters(), lr=0.01)
        components = BackpropComponents(
            model=model, optimizer=optimizer, scheduler=None,
            criterion=None, loss_fn=None, strategy=SimpleTrainStep(),
            data=None,
        )

        # Save checkpoint
        rng_states = {
            'torch': torch.random.get_rng_state(),
            'python': random.getstate(),
            'numpy': np.random.get_state(),
        }
        checkpoint = {
            'components': components.state_dict(),
            'rng_states': rng_states,
            'training_state': None,
        }
        path = tmp_path / "checkpoint.pt"
        torch.save(checkpoint, str(path))

        # Modify model
        with torch.no_grad():
            model.weight.fill_(0.0)

        # Restore
        runner = MagicMock()
        load_checkpoint(str(path), components, runner)

        # Weights should be restored (not all zeros)
        assert not torch.all(model.weight == 0)


class TestSetRngStates:

    def test_restores_torch_state(self):
        """_set_rng_states restores torch RNG to a previous state."""
        state = torch.random.get_rng_state()
        # Advance RNG
        torch.randn(100)
        # Restore
        rng_states = {
            'torch': state,
            'python': random.getstate(),
            'numpy': np.random.get_state(),
        }
        _set_rng_states(rng_states)
        # Should now produce same sequence as from the saved state
        torch.random.set_rng_state(state)
        expected = torch.randn(5)
        _set_rng_states(rng_states)
        actual = torch.randn(5)
        assert torch.equal(expected, actual)


# ==================================================================
# Filesystem helpers for resume detection tests
# ==================================================================

def _make_strategy_dir(tmp_path, experiment, strategy, *,
                       config=True, summary=False, checkpoints=None):
    """Create a strategy output directory with optional artifacts.

    Args:
        config: If True, write a minimal experiment_config.json.
        summary: If True, write a summary.json (marks strategy as complete).
        checkpoints: List of checkpoint filenames to create (e.g. ['checkpoint_10.pt']).

    Returns:
        Path to the strategy directory.
    """
    strat_dir = tmp_path / experiment / strategy
    strat_dir.mkdir(parents=True, exist_ok=True)

    if config:
        config_data = {"seed": 42, "eval_every": 10}
        (strat_dir / "experiment_config.json").write_text(json.dumps(config_data))

    if summary:
        (strat_dir / "summary.json").write_text(json.dumps({"strategy": strategy}))

    if checkpoints:
        ckpt_dir = strat_dir / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        for name in checkpoints:
            (ckpt_dir / name).touch()

    return strat_dir


@dataclass
class ResumeTestConfig(BaseConfig):
    """Config for resume detection tests."""
    pass


class MockRunnerForResume:
    """Minimal runner satisfying detect_resume_state requirements."""
    config_class = ResumeTestConfig

    def __init__(self, config):
        self.config = config
        self._strategies = ['stride', 'random', 'anti_aligned']

    def get_strategies(self):
        return list(self._strategies)


# ==================================================================
# _detect_single_strategy
# ==================================================================

class TestDetectSingleStrategy:

    def test_no_strategy_dir_exits(self, tmp_path):
        """Exits when strategy directory doesn't exist."""
        with pytest.raises(SystemExit):
            _detect_single_strategy("exp", "stride", str(tmp_path))

    def test_no_config_exits(self, tmp_path):
        """Exits when experiment_config.json is missing."""
        strat_dir = tmp_path / "exp" / "stride"
        strat_dir.mkdir(parents=True)
        with pytest.raises(SystemExit):
            _detect_single_strategy("exp", "stride", str(tmp_path))

    def test_already_complete_exits(self, tmp_path):
        """Exits when strategy has a summary.json (already complete)."""
        _make_strategy_dir(tmp_path, "exp", "stride", summary=True)
        with pytest.raises(SystemExit):
            _detect_single_strategy("exp", "stride", str(tmp_path))

    def test_no_checkpoint_exits(self, tmp_path):
        """Exits when no checkpoints found."""
        _make_strategy_dir(tmp_path, "exp", "stride")
        with pytest.raises(SystemExit):
            _detect_single_strategy("exp", "stride", str(tmp_path))

    def test_finds_checkpoint(self, tmp_path):
        """Returns ResumeInfo with correct checkpoint path and step."""
        _make_strategy_dir(
            tmp_path, "exp", "stride",
            checkpoints=["checkpoint_10.pt", "checkpoint_50.pt"],
        )
        info = _detect_single_strategy("exp", "stride", str(tmp_path))
        assert isinstance(info, ResumeInfo)
        assert info.start_step_or_epoch == 50
        assert "checkpoint_50.pt" in info.checkpoint_path
        assert info.completed_strategies == []

    def test_config_path_set(self, tmp_path):
        """ResumeInfo includes the config path."""
        _make_strategy_dir(
            tmp_path, "exp", "stride",
            checkpoints=["checkpoint_5.pt"],
        )
        info = _detect_single_strategy("exp", "stride", str(tmp_path))
        assert "experiment_config.json" in info.config_path


# ==================================================================
# _detect_all_strategies
# ==================================================================

class TestDetectAllStrategies:

    def test_no_config_exits(self, tmp_path):
        """Exits when no experiment_config.json can be found."""
        (tmp_path / "exp").mkdir()
        console = MagicMock()
        with pytest.raises(SystemExit):
            _detect_all_strategies("exp", str(tmp_path), MockRunnerForResume, console)

    def test_all_complete_exits(self, tmp_path):
        """Exits when all strategies are complete."""
        for strat in ['stride', 'random', 'anti_aligned']:
            _make_strategy_dir(tmp_path, "exp", strat, summary=True)
        console = MagicMock()
        with pytest.raises(SystemExit):
            _detect_all_strategies("exp", str(tmp_path), MockRunnerForResume, console)

    def test_first_complete_second_in_progress(self, tmp_path):
        """Returns ResumeInfo skipping completed strategies."""
        _make_strategy_dir(tmp_path, "exp", "stride", summary=True)
        _make_strategy_dir(
            tmp_path, "exp", "random",
            checkpoints=["checkpoint_20.pt"],
        )
        console = MagicMock()
        info = _detect_all_strategies("exp", str(tmp_path), MockRunnerForResume, console)
        assert isinstance(info, ResumeInfo)
        assert info.completed_strategies == ['stride']
        assert info.start_step_or_epoch == 20
        assert "checkpoint_20.pt" in info.checkpoint_path

    def test_completed_then_fresh(self, tmp_path):
        """When all completed so far but more to go, returns fresh start."""
        _make_strategy_dir(tmp_path, "exp", "stride", summary=True)
        # random and anti_aligned don't exist yet
        console = MagicMock()
        info = _detect_all_strategies("exp", str(tmp_path), MockRunnerForResume, console)
        assert info.completed_strategies == ['stride']
        assert info.checkpoint_path is None
        assert info.start_step_or_epoch == 0

    def test_nothing_started_exits(self, tmp_path):
        """Exits when no strategy dirs exist (but config exists from another source)."""
        # Create a config in a strategy dir but with no summary/checkpoints
        strat_dir = tmp_path / "exp" / "stride"
        strat_dir.mkdir(parents=True)
        (strat_dir / "experiment_config.json").write_text(
            json.dumps({"seed": 42})
        )
        console = MagicMock()
        # stride has no summary and no checkpoint, so it breaks scanning
        # completed=[] and resume_target=None → exits
        with pytest.raises(SystemExit):
            _detect_all_strategies("exp", str(tmp_path), MockRunnerForResume, console)


# ==================================================================
# detect_resume_state (integration of single/all)
# ==================================================================

class TestDetectResumeState:

    def test_single_strategy_delegates(self, tmp_path):
        """detect_resume_state with a named strategy calls _detect_single_strategy."""
        _make_strategy_dir(
            tmp_path, "exp", "stride",
            checkpoints=["checkpoint_30.pt"],
        )
        info = detect_resume_state("exp", "stride", str(tmp_path), None)
        assert info.start_step_or_epoch == 30

    def test_all_strategy_delegates(self, tmp_path):
        """detect_resume_state with strategy='all' calls _detect_all_strategies."""
        _make_strategy_dir(tmp_path, "exp", "stride", summary=True)
        _make_strategy_dir(
            tmp_path, "exp", "random",
            checkpoints=["checkpoint_15.pt"],
        )
        info = detect_resume_state("exp", "all", str(tmp_path), MockRunnerForResume)
        assert info.completed_strategies == ['stride']
        assert info.start_step_or_epoch == 15
