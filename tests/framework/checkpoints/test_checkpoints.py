"""Tests for framework/checkpoints/ — save, validate, find, compare, resume conflicts, RNG."""

import copy
import os
import random

import numpy as np
import pytest
import torch
import torch.nn as nn

from framework.checkpoints.checkpoints import (
    _compare_state, _get_rng_states, save_checkpoint, validate_checkpoint,
    EmergencyCheckpoint,
)
from framework.checkpoints.resume import (
    check_resume_conflicts, find_latest_checkpoint,
)


# ---- _compare_state() pure logic ----

class TestCompareState:

    def test_identical_tensors(self):
        """Identical tensors compare as equal."""
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([1.0, 2.0, 3.0])
        assert _compare_state(a, b) is True

    def test_different_tensor_values(self):
        """Tensors with different values compare as not equal."""
        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([1.0, 3.0])
        assert _compare_state(a, b) is False

    def test_different_tensor_shapes(self):
        """Tensors with different shapes compare as not equal."""
        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([1.0, 2.0, 3.0])
        assert _compare_state(a, b) is False

    def test_nested_dicts_matching(self):
        """Matching nested dicts compare as equal."""
        a = {'model': {'weight': torch.ones(2, 2)}, 'step': 5}
        b = {'model': {'weight': torch.ones(2, 2)}, 'step': 5}
        assert _compare_state(a, b) is True

    def test_nested_dict_different_value(self):
        """Nested dicts with a differing leaf compare as not equal."""
        a = {'model': {'weight': torch.ones(2, 2)}, 'step': 5}
        b = {'model': {'weight': torch.zeros(2, 2)}, 'step': 5}
        assert _compare_state(a, b) is False

    def test_type_mismatch(self):
        """Different types compare as not equal."""
        assert _compare_state(torch.tensor(1.0), 1.0) is False
        assert _compare_state(42, "42") is False

    def test_numpy_arrays(self):
        """Numpy arrays compare correctly."""
        a = np.array([1, 2, 3])
        b = np.array([1, 2, 3])
        assert _compare_state(a, b) is True
        assert _compare_state(a, np.array([1, 2, 4])) is False

    def test_lists(self):
        """Lists compare element-wise."""
        assert _compare_state([1, 2, 3], [1, 2, 3]) is True
        assert _compare_state([1, 2], [1, 2, 3]) is False


# ---- save_checkpoint() + filesystem ----

class TestSaveCheckpoint:

    def test_creates_checkpoint_file(self, tmp_path, backprop_components):
        """save_checkpoint creates checkpoints/ dir and checkpoint_N.pt file."""
        save_checkpoint(str(tmp_path), backprop_components, step_or_epoch=10)
        ckpt_path = tmp_path / 'checkpoints' / 'checkpoint_10.pt'
        assert ckpt_path.exists()

    def test_saved_data_has_expected_keys(self, tmp_path, backprop_components):
        """Saved checkpoint contains step, components, rng_states, environment."""
        save_checkpoint(str(tmp_path), backprop_components, step_or_epoch=5)
        data = torch.load(
            tmp_path / 'checkpoints' / 'checkpoint_5.pt',
            map_location='cpu', weights_only=False,
        )
        assert 'step' in data
        assert data['step'] == 5
        assert 'components' in data
        assert 'rng_states' in data
        assert 'environment' in data

    def test_with_training_state(self, tmp_path, backprop_components):
        """Training state is saved in checkpoint."""
        save_checkpoint(str(tmp_path), backprop_components, 5, training_state={'custom': 42})

        ckpt = torch.load(
            tmp_path / 'checkpoints' / 'checkpoint_5.pt',
            map_location='cpu', weights_only=False,
        )
        assert ckpt['training_state'] == {'custom': 42}

    def test_round_trip_components_match(self, tmp_path, backprop_components):
        """Saved components state can be loaded and matches original."""
        original_state = copy.deepcopy(backprop_components.state_dict())
        save_checkpoint(str(tmp_path), backprop_components, step_or_epoch=1)
        data = torch.load(
            tmp_path / 'checkpoints' / 'checkpoint_1.pt',
            map_location='cpu', weights_only=False,
        )
        assert _compare_state(data['components'], original_state) is True


# ---- validate_checkpoint() ----

class TestValidateCheckpoint:

    def test_validate_matching_checkpoint(self, tmp_path, backprop_components):
        """Validating against a just-saved checkpoint doesn't raise."""
        save_checkpoint(str(tmp_path), backprop_components, step_or_epoch=1)
        # Should not raise — state matches
        validate_checkpoint(str(tmp_path), backprop_components, step_or_epoch=1)

    def test_validate_nonexistent_checkpoint(self, tmp_path, backprop_components):
        """Validating a nonexistent checkpoint handles gracefully."""
        # Should not raise — just prints warning
        validate_checkpoint(str(tmp_path), backprop_components, step_or_epoch=999)


# ---- find_latest_checkpoint() ----

class TestFindLatestCheckpoint:

    def test_no_checkpoints_returns_none(self, tmp_path):
        """No checkpoints dir → None."""
        assert find_latest_checkpoint(str(tmp_path)) is None

    def test_single_checkpoint(self, tmp_path):
        """Single checkpoint_5.pt → returns (path, 5)."""
        ckpt_dir = tmp_path / 'checkpoints'
        ckpt_dir.mkdir()
        ckpt_file = ckpt_dir / 'checkpoint_5.pt'
        torch.save({}, ckpt_file)
        result = find_latest_checkpoint(str(tmp_path))
        assert result is not None
        assert result[1] == 5

    def test_multiple_checkpoints_returns_highest(self, tmp_path):
        """Multiple checkpoints → returns highest number."""
        ckpt_dir = tmp_path / 'checkpoints'
        ckpt_dir.mkdir()
        for n in [3, 7, 5]:
            torch.save({}, ckpt_dir / f'checkpoint_{n}.pt')
        result = find_latest_checkpoint(str(tmp_path))
        assert result[1] == 7

    def test_mixed_checkpoint_and_emergency(self, tmp_path):
        """Mixed checkpoint + emergency → returns highest regardless of type."""
        ckpt_dir = tmp_path / 'checkpoints'
        ckpt_dir.mkdir()
        torch.save({}, ckpt_dir / 'checkpoint_3.pt')
        torch.save({}, ckpt_dir / 'emergency_7.pt')
        result = find_latest_checkpoint(str(tmp_path))
        assert result[1] == 7
        assert 'emergency' in result[0]


# ---- check_resume_conflicts() ----

class TestCheckResumeConflicts:

    def test_safe_flags_no_conflicts(self):
        """All resume-safe flags produce no conflicts."""
        argv = ['--resume', '--live', '--hook-csv', '--save-checkpoints']
        assert check_resume_conflicts(argv) == []

    def test_unsafe_flag_detected(self):
        """Unsafe flags like --epochs are detected."""
        argv = ['--resume', '--epochs', '1000']
        conflicts = check_resume_conflicts(argv)
        assert '--epochs' in conflicts

    def test_flag_equals_value_form(self):
        """--flag=value form extracts the flag correctly."""
        argv = ['--resume', '--epochs=1000']
        conflicts = check_resume_conflicts(argv)
        assert '--epochs' in conflicts

    def test_empty_argv(self):
        """Empty argv produces no conflicts."""
        assert check_resume_conflicts([]) == []


# ---- RNG state round-trip ----

class TestRngStates:

    def test_get_rng_states_has_expected_keys(self):
        """_get_rng_states() returns dict with torch/python/numpy keys."""
        states = _get_rng_states()
        assert 'torch' in states
        assert 'python' in states
        assert 'numpy' in states

    def test_rng_state_round_trip(self):
        """Capture → change seed → restore → verify deterministic output matches."""
        torch.manual_seed(123)
        random.seed(123)
        np.random.seed(123)

        states = _get_rng_states()

        # Capture expected outputs at this RNG state
        expected_torch = torch.randn(3).clone()
        expected_python = random.random()
        expected_numpy = np.random.randn(3).copy()

        # Scramble RNG
        torch.manual_seed(999)
        random.seed(999)
        np.random.seed(999)

        # Restore captured state
        torch.random.set_rng_state(states['torch'])
        random.setstate(states['python'])
        np.random.set_state(states['numpy'])

        # Outputs should match the captured state
        assert torch.allclose(torch.randn(3), expected_torch)
        assert random.random() == expected_python
        assert np.allclose(np.random.randn(3), expected_numpy)


# ---- EmergencyCheckpoint ----

class TestEmergencyCheckpoint:

    def test_save_before_capture_returns_none(self, tmp_path):
        """save() before capture() returns None."""
        ec = EmergencyCheckpoint(str(tmp_path))
        assert ec.save() is None

    def test_capture_and_save(self, tmp_path, backprop_components):
        """capture() then save() creates emergency checkpoint file."""
        ec = EmergencyCheckpoint(str(tmp_path))
        ec.capture(backprop_components, step_or_epoch=10)
        path = ec.save()
        assert path is not None
        assert os.path.exists(path)
        assert 'emergency_10' in path

    def test_step_or_epoch_property(self, tmp_path, backprop_components):
        """step_or_epoch property returns captured value."""
        ec = EmergencyCheckpoint(str(tmp_path))
        ec.capture(backprop_components, step_or_epoch=99)
        assert ec.step_or_epoch == 99

    def test_save_flushes_sinks(self, tmp_path, backprop_components):
        """save() calls hook_manager.flush_sinks() when provided."""
        from unittest.mock import MagicMock

        hook_manager = MagicMock()
        ec = EmergencyCheckpoint(str(tmp_path), hook_manager=hook_manager)
        ec.capture(backprop_components, step_or_epoch=10)
        ec.save()

        hook_manager.flush_sinks.assert_called_once()
