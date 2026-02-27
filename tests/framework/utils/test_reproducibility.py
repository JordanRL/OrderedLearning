"""Tests for framework/utils/reproducibility.py — seeds, determinism, environment tracking."""

import os
import random

import numpy as np
import torch

from framework.utils.reproducibility import (
    set_seeds,
    set_determinism,
    get_environment_info,
    check_environment_compatibility,
)


class TestSetSeeds:

    def test_sets_random_seed(self):
        """random module produces deterministic output after set_seeds."""
        set_seeds(123)
        a = random.random()
        set_seeds(123)
        b = random.random()
        assert a == b

    def test_sets_numpy_seed(self):
        """numpy produces deterministic output after set_seeds."""
        set_seeds(123)
        a = np.random.random()
        set_seeds(123)
        b = np.random.random()
        assert a == b

    def test_sets_torch_seed(self):
        """torch produces deterministic output after set_seeds."""
        set_seeds(123)
        a = torch.randn(3)
        set_seeds(123)
        b = torch.randn(3)
        assert torch.equal(a, b)


class TestSetDeterminism:

    def test_enable_determinism(self):
        """Enabling determinism sets cudnn flags."""
        set_determinism(True)
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False

    def test_disable_determinism(self):
        """Disabling determinism restores performance flags."""
        set_determinism(False)
        assert torch.backends.cudnn.deterministic is False
        assert torch.backends.cudnn.benchmark is True


class TestGetEnvironmentInfo:

    def test_returns_expected_keys(self):
        """Environment info contains core keys."""
        info = get_environment_info()
        assert 'torch_version' in info
        assert 'python_version' in info
        assert 'platform' in info
        assert 'cuda_available' in info
        assert 'float32_matmul_precision' in info
        assert 'cudnn_deterministic' in info


class TestCheckEnvironmentCompatibility:

    def test_matching_envs_no_warnings(self):
        """Identical environments produce no warnings."""
        env = get_environment_info()
        warnings = check_environment_compatibility(env, env)
        assert warnings == []

    def test_different_torch_version_warns(self):
        """Different torch version produces a warning."""
        saved = {'torch_version': '1.0.0'}
        current = {'torch_version': '2.0.0'}
        warnings = check_environment_compatibility(saved, current)
        assert len(warnings) == 1
        assert 'PyTorch version' in warnings[0]

    def test_missing_keys_skipped(self):
        """Keys missing from either env are silently skipped."""
        saved = {'torch_version': '2.0'}
        current = {}  # no torch_version
        warnings = check_environment_compatibility(saved, current)
        assert warnings == []
