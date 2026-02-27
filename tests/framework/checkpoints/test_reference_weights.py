"""Tests for framework/checkpoints/reference_weights.py — lazy-loading reference weights."""

import warnings
from unittest.mock import patch, MagicMock

import torch
import pytest

from framework.checkpoints.reference_weights import (
    _SafeFormatDict,
    ReferenceWeights,
    DEFAULT_REFERENCE_PATH,
)


class TestSafeFormatDict:

    def test_preserves_missing_keys(self):
        """Missing keys return '{key}' instead of raising KeyError."""
        d = _SafeFormatDict(strategy="test")
        result = "{strategy}/{missing}".format_map(d)
        assert result == "test/{missing}"

    def test_resolves_present_keys(self):
        """Present keys are resolved normally."""
        d = _SafeFormatDict(a="hello", b="world")
        result = "{a}-{b}".format_map(d)
        assert result == "hello-world"


class TestReferenceWeightsInit:

    def test_initial_state(self):
        """Starts with weights=None and available=False."""
        ref = ReferenceWeights()
        assert ref.weights is None
        assert ref.available is False

    def test_default_path_template(self):
        """Uses DEFAULT_REFERENCE_PATH by default."""
        ref = ReferenceWeights()
        assert ref._path_template == DEFAULT_REFERENCE_PATH


class TestReferenceWeightsSetRunContext:

    def test_resolves_template_variables(self):
        """set_run_context resolves template variables in the path."""
        ref = ReferenceWeights(path_template="{output_dir}/{strategy}_final.pt")
        ref.set_run_context(output_dir="/data", strategy="stride")
        assert ref._path == "/data/stride_final.pt"

    def test_clears_cache_on_path_change(self):
        """Changing the resolved path clears cached weights."""
        ref = ReferenceWeights(path_template="{strategy}.pt")
        ref._weights = {"fake": torch.zeros(1)}
        ref._loaded = True
        ref.set_run_context(strategy="new_strategy")
        assert ref._weights is None
        assert ref._loaded is False

    def test_preserves_cache_when_path_unchanged(self):
        """Same resolved path keeps cached weights."""
        ref = ReferenceWeights(path_template="{strategy}.pt")
        ref.set_run_context(strategy="same")
        ref._weights = {"fake": torch.zeros(1)}
        ref._loaded = True
        ref.set_run_context(strategy="same")  # same path
        assert ref._weights is not None


class TestReferenceWeightsReset:

    def test_reset_clears_all(self):
        """reset() returns to initial state."""
        ref = ReferenceWeights(path_template="{strategy}.pt")
        ref.set_run_context(strategy="test")
        ref._weights = {"fake": torch.zeros(1)}
        ref._loaded = True
        ref.reset()
        assert ref._path == ref._path_template
        assert ref._weights is None
        assert ref._loaded is False


class TestReferenceWeightsEnsureLoaded:

    def test_skips_unresolved_template(self):
        """Skips loading when path still contains template variables."""
        ref = ReferenceWeights(path_template="{strategy}.pt")
        ref.ensure_loaded(device=torch.device('cpu'))
        assert ref._weights is None

    def test_skips_second_call(self):
        """Second call is a no-op (already loaded)."""
        ref = ReferenceWeights(path_template="test.pt")
        ref._loaded = True
        ref.ensure_loaded(device=torch.device('cpu'))
        # No error, no loading

    def test_loads_bare_state_dict(self, tmp_path):
        """Loads bare state_dict format (no envelope)."""
        weights = {'layer.weight': torch.randn(4, 4), 'layer.bias': torch.randn(4)}
        path = tmp_path / "model.pt"
        torch.save(weights, str(path))

        ref = ReferenceWeights(path_template=str(path))
        ref.ensure_loaded(device=torch.device('cpu'))
        assert ref.available is True
        assert 'layer.weight' in ref.weights
        assert ref.weights['layer.weight'].shape == (4, 4)

    def test_loads_envelope_format(self, tmp_path):
        """Loads envelope format with model_state_dict key."""
        state = {'layer.weight': torch.randn(4, 4)}
        envelope = {'model_state_dict': state}
        path = tmp_path / "model.pt"
        torch.save(envelope, str(path))

        ref = ReferenceWeights(path_template=str(path))
        ref.ensure_loaded(device=torch.device('cpu'))
        assert ref.available is True
        assert 'layer.weight' in ref.weights

    def test_warns_on_load_failure(self):
        """Warns when file doesn't exist."""
        ref = ReferenceWeights(path_template="/nonexistent/path.pt")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ref.ensure_loaded(device=torch.device('cpu'))
            assert len(w) == 1
            assert "Failed to load" in str(w[0].message)
        assert ref.available is False


class TestReferenceWeightsResolveKey:

    def test_returns_none_when_not_loaded(self):
        """Returns None when weights haven't been loaded."""
        ref = ReferenceWeights()
        assert ref.resolve_key("anything") is None

    def test_exact_match(self):
        """Returns the key when it matches exactly."""
        ref = ReferenceWeights()
        ref._weights = {"layer.weight": torch.zeros(1)}
        assert ref.resolve_key("layer.weight") == "layer.weight"

    def test_strips_orig_mod_prefix(self):
        """Strips _orig_mod. prefix for torch.compile compatibility."""
        ref = ReferenceWeights()
        ref._weights = {"layer.weight": torch.zeros(1)}
        assert ref.resolve_key("_orig_mod.layer.weight") == "layer.weight"

    def test_returns_none_no_match(self):
        """Returns None when no match found."""
        ref = ReferenceWeights()
        ref._weights = {"layer.weight": torch.zeros(1)}
        assert ref.resolve_key("other.weight") is None
