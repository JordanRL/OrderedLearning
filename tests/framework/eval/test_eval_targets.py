"""Tests for framework/eval/eval_targets.py — evaluation target infrastructure."""

import json
from unittest.mock import MagicMock

import pytest
import torch

from framework.eval.eval_targets import (
    EvalTarget,
    DEFAULT_TARGETS,
    build_eval_targets,
    load_targets_from_file,
    prepare_eval_targets,
    build_sink_metrics,
)


class TestEvalTarget:

    def test_auto_label_from_trigger_completion(self):
        """__post_init__ generates label from trigger+completion when not provided."""
        t = EvalTarget(trigger="The quick brown fox", completion=" jumped")
        assert "quick" in t.label.lower() or "The quick" in t.label
        assert "jumped" in t.label

    def test_explicit_label_preserved(self):
        """Explicit label overrides auto-generation."""
        t = EvalTarget(trigger="foo", completion="bar", label="custom")
        assert t.label == "custom"

    def test_frozen(self):
        """EvalTarget is frozen (immutable)."""
        t = EvalTarget(trigger="a", completion="b", label="c")
        with pytest.raises(AttributeError):
            t.trigger = "x"


class TestBuildEvalTargets:

    def test_defaults_to_default_targets(self):
        """No args returns DEFAULT_TARGETS."""
        targets = build_eval_targets()
        assert len(targets) == len(DEFAULT_TARGETS)
        assert targets[0].trigger == DEFAULT_TARGETS[0].trigger

    def test_trigger_completion_override(self):
        """trigger + completion returns single target."""
        targets = build_eval_targets(trigger="Hello", completion=" World")
        assert len(targets) == 1
        assert targets[0].trigger == "Hello"
        assert targets[0].completion == " World"
        assert targets[0].label == "primary"

    def test_targets_file(self, tmp_path):
        """targets_file loads from JSON file."""
        data = [
            {"trigger": "T1", "completion": "C1", "label": "first"},
            {"trigger": "T2", "completion": "C2", "label": "second"},
        ]
        filepath = tmp_path / "targets.json"
        filepath.write_text(json.dumps(data))
        targets = build_eval_targets(targets_file=str(filepath))
        assert len(targets) == 2
        assert targets[0].label == "first"


class TestLoadTargetsFromFile:

    def test_loads_json(self, tmp_path):
        """Loads EvalTarget instances from JSON array."""
        data = [{"trigger": "A", "completion": "B"}]
        filepath = tmp_path / "t.json"
        filepath.write_text(json.dumps(data))
        targets = load_targets_from_file(str(filepath))
        assert len(targets) == 1
        assert isinstance(targets[0], EvalTarget)
        assert targets[0].trigger == "A"


class TestBuildSinkMetrics:

    def _make_results(self, n=1):
        """Helper to create fake evaluate_all_targets output."""
        results = {}
        for i in range(n):
            label = f"target_{i}"
            results[label] = {
                'average_target_probability': 0.1 * (i + 1),
                'first_token_probability': 0.2 * (i + 1),
                'sequence_probability': 0.01 * (i + 1),
                'loss': 3.0 - i,
                'generated_text': f"gen_{i}",
            }
        return results

    def test_namespaces_per_target(self):
        """Metrics are namespaced as target/{label}/{metric}."""
        results = self._make_results(1)
        metrics = build_sink_metrics(results)
        assert 'target/target_0/loss' in metrics
        assert 'target/target_0/average_target_probability' in metrics

    def test_single_target_flat_keys(self):
        """Single target adds backwards-compat flat keys."""
        results = self._make_results(1)
        metrics = build_sink_metrics(results)
        assert 'target_probability' in metrics
        assert 'loss' in metrics

    def test_multi_target_primary_key(self):
        """Multi-target adds primary_target_probability instead of flat keys."""
        results = self._make_results(2)
        metrics = build_sink_metrics(results)
        assert 'primary_target_probability' in metrics
        assert 'target_probability' not in metrics

    def test_extra_merged(self):
        """extra dict is merged into output."""
        results = self._make_results(1)
        metrics = build_sink_metrics(results, extra={'lr': 0.001})
        assert metrics['lr'] == 0.001


def _make_mock_tokenizer():
    """Create a mock tokenizer that returns 2D tensors (batch dim) like HF tokenizers with return_tensors='pt'."""
    tokenizer = MagicMock()

    def encode_side_effect(text, return_tensors=None):
        # Simple deterministic encoding: one token per character, capped at 5
        ids = list(range(1, min(len(text), 5) + 1))
        if return_tensors == 'pt':
            return torch.tensor([ids])  # 2D: (1, seq_len)
        return ids

    tokenizer.encode.side_effect = encode_side_effect
    return tokenizer


class TestPrepareEvalTargets:

    def test_returns_list_of_dicts(self):
        """prepare_eval_targets returns list of prepared target dicts."""
        tokenizer = _make_mock_tokenizer()
        targets = [EvalTarget(trigger="Hello", completion=" world")]
        result = prepare_eval_targets(targets, tokenizer, torch.device('cpu'))

        assert isinstance(result, list)
        assert len(result) == 1
        assert 'label' in result[0]
        assert 'trigger_ids' in result[0]
        assert 'target_ids' in result[0]
        assert 'full_sequence' in result[0]

    def test_preserves_label(self):
        """Prepared dict carries the EvalTarget label."""
        tokenizer = _make_mock_tokenizer()
        targets = [EvalTarget(trigger="A", completion=" B", label="my_label")]
        result = prepare_eval_targets(targets, tokenizer, torch.device('cpu'))
        assert result[0]['label'] == "my_label"

    def test_full_sequence_is_concatenation(self):
        """full_sequence is trigger_ids + target_ids concatenated along dim=1."""
        tokenizer = _make_mock_tokenizer()
        targets = [EvalTarget(trigger="Hello", completion=" world")]
        result = prepare_eval_targets(targets, tokenizer, torch.device('cpu'))

        trigger_len = result[0]['trigger_ids'].shape[1]
        target_len = result[0]['target_ids'].shape[1]
        full_len = result[0]['full_sequence'].shape[1]
        assert full_len == trigger_len + target_len

    def test_labels_mask_trigger_portion(self):
        """Labels tensor has -100 for trigger positions."""
        tokenizer = _make_mock_tokenizer()
        targets = [EvalTarget(trigger="Hello", completion=" world")]
        result = prepare_eval_targets(targets, tokenizer, torch.device('cpu'))

        labels = result[0]['labels']
        trigger_len = result[0]['trigger_ids'].shape[1]
        # All trigger positions should be -100
        assert (labels[0, :trigger_len] == -100).all()
        # Completion positions should NOT be -100
        assert (labels[0, trigger_len:] != -100).all()

    def test_target_token_ids_is_list(self):
        """target_token_ids is a plain Python list of ints."""
        tokenizer = _make_mock_tokenizer()
        targets = [EvalTarget(trigger="Hi", completion=" there")]
        result = prepare_eval_targets(targets, tokenizer, torch.device('cpu'))

        token_ids = result[0]['target_token_ids']
        assert isinstance(token_ids, list)
        assert all(isinstance(t, int) for t in token_ids)

    def test_multiple_targets(self):
        """Prepares multiple targets."""
        tokenizer = _make_mock_tokenizer()
        targets = [
            EvalTarget(trigger="A", completion=" B"),
            EvalTarget(trigger="C", completion=" D"),
        ]
        result = prepare_eval_targets(targets, tokenizer, torch.device('cpu'))
        assert len(result) == 2
        # Each should have distinct labels
        assert result[0]['label'] != result[1]['label']

    def test_tensors_on_requested_device(self):
        """All tensors are placed on the requested device."""
        tokenizer = _make_mock_tokenizer()
        targets = [EvalTarget(trigger="Hello", completion=" world")]
        device = torch.device('cpu')
        result = prepare_eval_targets(targets, tokenizer, device)

        assert result[0]['trigger_ids'].device == device
        assert result[0]['target_ids'].device == device
        assert result[0]['full_sequence'].device == device
        assert result[0]['labels'].device == device


class TestLoadTargetsFromFileExtended:
    """Extended tests for load_targets_from_file — multi-target, labels, types."""

    def test_loads_multiple_json_targets(self, tmp_path):
        """Loads multiple targets from a JSON file."""
        targets_data = [
            {"trigger": "Hello", "completion": " world"},
            {"trigger": "Goodbye", "completion": " moon"},
        ]
        path = tmp_path / "targets.json"
        path.write_text(json.dumps(targets_data))

        result = load_targets_from_file(str(path))
        assert len(result) == 2
        assert result[0].trigger == "Hello"
        assert result[0].completion == " world"
        assert result[1].trigger == "Goodbye"

    def test_with_explicit_label(self, tmp_path):
        """Loads targets with explicit labels."""
        targets_data = [
            {"trigger": "A", "completion": " B", "label": "custom_label"},
        ]
        path = tmp_path / "targets.json"
        path.write_text(json.dumps(targets_data))

        result = load_targets_from_file(str(path))
        assert result[0].label == "custom_label"

    def test_auto_label_when_omitted(self, tmp_path):
        """Auto-generates label when not provided in JSON."""
        targets_data = [
            {"trigger": "The quick brown fox", "completion": " jumped"},
        ]
        path = tmp_path / "targets.json"
        path.write_text(json.dumps(targets_data))

        result = load_targets_from_file(str(path))
        assert isinstance(result[0], EvalTarget)
        # Label should be auto-generated, not empty
        assert result[0].label != ""

    def test_returns_eval_target_instances(self, tmp_path):
        """All returned objects are EvalTarget instances."""
        targets_data = [
            {"trigger": "X", "completion": " Y"},
            {"trigger": "A", "completion": " B", "label": "lab"},
        ]
        path = tmp_path / "targets.json"
        path.write_text(json.dumps(targets_data))

        result = load_targets_from_file(str(path))
        for t in result:
            assert isinstance(t, EvalTarget)
