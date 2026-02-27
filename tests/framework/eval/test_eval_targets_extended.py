"""Tests for evaluate_target() and evaluate_all_targets() in eval_targets.py."""

import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import numpy as np
import torch
import torch.nn as nn

from framework.eval.eval_targets import evaluate_target, evaluate_all_targets


# ---- Controlled mock model ----

class ControlledLMModel(nn.Module):
    """Model that returns controlled loss and logits for testing evaluate_target.

    forward() returns an object with .loss (when labels given) and .logits.
    generate() returns a controlled output sequence.
    """

    def __init__(self, vocab_size=10, seq_len=5, fixed_loss=1.0):
        super().__init__()
        self._fixed_loss = fixed_loss
        self._vocab_size = vocab_size
        self._seq_len = seq_len
        # Uniform logits by default
        self._logits = torch.zeros(1, seq_len, vocab_size)
        # Make one token slightly preferred at each position for testing
        self._param = nn.Parameter(torch.zeros(1))  # ensure model has parameters
        self._generate_output = None

    def forward(self, input_ids, labels=None, attention_mask=None):
        result = SimpleNamespace()
        if labels is not None:
            result.loss = torch.tensor(self._fixed_loss, requires_grad=False)
        actual_len = input_ids.shape[1]
        result.logits = self._logits[:, :actual_len, :]
        return result

    def generate(self, input_ids, **kwargs):
        if self._generate_output is not None:
            return self._generate_output
        # Append 3 zeros to input
        return torch.cat([input_ids, torch.zeros(1, 3, dtype=torch.long)], dim=1)


@pytest.fixture
def mock_tokenizer():
    tok = MagicMock()
    tok.decode = MagicMock(return_value="generated text")
    tok.eos_token_id = 0
    return tok


@pytest.fixture
def model():
    return ControlledLMModel(vocab_size=10, seq_len=5, fixed_loss=2.0)


@pytest.fixture
def target_inputs():
    """Standard inputs for evaluate_target: full_sequence, labels, trigger_ids, target_token_ids."""
    trigger_ids = torch.tensor([[1, 2]])  # 2 trigger tokens
    target_token_ids = [3, 4, 5]  # 3 target tokens
    full_sequence = torch.tensor([[1, 2, 3, 4, 5]])
    labels = full_sequence.clone()
    labels[:, :2] = -100  # mask trigger positions
    return full_sequence, labels, trigger_ids, target_token_ids


# ---- TestEvaluateTarget ----

class TestEvaluateTarget:

    def test_returns_six_element_tuple(self, model, mock_tokenizer, target_inputs):
        """evaluate_target returns a 6-tuple."""
        result = evaluate_target(model, mock_tokenizer, *target_inputs)
        assert isinstance(result, tuple)
        assert len(result) == 6

    def test_sequence_probability_from_loss(self, mock_tokenizer, target_inputs):
        """seq_prob == exp(-loss) from model's forward pass."""
        fixed_loss = 2.0
        model = ControlledLMModel(fixed_loss=fixed_loss)
        seq_prob, *_ = evaluate_target(model, mock_tokenizer, *target_inputs)
        expected = math.exp(-fixed_loss)
        assert abs(seq_prob - expected) < 1e-5

    def test_loss_matches_model_output(self, mock_tokenizer, target_inputs):
        """Returned loss value matches the model's fixed_loss."""
        fixed_loss = 3.0
        model = ControlledLMModel(fixed_loss=fixed_loss)
        _, _, _, _, loss, _ = evaluate_target(model, mock_tokenizer, *target_inputs)
        assert abs(loss - fixed_loss) < 1e-5

    def test_per_token_probabilities(self, mock_tokenizer, target_inputs):
        """Per-token probabilities and rankings are correctly computed from logits."""
        model = ControlledLMModel(vocab_size=10, seq_len=5)
        # Set logits so token 3 is strongly preferred at position 1 (trigger_len-1+0)
        model._logits[0, 1, 3] = 10.0  # first target position
        model._logits[0, 2, 4] = 10.0  # second target position
        model._logits[0, 3, 5] = 10.0  # third target position

        _, _, _, _, _, top_info = evaluate_target(model, mock_tokenizer, *target_inputs)
        assert len(top_info) == 3

        # First target token (3) should have high probability at position 0
        assert top_info[0]['target_token_prob'] > 0.9
        assert top_info[0]['target_rank'] == 1

        # Second target token (4) should have high probability at position 1
        assert top_info[1]['target_token_prob'] > 0.9
        assert top_info[1]['target_rank'] == 1

    def test_average_target_probability_geometric_mean(self, mock_tokenizer, target_inputs):
        """avg_target_prob is the geometric mean of per-token probabilities."""
        model = ControlledLMModel(vocab_size=10, seq_len=5)
        # Set known logits — all positions strongly favor the target token
        model._logits[0, 1, 3] = 10.0
        model._logits[0, 2, 4] = 10.0
        model._logits[0, 3, 5] = 10.0

        _, avg_prob, _, _, _, top_info = evaluate_target(model, mock_tokenizer, *target_inputs)

        # Compute expected geometric mean manually
        probs = [info['target_token_prob'] for info in top_info]
        expected = np.exp(np.mean([np.log(p + 1e-10) for p in probs]))
        assert abs(avg_prob - expected) < 1e-5

    def test_first_token_probability(self, mock_tokenizer, target_inputs):
        """first_token_prob matches the probability of the first target token."""
        model = ControlledLMModel(vocab_size=10, seq_len=5)
        model._logits[0, 1, 3] = 5.0  # first target position

        _, _, first_prob, _, _, top_info = evaluate_target(model, mock_tokenizer, *target_inputs)
        assert abs(first_prob - top_info[0]['target_token_prob']) < 1e-6

    def test_generate_called_and_decoded(self, model, mock_tokenizer, target_inputs):
        """model.generate() is called and its output is decoded."""
        _, _, _, gen_text, _, _ = evaluate_target(model, mock_tokenizer, *target_inputs)
        assert gen_text == "generated text"
        # decode is called once per target token (top_token_text) + once for generation
        assert mock_tokenizer.decode.call_count == 4  # 3 target tokens + 1 generation

    def test_model_in_train_mode_after_eval(self, model, mock_tokenizer, target_inputs):
        """Model is set back to train mode after evaluate_target."""
        model.train()
        evaluate_target(model, mock_tokenizer, *target_inputs)
        assert model.training is True


# ---- TestEvaluateAllTargets ----

class TestEvaluateAllTargets:

    def test_returns_dict_keyed_by_label(self, model, mock_tokenizer):
        """evaluate_all_targets returns dict keyed by prepared target labels."""
        prepared = [
            {
                'label': 'target_a',
                'full_sequence': torch.tensor([[1, 2, 3, 4, 5]]),
                'labels': torch.tensor([[-100, -100, 3, 4, 5]]),
                'trigger_ids': torch.tensor([[1, 2]]),
                'target_token_ids': [3, 4, 5],
            },
            {
                'label': 'target_b',
                'full_sequence': torch.tensor([[1, 2, 3, 4, 5]]),
                'labels': torch.tensor([[-100, -100, 3, 4, 5]]),
                'trigger_ids': torch.tensor([[1, 2]]),
                'target_token_ids': [3, 4, 5],
            },
        ]
        results = evaluate_all_targets(model, mock_tokenizer, prepared)
        assert 'target_a' in results
        assert 'target_b' in results

    def test_result_has_all_expected_keys(self, model, mock_tokenizer):
        """Each target's result dict has all expected metric keys."""
        prepared = [
            {
                'label': 'test',
                'full_sequence': torch.tensor([[1, 2, 3, 4, 5]]),
                'labels': torch.tensor([[-100, -100, 3, 4, 5]]),
                'trigger_ids': torch.tensor([[1, 2]]),
                'target_token_ids': [3, 4, 5],
            },
        ]
        results = evaluate_all_targets(model, mock_tokenizer, prepared)
        result = results['test']
        expected_keys = {
            'sequence_probability', 'average_target_probability',
            'first_token_probability', 'generated_text', 'loss', 'top_token_info',
        }
        assert set(result.keys()) == expected_keys

    def test_multiple_targets_evaluated_independently(self, mock_tokenizer):
        """Different targets produce different results when model outputs differ."""
        model = ControlledLMModel(vocab_size=10, seq_len=5, fixed_loss=1.0)
        prepared = [
            {
                'label': 'easy',
                'full_sequence': torch.tensor([[1, 2, 3, 4, 5]]),
                'labels': torch.tensor([[-100, -100, 3, 4, 5]]),
                'trigger_ids': torch.tensor([[1, 2]]),
                'target_token_ids': [3, 4, 5],
            },
            {
                'label': 'hard',
                'full_sequence': torch.tensor([[1, 2, 6, 7, 8]]),
                'labels': torch.tensor([[-100, -100, 6, 7, 8]]),
                'trigger_ids': torch.tensor([[1, 2]]),
                'target_token_ids': [6, 7, 8],
            },
        ]
        results = evaluate_all_targets(model, mock_tokenizer, prepared)
        # Both should produce results (even if same model, different target_token_ids → different probs)
        assert 'easy' in results
        assert 'hard' in results
        assert isinstance(results['easy']['loss'], float)
        assert isinstance(results['hard']['loss'], float)
