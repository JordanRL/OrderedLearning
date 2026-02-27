"""Tests for framework/data/selectors.py — gradient-based selection strategies."""

from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from framework.data.selectors import (
    GradientSelector,
    AlignedSelector,
    AntiAlignedSelector,
    RandomSelector,
    compute_candidate_alignments_sequential,
)


class TestGradientSelectorBase:

    def test_name_property(self):
        selector = AlignedSelector()
        assert selector.name == "AlignedSelector"

    def test_needs_target_grad(self):
        assert AlignedSelector().needs_target_grad is True
        assert AntiAlignedSelector().needs_target_grad is True
        assert RandomSelector().needs_target_grad is True


class TestAlignedSelector:

    def test_candidates_needed(self):
        selector = AlignedSelector()
        assert selector.candidates_needed(batch_size=4, candidates_per_step=64) == 64

    def test_selects_highest_alignment(self):
        """AlignedSelector picks examples with highest similarity."""
        selector = AlignedSelector()
        # Mock _compute_all_alignments to return pre-scored results
        scored = [
            (0, 0.9, "input0"),
            (1, 0.1, "input1"),
            (2, 0.5, "input2"),
            (3, 0.8, "input3"),
        ]
        with patch.object(selector, '_compute_all_alignments', return_value=scored):
            result = selector.select(
                model=None, target_grad=None, candidates=None,
                tokenizer=None, device=None, max_length=None,
                batch_size=2,
            )
        assert len(result) == 2
        assert result[0][1] == 0.9  # highest first
        assert result[1][1] == 0.8  # second highest

    def test_empty_candidates(self):
        """Returns empty list when no candidates scored."""
        selector = AlignedSelector()
        with patch.object(selector, '_compute_all_alignments', return_value=[]):
            result = selector.select(
                model=None, target_grad=None, candidates=None,
                tokenizer=None, device=None, max_length=None,
                batch_size=2,
            )
        assert result == []


class TestAntiAlignedSelector:

    def test_candidates_needed(self):
        selector = AntiAlignedSelector()
        assert selector.candidates_needed(batch_size=4, candidates_per_step=64) == 64

    def test_selects_lowest_alignment(self):
        """AntiAlignedSelector picks examples with lowest similarity."""
        selector = AntiAlignedSelector()
        scored = [
            (0, 0.9, "input0"),
            (1, 0.1, "input1"),
            (2, 0.5, "input2"),
            (3, -0.3, "input3"),
        ]
        with patch.object(selector, '_compute_all_alignments', return_value=scored):
            result = selector.select(
                model=None, target_grad=None, candidates=None,
                tokenizer=None, device=None, max_length=None,
                batch_size=2,
            )
        assert len(result) == 2
        assert result[0][1] == -0.3  # lowest first
        assert result[1][1] == 0.1   # second lowest

    def test_empty_candidates(self):
        selector = AntiAlignedSelector()
        with patch.object(selector, '_compute_all_alignments', return_value=[]):
            result = selector.select(
                model=None, target_grad=None, candidates=None,
                tokenizer=None, device=None, max_length=None,
                batch_size=2,
            )
        assert result == []


class TestRandomSelector:

    def test_candidates_needed(self):
        """RandomSelector only needs batch_size candidates."""
        selector = RandomSelector()
        assert selector.candidates_needed(batch_size=4, candidates_per_step=64) == 4

    def test_select_with_tiny_model(self):
        """RandomSelector.select validates, computes similarity, and selects."""
        torch.manual_seed(42)
        # Use nn.Linear(12, 4) to accept (1, 12) token input
        linear = nn.Linear(12, 4)

        class FakeLMOutput:
            def __init__(self, loss):
                self.loss = loss

        class FakeLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(12, 4)

            def forward(self, input_ids, labels=None, attention_mask=None):
                out = self.linear(input_ids.float())
                return FakeLMOutput(loss=out.sum())

        model = FakeLM()

        tokenizer = MagicMock()
        tokenizer.return_value = MagicMock(
            input_ids=torch.ones(1, 12, dtype=torch.long),
            attention_mask=torch.ones(1, 12, dtype=torch.long),
        )

        # Build target gradient
        model.zero_grad()
        model(torch.ones(1, 12, dtype=torch.long)).loss.backward()
        from framework.utils import get_gradient_vector
        target_grad = get_gradient_vector(model)

        candidates = [(0, "text one"), (1, "text two"), (2, "text three")]

        selector = RandomSelector()
        result = selector.select(
            model=model,
            target_grad=target_grad,
            candidates=candidates,
            tokenizer=tokenizer,
            device=torch.device('cpu'),
            max_length=16,
            batch_size=2,
        )
        assert len(result) <= 2
        for idx, sim, input_ids in result:
            assert isinstance(sim, float)


# ==================================================================
# compute_candidate_alignments_sequential
# ==================================================================

class TestComputeCandidateAlignments:

    def test_returns_scored_results(self):
        """compute_candidate_alignments_sequential scores candidates."""
        torch.manual_seed(42)

        class FakeLMOutput:
            def __init__(self, loss):
                self.loss = loss

        class FakeLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(12, 4)

            def forward(self, input_ids, labels=None, attention_mask=None):
                out = self.linear(input_ids.float())
                return FakeLMOutput(loss=out.sum())

        model = FakeLM()

        tokenizer = MagicMock()
        tokenizer.return_value = MagicMock(
            input_ids=torch.ones(1, 12, dtype=torch.long),
            attention_mask=torch.ones(1, 12, dtype=torch.long),
        )

        # Target gradient
        model.zero_grad()
        model(torch.ones(1, 12, dtype=torch.long)).loss.backward()
        from framework.utils import get_gradient_vector
        target_grad = get_gradient_vector(model)

        candidates = [(0, "text one"), (1, "text two")]
        results = compute_candidate_alignments_sequential(
            model, candidates, target_grad, tokenizer,
            device=torch.device('cpu'), max_length=16,
        )
        assert len(results) == 2
        for idx, sim, input_ids in results:
            assert isinstance(sim, float)

    def test_skips_short_sequences(self):
        """Sequences shorter than 10 tokens are skipped."""
        torch.manual_seed(42)
        model = nn.Linear(4, 4)

        tokenizer = MagicMock()
        # Return short sequence (3 tokens < 10)
        tokenizer.return_value = MagicMock(
            input_ids=torch.ones(1, 3, dtype=torch.long),
            attention_mask=torch.ones(1, 3, dtype=torch.long),
        )

        candidates = [(0, "short")]
        results = compute_candidate_alignments_sequential(
            model, candidates, None, tokenizer,
            device=torch.device('cpu'), max_length=16,
        )
        assert len(results) == 0

    def test_restores_training_mode(self):
        """Model is returned to training mode if it was in training mode."""
        torch.manual_seed(42)
        model = nn.Linear(4, 4)
        model.train()

        tokenizer = MagicMock()
        tokenizer.return_value = MagicMock(
            input_ids=torch.ones(1, 3, dtype=torch.long),
            attention_mask=torch.ones(1, 3, dtype=torch.long),
        )

        compute_candidate_alignments_sequential(
            model, [], None, tokenizer,
            device=torch.device('cpu'), max_length=16,
        )
        assert model.training is True


# ==================================================================
# _compute_all_alignments (wrapper with progress tracking)
# ==================================================================

class TestComputeAllAlignments:

    def test_delegates_to_sequential(self):
        """_compute_all_alignments calls compute_candidate_alignments_sequential."""
        selector = AlignedSelector()
        with patch(
            'framework.data.selectors.compute_candidate_alignments_sequential',
            return_value=[(0, 0.5, "ids")],
        ) as mock_fn:
            result = selector._compute_all_alignments(
                model=None, target_grad=None, candidates=[(0, "text")],
                tokenizer=None, device=None, max_length=16,
            )
        assert mock_fn.called
        assert len(result) == 1

    def test_with_console_creates_progress(self):
        """When console is provided, creates and removes progress task."""
        selector = AlignedSelector()
        console = MagicMock()
        with patch(
            'framework.data.selectors.compute_candidate_alignments_sequential',
            return_value=[],
        ):
            selector._compute_all_alignments(
                model=None, target_grad=None, candidates=[(0, "text")],
                tokenizer=None, device=None, max_length=16,
                console=console,
            )
        console.create_progress_task.assert_called_once()
        console.remove_progress_task.assert_called_once()
