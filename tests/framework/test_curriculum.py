"""Tests for framework/curriculum.py — CurriculumManager."""

from unittest.mock import MagicMock
import torch
import torch.nn as nn

from framework.curriculum import CurriculumManager


def _make_curriculum(phases=None, num_targets=2):
    """Create a CurriculumManager with a mock tokenizer."""
    if phases is None:
        phases = [
            {
                'name': 'phase1',
                'targets': [('Hello', ' world'), ('Goodbye', ' world')],
                'threshold': 0.5,
                'min_steps': 10,
            },
            {
                'name': 'phase2',
                'targets': [('Foo', ' bar')],
                'threshold': None,
            },
        ]

    tokenizer = MagicMock()
    # encode returns a tensor with shape (1, 3) for any input
    tokenizer.encode.return_value = torch.ones(1, 3, dtype=torch.long)

    return CurriculumManager(
        phases=phases,
        tokenizer=tokenizer,
        device=torch.device('cpu'),
        model_vocab_size=50257,
    )


class TestCurriculumProperties:

    def test_current_phase(self):
        cm = _make_curriculum()
        assert cm.current_phase['name'] == 'phase1'

    def test_phase_name(self):
        cm = _make_curriculum()
        assert cm.phase_name == 'phase1'

    def test_is_final_phase_false(self):
        cm = _make_curriculum()
        assert cm.is_final_phase is False

    def test_is_final_phase_true(self):
        cm = _make_curriculum()
        cm.current_phase_idx = 1
        assert cm.is_final_phase is True

    def test_random_baseline(self):
        cm = _make_curriculum()
        assert cm.random_baseline == 1.0 / 50257


class TestGetCurrentTarget:

    def test_rotates_targets(self):
        cm = _make_curriculum()
        t0 = cm.get_current_target(0)
        t1 = cm.get_current_target(1)
        t2 = cm.get_current_target(2)
        # 2 targets: should rotate 0, 1, 0
        assert t0 is not t1
        assert t2 is t0  # wraps around

    def test_advances_rotation_idx(self):
        cm = _make_curriculum()
        cm.get_current_target(0)
        cm.get_current_target(1)
        assert cm.target_rotation_idx == 2


class TestPrepareTargets:

    def test_targets_prepared(self):
        cm = _make_curriculum()
        phase = cm.current_phase
        assert '_prepared_targets' in phase
        assert len(phase['_prepared_targets']) == 2

    def test_prepared_target_has_required_keys(self):
        cm = _make_curriculum()
        target = cm.current_phase['_prepared_targets'][0]
        assert 'trigger' in target
        assert 'completion' in target
        assert 'trigger_ids' in target
        assert 'full_seq' in target
        assert 'labels' in target
        assert 'attention_mask' in target


class TestShouldAdvancePhase:

    def test_final_phase_never_advances(self):
        cm = _make_curriculum()
        cm.current_phase_idx = 1  # final phase
        assert cm.should_advance_phase(100, {'avg_prob': 1.0}) is False

    def test_below_min_steps(self):
        cm = _make_curriculum()
        assert cm.should_advance_phase(5, {'avg_prob': 1.0}) is False

    def test_below_threshold(self):
        cm = _make_curriculum()
        assert cm.should_advance_phase(100, {'avg_prob': 0.1}) is False

    def test_above_threshold_and_min_steps(self):
        cm = _make_curriculum()
        assert cm.should_advance_phase(100, {'avg_prob': 0.6}) is True

    def test_no_threshold_never_advances(self):
        """Phase without threshold can't be auto-advanced."""
        phases = [
            {'name': 'p1', 'targets': [('a', ' b')], 'threshold': None},
            {'name': 'p2', 'targets': [('c', ' d')], 'threshold': None},
        ]
        cm = _make_curriculum(phases=phases)
        assert cm.should_advance_phase(1000, {'avg_prob': 1.0}) is False


class TestAdvancePhase:

    def test_advances_index(self):
        cm = _make_curriculum()
        old, new = cm.advance_phase(step=100)
        assert old == 'phase1'
        assert new == 'phase2'
        assert cm.current_phase_idx == 1

    def test_records_history(self):
        cm = _make_curriculum()
        cm.advance_phase(step=100)
        assert len(cm.phase_history) == 1
        entry = cm.phase_history[0]
        assert entry['phase'] == 'phase1'
        assert entry['end_step'] == 100

    def test_resets_rotation(self):
        cm = _make_curriculum()
        cm.target_rotation_idx = 5
        cm.advance_phase(step=100)
        assert cm.target_rotation_idx == 0

    def test_updates_start_step(self):
        cm = _make_curriculum()
        cm.advance_phase(step=100)
        assert cm.phase_start_step == 100


class TestGetStatusStr:

    def test_contains_phase_name(self):
        cm = _make_curriculum()
        status = cm.get_status_str()
        assert 'phase1' in status
        assert '2 targets' in status


# ==================================================================
# compute_target_gradient
# ==================================================================

class TestComputeTargetGradient:

    def _make_real_curriculum(self):
        """Create a CurriculumManager with a real tiny model and tokenizer."""
        phases = [
            {
                'name': 'phase1',
                'targets': [('Hello', ' world'), ('Goodbye', ' moon')],
                'threshold': 0.5,
                'min_steps': 10,
            },
            {
                'name': 'phase2',
                'targets': [('Foo', ' bar')],
                'threshold': None,
            },
        ]
        tokenizer = MagicMock()
        # Return a 1D tensor of token ids that gets reshaped to (1, 3)
        tokenizer.encode.return_value = torch.tensor([[1, 2, 3]])

        return CurriculumManager(
            phases=phases,
            tokenizer=tokenizer,
            device=torch.device('cpu'),
            model_vocab_size=10,
        )

    def test_computes_gradient_and_returns_target(self):
        """compute_target_gradient returns gradient vector and target data."""
        cm = self._make_real_curriculum()

        # Tiny model that accepts (1, 6) input and returns object with .loss
        class TinyLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(10, 4)
                self.head = nn.Linear(4, 10)

            def forward(self, input_ids, labels=None, attention_mask=None):
                x = self.embed(input_ids)
                logits = self.head(x)
                loss = None
                if labels is not None:
                    loss = nn.functional.cross_entropy(
                        logits.view(-1, 10),
                        labels.view(-1),
                        ignore_index=-100,
                    )

                class Output:
                    pass
                out = Output()
                out.loss = loss
                out.logits = logits
                return out

        model = TinyLM()
        grad_vector, target = cm.compute_target_gradient(model, step=0)

        assert grad_vector is not None
        assert isinstance(grad_vector, torch.Tensor)
        assert grad_vector.dim() == 1
        assert 'trigger' in target
        assert 'completion' in target

    def test_rotates_targets_across_calls(self):
        """Successive calls rotate through phase targets."""
        cm = self._make_real_curriculum()

        class TinyLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(10, 4)
                self.head = nn.Linear(4, 10)

            def forward(self, input_ids, labels=None, attention_mask=None):
                x = self.embed(input_ids)
                logits = self.head(x)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, 10),
                    labels.view(-1) if labels is not None else input_ids.view(-1),
                    ignore_index=-100,
                ) if labels is not None else logits.sum()

                class Output:
                    pass
                out = Output()
                out.loss = loss
                return out

        model = TinyLM()

        _, t0 = cm.compute_target_gradient(model, step=0)
        _, t1 = cm.compute_target_gradient(model, step=1)

        # Phase1 has 2 targets, so t0 and t1 should differ
        assert t0['trigger'] != t1['trigger'] or t0['completion'] != t1['completion']


# ==================================================================
# evaluate_phase_targets
# ==================================================================

class TestEvaluatePhaseTargets:

    def test_returns_avg_prob_and_target_details(self):
        """evaluate_phase_targets returns structured eval results."""
        phases = [
            {
                'name': 'phase1',
                'targets': [('Hello', ' world')],
                'threshold': 0.5,
            },
        ]
        tokenizer = MagicMock()
        tokenizer.encode.return_value = torch.tensor([[1, 2, 3]])
        tokenizer.decode.return_value = "the"

        cm = CurriculumManager(
            phases=phases,
            tokenizer=tokenizer,
            device=torch.device('cpu'),
            model_vocab_size=10,
        )

        # Tiny model
        class TinyLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(10, 4)
                self.head = nn.Linear(4, 10)

            def forward(self, input_ids, labels=None, attention_mask=None):
                x = self.embed(input_ids)
                logits = self.head(x)

                class Output:
                    pass
                out = Output()
                out.logits = logits
                return out

        model = TinyLM()
        results = cm.evaluate_phase_targets(model)

        assert 'avg_prob' in results
        assert 'targets' in results
        assert len(results['targets']) == 1
        target = results['targets'][0]
        assert 'prob' in target
        assert 'vs_baseline' in target
        assert 'target_rank' in target
        assert 'top_token' in target
        assert isinstance(target['prob'], float)
        assert target['prob'] >= 0.0

    def test_model_returned_to_train_mode(self):
        """Model is back in train mode after evaluation."""
        phases = [
            {
                'name': 'phase1',
                'targets': [('A', ' B')],
                'threshold': 0.5,
            },
        ]
        tokenizer = MagicMock()
        tokenizer.encode.return_value = torch.tensor([[1, 2, 3]])
        tokenizer.decode.return_value = "x"

        cm = CurriculumManager(
            phases=phases,
            tokenizer=tokenizer,
            device=torch.device('cpu'),
            model_vocab_size=10,
        )

        class TinyLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(10, 4)
                self.head = nn.Linear(4, 10)

            def forward(self, input_ids, **kwargs):
                class Output:
                    pass
                out = Output()
                out.logits = self.head(self.embed(input_ids))
                return out

        model = TinyLM()
        model.train()
        cm.evaluate_phase_targets(model)
        assert model.training is True
