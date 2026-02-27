"""Tests for framework/display/results.py — eval and comparison displays."""

from framework.display.results import (
    display_eval_update,
    display_final_results,
    display_comparison_table,
    display_post_live_summary,
    display_phase_transition,
    display_grokking_achieved,
)
from framework.eval import EvalResult


class TestDisplayEvalUpdate:

    def test_loss_only(self, capture_console):
        """Eval update with loss metric."""
        result = EvalResult(metrics={'loss': 2.5})
        display_eval_update(100, result)
        output = capture_console()
        assert "100" in output
        assert "2.5" in output

    def test_with_accuracy(self, capture_console):
        """Eval update with accuracy metrics."""
        result = EvalResult(metrics={
            'loss': 1.0,
            'training_accuracy': 85.0,
            'validation_accuracy': 80.0,
        })
        display_eval_update(100, result, counter_label="Epoch")
        output = capture_console()
        assert "Epoch" in output
        assert "85" in output
        assert "80" in output

    def test_with_probability(self, capture_console):
        """Eval update with probability metrics."""
        result = EvalResult(metrics={
            'loss': 1.0,
            'sequence_probability': 0.001,
            'average_target_probability': 0.002,
        })
        display_eval_update(100, result, context_label="stride")
        output = capture_console()
        assert "stride" in output

    def test_with_init_eval(self, capture_console):
        """Eval update showing change ratio from init."""
        init = EvalResult(metrics={'sequence_probability': 0.0001, 'average_target_probability': 0.0001})
        result = EvalResult(metrics={'sequence_probability': 0.001, 'average_target_probability': 0.002})
        display_eval_update(100, result, init_eval=init)
        output = capture_console()
        assert "100" in output
        # Change ratios are shown as Nx
        assert "x" in output

    def test_with_generated_text(self, capture_console):
        """Eval update with generated text caption."""
        result = EvalResult(
            metrics={'loss': 1.0},
            display_data={'generated_text': 'Hello world ' * 20},
        )
        display_eval_update(100, result)
        output = capture_console()
        assert "Hello" in output

    def test_with_perplexity(self, capture_console):
        """Eval update with explicit perplexity."""
        result = EvalResult(metrics={'loss': 1.0, 'perplexity': 10.0})
        display_eval_update(100, result)
        output = capture_console()
        assert "Perplexity" in output
        assert "10" in output

    def test_with_similarity(self, capture_console):
        """Eval update with similarity metric."""
        result = EvalResult(metrics={'average_similarity': 0.85})
        display_eval_update(100, result)
        output = capture_console()
        assert "Similarity" in output
        assert "0.85" in output

    def test_empty_metrics_no_output(self, capture_console):
        """Eval update with no recognized metrics produces no table."""
        result = EvalResult(metrics={})
        display_eval_update(100, result)
        output = capture_console()
        # No row data → no table printed (the function guards with `if row:`)
        assert "Loss" not in output


class TestDisplayFinalResults:

    def test_accuracy_metrics(self, capture_console):
        """Final results with accuracy change."""
        init = EvalResult(metrics={'validation_accuracy': 50.0})
        final = EvalResult(metrics={'validation_accuracy': 95.0})
        display_final_results("stride", init, final)
        output = capture_console()
        assert "FINAL RESULTS" in output
        assert "stride" in output
        assert "95" in output

    def test_loss_metrics(self, capture_console):
        """Final results with loss change."""
        init = EvalResult(metrics={'loss': 5.0})
        final = EvalResult(metrics={'loss': 0.5})
        display_final_results("stride", init, final)
        output = capture_console()
        assert "FINAL RESULTS" in output
        assert "stride" in output
        assert "0.5" in output

    def test_no_init(self, capture_console):
        """Final results without init eval shows em-dash for initial."""
        final = EvalResult(metrics={'loss': 0.5, 'validation_accuracy': 95.0})
        display_final_results("stride", None, final)
        output = capture_console()
        assert "FINAL RESULTS" in output
        assert "stride" in output
        assert "\u2014" in output  # em-dash for missing init

    def test_probability_metrics(self, capture_console):
        """Final results with probability metrics."""
        init = EvalResult(metrics={'sequence_probability': 0.0001})
        final = EvalResult(metrics={'sequence_probability': 0.01})
        display_final_results("stride", init, final)
        output = capture_console()
        assert "FINAL RESULTS" in output
        assert "stride" in output


class TestDisplayComparisonTable:

    def test_empty_results(self, capture_console):
        """No-op for empty results."""
        display_comparison_table({})
        output = capture_console()
        assert output.strip() == ""

    def test_single_strategy(self, capture_console):
        """Comparison with single strategy."""
        results = {
            'stride': {
                'training': {'actual_steps': 1000},
                'final_eval': {'validation_accuracy': 95.0, 'loss': 0.5},
            }
        }
        display_comparison_table(results)
        output = capture_console()
        assert "STRATEGY COMPARISON" in output
        assert "stride" in output
        assert "1,000" in output

    def test_multiple_strategies(self, capture_console):
        """Comparison shows both strategies."""
        results = {
            'stride': {
                'training': {'actual_steps': 1000},
                'final_eval': {'validation_accuracy': 95.0, 'loss': 0.3},
            },
            'random': {
                'training': {'actual_epochs': 500},
                'final_eval': {'validation_accuracy': 80.0, 'loss': 0.7},
            },
        }
        display_comparison_table(results)
        output = capture_console()
        assert "stride" in output
        assert "random" in output

    def test_with_explicit_metric_keys(self, capture_console):
        """Comparison with explicit metric keys."""
        results = {
            'a': {'training': {}, 'final_eval': {'acc': 90.0}},
        }
        display_comparison_table(results, metric_keys=['acc'])
        output = capture_console()
        assert "acc" in output
        assert "90" in output


class TestDisplayPostLiveSummary:

    def test_empty(self, capture_console):
        """No-op for empty results."""
        display_post_live_summary({})
        output = capture_console()
        assert output.strip() == ""

    def test_single_strategy(self, capture_console):
        """Post-live summary with one strategy."""
        results = {
            'stride': {
                'init_eval': {'loss': 5.0},
                'final_eval': {'loss': 0.5},
                'training': {'actual_steps': 1000},
            }
        }
        display_post_live_summary(results)
        output = capture_console()
        assert "stride" in output
        assert "0.5" in output

    def test_multiple_strategies(self, capture_console):
        """Post-live summary triggers comparison table for multiple strategies."""
        results = {
            'stride': {
                'init_eval': {'loss': 5.0},
                'final_eval': {'loss': 0.5},
                'training': {'actual_steps': 1000},
            },
            'random': {
                'init_eval': None,
                'final_eval': {'loss': 1.0},
                'training': {},
            },
        }
        display_post_live_summary(results)
        output = capture_console()
        assert "stride" in output
        assert "random" in output
        assert "STRATEGY COMPARISON" in output

    def test_accuracy_in_summary(self, capture_console):
        """Post-live summary with accuracy metrics."""
        results = {
            'stride': {
                'init_eval': {'training_accuracy': 10.0},
                'final_eval': {'training_accuracy': 95.0},
                'training': {'actual_epochs': 500},
            },
        }
        display_post_live_summary(results)
        output = capture_console()
        assert "stride" in output
        assert "95" in output


class TestDisplayPhaseTransition:

    def test_minimal(self, capture_console):
        """Phase transition without metrics."""
        display_phase_transition("phase1", "phase2")
        output = capture_console()
        assert "Phase Transition" in output
        assert "phase1" in output
        assert "phase2" in output

    def test_with_metrics(self, capture_console):
        """Phase transition with float metrics."""
        display_phase_transition("phase1", "phase2", metrics={"avg_prob": 0.5})
        output = capture_console()
        assert "phase1" in output
        assert "phase2" in output
        assert "0.5" in output


class TestDisplayGrokkingAchieved:

    def test_basic(self, capture_console):
        """Grokking achieved display."""
        display_grokking_achieved(500)
        output = capture_console()
        assert "Grokking" in output
        assert "500" in output
