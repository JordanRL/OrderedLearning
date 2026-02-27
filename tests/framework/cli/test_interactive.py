"""Tests for framework/cli/interactive.py — helper functions."""

import argparse
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from framework.cli.interactive import (
    _prompt_for_action,
    _prompt_advanced,
    _print_summary,
    ADVANCED_DESTS,
)


# ==================================================================
# _prompt_for_action
# ==================================================================

class TestPromptForAction:

    def test_choice_action(self):
        """Choice-based action prompts with choices and returns selection."""
        action = argparse.Action(
            option_strings=['--mode'],
            dest='mode',
            choices=['train', 'eval'],
            help='Mode selection',
        )
        with patch('framework.cli.interactive.OLConsole') as MockConsole:
            mock = MockConsole.return_value
            mock.prompt.return_value = 'eval'
            result = _prompt_for_action(action, 'train')
        assert result == 'eval'

    def test_boolean_store_true_action(self):
        """store_true action prompts with confirm."""
        parser = argparse.ArgumentParser()
        action = parser.add_argument('--verbose', action='store_true')
        with patch('framework.cli.interactive.OLConsole') as MockConsole:
            mock = MockConsole.return_value
            mock.confirm.return_value = True
            result = _prompt_for_action(action, False)
        assert result is True

    def test_integer_action(self):
        """Integer-typed action prompts and converts result."""
        parser = argparse.ArgumentParser()
        action = parser.add_argument('--epochs', type=int, default=100)
        with patch('framework.cli.interactive.OLConsole') as MockConsole:
            mock = MockConsole.return_value
            mock.prompt.return_value = '200'
            result = _prompt_for_action(action, 100)
        assert result == 200

    def test_integer_action_invalid_input_returns_current(self):
        """Non-numeric input for integer action returns current value."""
        parser = argparse.ArgumentParser()
        action = parser.add_argument('--epochs', type=int, default=100)
        with patch('framework.cli.interactive.OLConsole') as MockConsole:
            mock = MockConsole.return_value
            mock.prompt.return_value = 'not_a_number'
            result = _prompt_for_action(action, 100)
        assert result == 100

    def test_string_action_fallback(self):
        """String action uses prompt fallback."""
        parser = argparse.ArgumentParser()
        action = parser.add_argument('--name', type=str, default='default')
        with patch('framework.cli.interactive.OLConsole') as MockConsole:
            mock = MockConsole.return_value
            mock.prompt.return_value = 'custom_name'
            result = _prompt_for_action(action, 'default')
        assert result == 'custom_name'

    def test_boolean_action_no_help(self):
        """Boolean action without help text formats label correctly."""
        parser = argparse.ArgumentParser()
        action = parser.add_argument('--my-flag', action='store_true', help='')
        with patch('framework.cli.interactive.OLConsole') as MockConsole:
            mock = MockConsole.return_value
            mock.confirm.return_value = False
            result = _prompt_for_action(action, False)
        assert result is False

    def test_integer_action_with_int_current_value(self):
        """Integer current_value triggers int path even without type=int."""
        parser = argparse.ArgumentParser()
        action = parser.add_argument('--count', default=5)
        with patch('framework.cli.interactive.OLConsole') as MockConsole:
            mock = MockConsole.return_value
            mock.prompt.return_value = '10'
            result = _prompt_for_action(action, 5)  # current_value is int
        assert result == 10


# ==================================================================
# _prompt_advanced
# ==================================================================

class TestPromptAdvanced:

    def test_prompts_unskipped_dests(self):
        """Prompts for ADVANCED_DESTS not in skip set."""
        parser = argparse.ArgumentParser()
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--batch-size', type=int, default=32)
        parser.add_argument('--output-dir', type=str, default='output')
        actions_by_dest = {a.dest: a for a in parser._actions if a.dest != 'help'}

        args_dict = {'seed': 42, 'batch_size': 32, 'output_dir': 'output'}

        with patch('framework.cli.interactive._prompt_for_action') as mock_prompt:
            mock_prompt.side_effect = lambda action, val: val  # return current value
            _prompt_advanced(args_dict, actions_by_dest, skip={'batch_size'})

        # batch_size was skipped, seed should have been prompted
        prompted_dests = [call.args[0].dest for call in mock_prompt.call_args_list]
        assert 'batch_size' not in prompted_dests
        assert 'seed' in prompted_dests

    def test_skips_missing_dests(self):
        """Dests not in actions_by_dest are silently skipped."""
        parser = argparse.ArgumentParser()
        parser.add_argument('--seed', type=int, default=42)
        actions_by_dest = {a.dest: a for a in parser._actions if a.dest != 'help'}
        args_dict = {'seed': 42}

        with patch('framework.cli.interactive._prompt_for_action') as mock_prompt:
            mock_prompt.side_effect = lambda action, val: val
            # No crash even though batch_size etc. aren't in actions
            _prompt_advanced(args_dict, actions_by_dest, skip=set())


# ==================================================================
# _print_summary
# ==================================================================

class TestPrintSummary:

    def test_runs_without_error(self):
        """_print_summary executes without crashing (console is NULL)."""
        runner_cls = MagicMock()
        runner_cls.interactive_args = ['strategy', 'epochs']

        args_dict = {
            'strategy': 'stride',
            'epochs': 1000,
            'live': True,
            'seed': 42,
            'with_hooks': None,
            'batch_size': 32,
            'output_dir': 'output',
        }

        # Should not raise
        _print_summary('mod_arithmetic', runner_cls, args_dict)

    def test_skips_none_and_false_values(self):
        """Settings with None/False/none values are skipped."""
        runner_cls = MagicMock()
        runner_cls.interactive_args = []

        args_dict = {
            'live': False,
            'with_hooks': None,
            'seed': 42,
            'batch_size': None,
            'output_dir': 'none',
        }

        # Should not raise — just exercises the filter logic
        _print_summary('test', runner_cls, args_dict)


# ==================================================================
# ADVANCED_DESTS constant
# ==================================================================

class TestAdvancedDests:

    def test_contains_expected_keys(self):
        assert 'seed' in ADVANCED_DESTS
        assert 'batch_size' in ADVANCED_DESTS
        assert 'output_dir' in ADVANCED_DESTS
