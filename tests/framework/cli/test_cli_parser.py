"""Tests for framework/cli/cli_parser.py — OLArgumentParser."""

import argparse

import pytest

from framework.cli.cli_parser import OLArgumentParser


class TestOLArgumentParserInit:

    def test_creates_parser(self):
        """Constructor produces a usable parser."""
        parser = OLArgumentParser(experiment_name="test_exp")
        assert isinstance(parser, argparse.ArgumentParser)

    def test_stores_experiment_name(self):
        parser = OLArgumentParser(experiment_name="my_exp")
        assert parser.experiment_name == "my_exp"

    def test_default_experiment_name_is_none(self):
        parser = OLArgumentParser()
        assert parser.experiment_name is None

    def test_has_help_action(self):
        """Parser registers --help / -h."""
        parser = OLArgumentParser()
        option_strings = [
            s for action in parser._actions for s in action.option_strings
        ]
        assert '-h' in option_strings
        assert '--help' in option_strings


class TestOLArgumentParserError:

    def test_error_raises_system_exit(self):
        """error() raises SystemExit(1)."""
        parser = OLArgumentParser()
        with pytest.raises(SystemExit) as exc_info:
            parser.error("something went wrong")
        assert exc_info.value.code == 1

    def test_exit_with_zero_status(self):
        """exit(0) raises SystemExit(0)."""
        parser = OLArgumentParser()
        with pytest.raises(SystemExit) as exc_info:
            parser.exit(0)
        assert exc_info.value.code == 0

    def test_exit_with_nonzero_status(self):
        """exit(1) raises SystemExit(1)."""
        parser = OLArgumentParser()
        with pytest.raises(SystemExit) as exc_info:
            parser.exit(1, "failure")
        assert exc_info.value.code == 1


class TestOLArgumentParserHelp:

    def test_print_help_raises_no_error(self):
        """print_help() runs without error (output suppressed by NULL console)."""
        parser = OLArgumentParser(experiment_name="test")
        parser.add_argument('--foo', help="A foo argument")
        # print_help doesn't raise (console is NULL in tests)
        parser.print_help()

    def test_print_usage_is_noop(self):
        """print_usage() is a no-op."""
        parser = OLArgumentParser()
        # Should not raise
        parser.print_usage()


class TestOLArgumentParserParsing:

    def test_parses_custom_arguments(self):
        """Parser correctly parses added arguments."""
        parser = OLArgumentParser()
        parser.add_argument('--epochs', type=int, default=100)
        parser.add_argument('--lr', type=float, default=0.01)
        args = parser.parse_args(['--epochs', '50', '--lr', '0.001'])
        assert args.epochs == 50
        assert args.lr == 0.001

    def test_invalid_arg_raises_system_exit(self):
        """Unrecognized arguments cause SystemExit via error()."""
        parser = OLArgumentParser()
        with pytest.raises(SystemExit):
            parser.parse_args(['--nonexistent'])

    def test_help_flag_raises_system_exit_zero(self):
        """--help causes SystemExit(0)."""
        parser = OLArgumentParser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(['--help'])
        assert exc_info.value.code == 0


class TestFormatActionFlags:

    def test_boolean_flag(self):
        """Store-true actions show just the flag, no metavar."""
        parser = OLArgumentParser()
        action = parser.add_argument('--verbose', action='store_true', help="Verbose")
        flags = OLArgumentParser._format_action_flags(action)
        assert '--verbose' in flags
        # Should NOT have a metavar like VERBOSE
        assert 'VERBOSE' not in flags

    def test_flag_with_choices(self):
        """Actions with choices show {choice1,choice2}."""
        parser = OLArgumentParser()
        action = parser.add_argument('--mode', choices=['train', 'eval'])
        flags = OLArgumentParser._format_action_flags(action)
        assert '{train,eval}' in flags

    def test_flag_with_type(self):
        """Actions with type show the type name as metavar."""
        parser = OLArgumentParser()
        action = parser.add_argument('--count', type=int)
        flags = OLArgumentParser._format_action_flags(action)
        assert 'INT' in flags or 'int' in flags

    def test_positional_arg(self):
        """Positional arguments show their dest as the flag."""
        parser = OLArgumentParser()
        action = parser.add_argument('experiment', nargs='?')
        flags = OLArgumentParser._format_action_flags(action)
        assert 'experiment' in flags

    def test_nargs_star(self):
        """nargs='*' appends ... to the metavar."""
        parser = OLArgumentParser()
        action = parser.add_argument('--items', nargs='*')
        flags = OLArgumentParser._format_action_flags(action)
        assert '...' in flags

    def test_explicit_metavar(self):
        """Explicit metavar is used when provided."""
        parser = OLArgumentParser()
        action = parser.add_argument('--output', metavar='PATH')
        flags = OLArgumentParser._format_action_flags(action)
        assert 'PATH' in flags
