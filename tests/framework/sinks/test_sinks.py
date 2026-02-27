"""Tests for framework/sinks — formatting helpers, path resolution, CSV/JSONL emit."""

import json
import pytest

from framework.sinks.base import (
    _flatten_for_csv, _format_metric_value, _format_number,
)
from framework.sinks.csv_sink import CSVSink
from framework.sinks.jsonl import JSONLSink
from framework.hooks.hook_point import HookPoint


class TestFlattenForCSV:

    def test_dict_to_semicolons(self):
        """Dicts become 'key:value;key:value'."""
        result = _flatten_for_csv({"a": 1, "b": 2})
        assert "a:1" in result
        assert "b:2" in result
        assert ";" in result

    def test_list_to_semicolons(self):
        """Lists become 'val;val;val'."""
        assert _flatten_for_csv([1, 2, 3]) == "1;2;3"

    def test_scalar_passthrough(self):
        """Scalars pass through unchanged."""
        assert _flatten_for_csv(42) == 42
        assert _flatten_for_csv("hello") == "hello"


class TestFormatNumber:

    def test_small_number_scientific(self):
        """Very small numbers use scientific notation."""
        result = _format_number(0.0001)
        assert "e" in result

    def test_normal_range_fixed(self):
        """Mid-range numbers use fixed-point."""
        result = _format_number(1.234)
        assert "e" not in result
        assert "1.234" in result

    def test_large_number_scientific(self):
        """Very large numbers use scientific notation."""
        result = _format_number(100000.0)
        assert "e" in result


class TestFormatMetricValue:

    def test_list_of_floats_shows_mean(self):
        """List of numbers shows the mean value."""
        result = _format_metric_value([1.0, 2.0, 3.0])
        assert "2.0" in result
        assert "n=3" in result

    def test_compact_suppresses_n(self):
        """compact=True omits (n=X) suffix."""
        result = _format_metric_value([1.0, 2.0], compact=True)
        assert "n=" not in result

    def test_empty_list(self):
        """Empty list shows [0 items]."""
        result = _format_metric_value([])
        assert "0 items" in result

    def test_non_numeric_list(self):
        """List of non-numbers shows [N items]."""
        result = _format_metric_value(["a", "b"])
        assert "2 items" in result

    def test_int_passthrough(self):
        """Integer values formatted as string."""
        result = _format_metric_value(42)
        assert result == "42"

    def test_float_formatted(self):
        """Float values go through _format_number."""
        result = _format_metric_value(1.5)
        assert "1.5" in result


class TestFilePathSinkInit:

    def test_raises_without_filepath_or_dirs(self):
        """FilePathSink raises ValueError when neither filepath nor (output_dir + experiment_name)."""
        with pytest.raises(ValueError, match="requires either"):
            CSVSink()

    def test_fixed_path_mode(self, tmp_path):
        """filepath= puts sink in fixed-path mode (no auto resolution)."""
        sink = CSVSink(filepath=str(tmp_path / "fixed.csv"))
        assert sink._auto_mode is False

    def test_auto_path_mode(self, tmp_path):
        """output_dir + experiment_name puts sink in auto mode."""
        sink = CSVSink(output_dir=str(tmp_path), experiment_name="exp")
        assert sink._auto_mode is True


class TestFilePathSinkSetRunContext:

    def test_resolves_path_on_context(self, tmp_path):
        """set_run_context(strategy=...) resolves the auto path."""
        sink = CSVSink(output_dir=str(tmp_path), experiment_name="exp")
        sink.set_run_context(strategy="my_strat")
        expected = tmp_path / "exp" / "my_strat" / "my_strat.csv"
        assert sink._filepath == expected


class TestFilePathSinkResolve:

    def test_resolve_path_no_collision(self, tmp_path):
        """First resolve returns strategy.ext without suffix."""
        sink = CSVSink(output_dir=str(tmp_path), experiment_name="test_exp")
        resolved = sink._resolve_path("my_strat")
        expected = tmp_path / "test_exp" / "my_strat" / "my_strat.csv"
        assert resolved == expected

    def test_resolve_path_collision_avoidance(self, tmp_path):
        """When file exists, appends _1, _2, etc."""
        strat_dir = tmp_path / "test_exp" / "my_strat"
        strat_dir.mkdir(parents=True)
        (strat_dir / "my_strat.csv").touch()

        sink = CSVSink(output_dir=str(tmp_path), experiment_name="test_exp")
        resolved = sink._resolve_path("my_strat")
        assert resolved == strat_dir / "my_strat_1.csv"


class TestCSVSinkEmit:

    def test_emit_creates_file_with_header(self, tmp_path):
        """Emitting metrics creates CSV with header and data row."""
        filepath = tmp_path / "test.csv"
        sink = CSVSink(filepath=str(filepath))
        sink.emit({"loss": 0.5, "acc": 0.9}, epoch=1, hook_point=HookPoint.POST_EPOCH)
        sink.flush()

        content = filepath.read_text()
        lines = content.strip().split('\n')
        assert len(lines) == 2  # header + 1 data row
        assert "loss" in lines[0]
        assert "acc" in lines[0]


    def test_emit_expands_columns(self, tmp_path):
        """New columns in later emits trigger header rewrite."""
        filepath = tmp_path / "test_expand.csv"
        sink = CSVSink(filepath=str(filepath))
        sink.emit({"loss": 0.5, "acc": 0.9}, epoch=1, hook_point=HookPoint.POST_EPOCH)
        sink.emit({"loss": 0.3, "acc": 0.95, "lr": 0.001}, epoch=2, hook_point=HookPoint.POST_EPOCH)
        sink.flush()

        content = filepath.read_text()
        lines = content.strip().split('\n')
        assert len(lines) == 3  # header + 2 data rows
        header = lines[0]
        assert "loss" in header
        assert "acc" in header
        assert "lr" in header


class TestJSONLSinkEmit:

    def test_emit_creates_jsonl_lines(self, tmp_path):
        """Emitting metrics writes valid JSON Lines."""
        filepath = tmp_path / "test.jsonl"
        sink = JSONLSink(filepath=str(filepath))
        sink.emit({"loss": 0.5}, epoch=1, hook_point=HookPoint.POST_EPOCH)
        sink.emit({"loss": 0.3}, epoch=2, hook_point=HookPoint.POST_EPOCH)
        sink.flush()

        lines = filepath.read_text().strip().split('\n')
        assert len(lines) == 2
        record = json.loads(lines[0])
        assert record["epoch"] == 1
        assert record["loss"] == 0.5
