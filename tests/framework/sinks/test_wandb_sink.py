"""Tests for framework/sinks/wandb.py — WandbSink."""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest
import torch

from framework.hooks.hook_point import HookPoint


@pytest.fixture
def mock_wandb():
    """Create a mock wandb module and inject it."""
    wandb = MagicMock()
    wandb.run = None
    wandb.Settings.return_value = MagicMock()
    wandb.Histogram = MagicMock()
    # Temporarily inject into sys.modules so import works
    old = sys.modules.get('wandb')
    sys.modules['wandb'] = wandb
    yield wandb
    if old is not None:
        sys.modules['wandb'] = old
    else:
        del sys.modules['wandb']


@pytest.fixture
def wandb_sink(mock_wandb):
    """WandbSink with mocked wandb."""
    from framework.sinks.wandb import WandbSink
    return WandbSink(project="test_project", group="test_group")


class TestWandbSinkInit:

    def test_stores_project(self, wandb_sink):
        assert wandb_sink._project == "test_project"

    def test_stores_group(self, wandb_sink):
        assert wandb_sink._group == "test_group"

    def test_auto_group_when_none(self, mock_wandb):
        from framework.sinks.wandb import WandbSink
        sink = WandbSink(project="test")
        assert "experiment_" in sink._group



class TestWandbSinkSetRunContext:

    def test_calls_init(self, wandb_sink, mock_wandb):
        """set_run_context calls wandb.init with correct params."""
        wandb_sink.set_run_context(strategy="stride")
        mock_wandb.init.assert_called_once()
        call_kwargs = mock_wandb.init.call_args[1]
        assert call_kwargs['project'] == "test_project"
        assert call_kwargs['name'] == "stride"

    def test_finishes_previous_run(self, wandb_sink, mock_wandb):
        """Finishes previous run before starting new one."""
        mock_wandb.run = MagicMock()  # Simulate active run
        wandb_sink.set_run_context(strategy="new")
        mock_wandb.finish.assert_called_once()


class TestWandbSinkEmit:

    def test_empty_metrics_no_op(self, wandb_sink, mock_wandb):
        wandb_sink.emit({}, 0, HookPoint.SNAPSHOT)
        mock_wandb.log.assert_not_called()

    def test_no_run_no_op(self, wandb_sink, mock_wandb):
        """No logging when no active run."""
        mock_wandb.run = None
        wandb_sink.emit({"metric": 1.0}, 0, HookPoint.SNAPSHOT)
        mock_wandb.log.assert_not_called()

    def test_scalar_metrics(self, wandb_sink, mock_wandb):
        """Scalar values are logged directly with correct value."""
        mock_wandb.run = MagicMock()
        wandb_sink.emit({"hook/metric": 1.5}, 10, HookPoint.SNAPSHOT)
        mock_wandb.log.assert_called_once()
        logged = mock_wandb.log.call_args[0][0]
        assert "hook/metric" in logged
        assert logged["hook/metric"] == 1.5

    def test_dict_metrics(self, wandb_sink, mock_wandb):
        """Dict values are flattened with sub-keys."""
        mock_wandb.run = MagicMock()
        wandb_sink.emit({"hook": {"sub_metric": 1.0}}, 10, HookPoint.SNAPSHOT)
        logged = mock_wandb.log.call_args[0][0]
        assert "hook/sub_metric" in logged

    def test_list_metrics_as_histogram(self, wandb_sink, mock_wandb):
        """List values are logged as histograms + mean."""
        mock_wandb.run = MagicMock()
        wandb_sink.emit({"metric": [1.0, 2.0, 3.0]}, 10, HookPoint.SNAPSHOT)
        logged = mock_wandb.log.call_args[0][0]
        assert "metric_mean" in logged

    def test_tensor_values(self, wandb_sink, mock_wandb):
        """Tensor values use .item() for scalar conversion."""
        mock_wandb.run = MagicMock()
        wandb_sink.emit({"metric": torch.tensor(1.5)}, 10, HookPoint.SNAPSHOT)
        mock_wandb.log.assert_called_once()
        logged = mock_wandb.log.call_args[0][0]
        assert logged["metric"] == pytest.approx(1.5)


class TestWandbSinkToScalar:

    def test_tensor(self, wandb_sink):
        assert wandb_sink._to_scalar(torch.tensor(3.14)) == pytest.approx(3.14, abs=0.01)

    def test_int(self, wandb_sink):
        assert wandb_sink._to_scalar(5) == 5

    def test_float(self, wandb_sink):
        assert wandb_sink._to_scalar(3.14) == 3.14


class TestWandbSinkFlush:

    def test_flush_with_active_run(self, wandb_sink, mock_wandb):
        mock_wandb.run = MagicMock()
        wandb_sink.flush()
        mock_wandb.finish.assert_called_once()

    def test_flush_without_run(self, wandb_sink, mock_wandb):
        mock_wandb.run = None
        wandb_sink.flush()
        mock_wandb.finish.assert_not_called()
