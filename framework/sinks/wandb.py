"""Weights & Biases sink for logging hook metrics to W&B."""

from __future__ import annotations

from typing import Any

from ..hooks.hook_point import HookPoint
from .base import MetricSink


class WandbSink(MetricSink):
    """Log hook metrics to Weights & Biases.

    Creates one W&B run per strategy within a shared group, so runs from
    the same experiment are grouped together but each strategy is tracked
    independently with its own metric curves.

    Metrics are logged as ``hook_point/hook_name/metric_name`` with
    ``epoch`` as the x-axis step.

    Requires ``wandb`` to be installed.
    """

    def __init__(
        self,
        project: str | None = None,
        group: str | None = None,
        config: dict | None = None,
    ):
        """
        Args:
            project: W&B project name.
            group: Experiment group name. If None, auto-generated from
                   a timestamp so all strategy runs are grouped together.
            config: Experiment config dict logged with every run.
        """
        try:
            import wandb
        except ImportError:
            raise ImportError(
                "WandbSink requires the 'wandb' package. "
                "Install it with: pip install wandb"
            )
        self._wandb = wandb
        self._project = project
        self._config = config or {}
        if group is None:
            import datetime
            group = f"experiment_{datetime.datetime.now():%Y%m%d_%H%M%S}"
        self._group = group

    def set_run_context(self, **kwargs):
        # Finish previous run if any
        if self._wandb.run is not None:
            self._wandb.finish()

        run_config = {**self._config, **kwargs}
        # Use the strategy (or other context) as the run name
        run_name = kwargs.get('strategy', None)

        self._wandb.init(
            project=self._project,
            group=self._group,
            name=run_name,
            config=run_config,
            reinit=True,
            settings=self._wandb.Settings(console="off"),
        )

    def emit(self, metrics: dict[str, Any], epoch: int, hook_point: HookPoint):
        if not metrics:
            return
        if self._wandb.run is None:
            return

        logged = {}
        for key, value in metrics.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float, bool)) or hasattr(sub_value, 'item'):
                        logged[f'{key}/{sub_key}'] = self._to_scalar(sub_value)
            elif isinstance(value, (list, tuple)):
                # Accumulated step metrics -- log as histogram + scalar summary
                scalars = [self._to_scalar(v) for v in value]
                if scalars and isinstance(scalars[0], (int, float)):
                    logged[key] = self._wandb.Histogram(scalars)
                    logged[f"{key}_mean"] = sum(scalars) / len(scalars)
                else:
                    logged[key] = scalars
            elif isinstance(value, (int, float, bool)) or hasattr(value, 'item'):
                logged[key] = self._to_scalar(value)
        if logged:
            self._wandb.log(logged, step=epoch)

    @staticmethod
    def _to_scalar(value: Any):
        """Coerce to a Python scalar for W&B logging."""
        if hasattr(value, 'item'):
            return value.item()
        return value

    def flush(self):
        if self._wandb.run is not None:
            self._wandb.finish()
