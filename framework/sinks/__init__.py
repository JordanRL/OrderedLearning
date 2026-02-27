"""Metric sinks that consume hook output.

Sinks receive scalar metrics from hooks and route them to different
destinations (console, CSV, JSONL, W&B).
"""

from .base import MetricSink, FilePathSink
from .console import ConsoleSink
from .csv_sink import CSVSink
from .jsonl import JSONLSink
from .wandb import WandbSink

__all__ = [
    'MetricSink',
    'FilePathSink',
    'ConsoleSink',
    'CSVSink',
    'JSONLSink',
    'WandbSink',
]
