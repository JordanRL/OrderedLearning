"""JSONL sink for appending hook metrics as JSON Lines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..hooks.hook_point import HookPoint
from ..utils import _json_default
from .base import FilePathSink


class JSONLSink(FilePathSink):
    """Append hook metrics as JSON Lines (one JSON object per line).

    Handles nested/structured values natively -- dicts, lists, and scalars
    are all written as proper JSON types. No column management needed.

    Two modes:
    - Fixed path: ``JSONLSink(filepath='path/to/file.jsonl')``
    - Auto path: ``JSONLSink(output_dir='output', experiment_name='presorted')``
      Defers path resolution to ``set_run_context(strategy=...)``, producing
      ``{output_dir}/{experiment_name}/{strategy}/{strategy}.jsonl`` with
      collision avoidance via numeric suffixes.
    """

    _file_extension = "jsonl"

    def __init__(
        self,
        filepath: str | Path | None = None,
        output_dir: str | None = None,
        experiment_name: str | None = None,
    ):
        """
        Args:
            filepath: Fixed path to the output .jsonl file.
            output_dir: Base output directory for auto-path mode.
            experiment_name: Experiment name for auto-path mode.
        """
        super().__init__(filepath, output_dir, experiment_name)

    def _ensure_open(self):
        if self._filepath is None:
            return
        if self._file is None:
            self._filepath.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(self._filepath, 'a', newline='')

    def emit(self, metrics: dict[str, Any], epoch: int, hook_point: HookPoint):
        if not metrics:
            return

        self._ensure_open()
        if self._file is None:
            return
        record = {"epoch": epoch, "hook_point": hook_point.name, **metrics}
        self._file.write(json.dumps(record, default=_json_default) + '\n')
        self._file.flush()
