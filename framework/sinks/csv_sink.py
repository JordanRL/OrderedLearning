"""CSV sink for appending hook metrics to a CSV file."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from ..hooks.hook_point import HookPoint
from .base import FilePathSink, _flatten_for_csv


class CSVSink(FilePathSink):
    """Append hook metrics to a CSV file.

    Writes incrementally on each emit. If a new column appears (e.g., when
    SNAPSHOT hooks fire for the first time after POST_EPOCH hooks have been
    writing), the file is rewritten with the expanded header.

    Two modes:
    - Fixed path: ``CSVSink(filepath='path/to/file.csv')``
    - Auto path: ``CSVSink(output_dir='output', experiment_name='presorted')``
      Defers path resolution to ``set_run_context(strategy=...)``, producing
      ``{output_dir}/{experiment_name}/{strategy}/{strategy}.csv`` with
      collision avoidance via numeric suffixes.
    """

    _file_extension = "csv"

    def __init__(
        self,
        filepath: str | Path | None = None,
        output_dir: str | None = None,
        experiment_name: str | None = None,
    ):
        """
        Args:
            filepath: Fixed path to the output CSV file.
            output_dir: Base output directory for auto-path mode.
            experiment_name: Experiment name for auto-path mode.
        """
        super().__init__(filepath, output_dir, experiment_name)
        self._fieldnames: list[str] = []
        self._rows: list[dict] = []
        self._writer = None

    def _on_new_context(self, **kwargs):
        """Reset CSV-specific state on context change."""
        self._fieldnames = []
        self._rows = []
        self._writer = None

    def _open(self):
        """Open the CSV file and write the header."""
        self._filepath.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._filepath, 'w', newline='')
        self._writer = csv.DictWriter(
            self._file, fieldnames=self._fieldnames, restval='',
        )
        self._writer.writeheader()

    def _rewrite(self):
        """Rewrite the entire file with the current fieldnames and rows."""
        if self._file is not None:
            self._file.close()
        self._open()
        for row in self._rows:
            self._writer.writerow(self._flatten_row(row))
        self._file.flush()

    @staticmethod
    def _flatten_row(row: dict) -> dict:
        """Flatten non-scalar values so every cell is CSV-safe."""
        return {k: _flatten_for_csv(v) for k, v in row.items()}

    def emit(self, metrics: dict[str, Any], epoch: int, hook_point: HookPoint):
        if not metrics:
            return
        if self._filepath is None:
            return

        row = {"epoch": epoch, "hook_point": hook_point.name, **metrics}
        self._rows.append(row)

        # Check if any new columns appeared
        new_keys = [k for k in row if k not in self._fieldnames]
        if new_keys:
            self._fieldnames.extend(new_keys)
            # Rewrite the whole file with the expanded header
            self._rewrite()
        else:
            # Append incrementally
            if self._writer is None:
                self._open()
            self._writer.writerow(self._flatten_row(row))
            self._file.flush()

    def flush(self):
        """Flush and close, also clearing the CSV writer."""
        super().flush()
        self._writer = None
