"""Analysis tool registry and base classes.

Provides the ToolRegistry for auto-discovery of analysis tools,
the AnalysisTool ABC that all tools implement, and the AnalysisContext
dataclass passed to every tool's run() method.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


def _metric_slug(name: str) -> str:
    """Convert a metric column name to a filename-safe token.

    Strips the hook prefix and replaces slashes with underscores.
    'training_metrics/loss' -> 'loss'
    'norms/norm_transformer.h.0' -> 'norm_transformer.h.0'
    """
    return name.split('/')[-1] if '/' in name else name


@dataclass
class AnalysisContext:
    """Context passed to every analysis tool.

    Attributes:
        experiment_name: Name of the experiment being analyzed.
        data: Merged DataFrame with step, strategy, and metric columns.
        strategies: Strategy names present in the data.
        output_dir: Tool-specific output directory (output/{experiment}/analysis/{tool}/).
        args: Full argparse namespace with CLI args.
        experiment_config: Loaded experiment_config.json dict, or None.
        resolver: MetricResolver for looking up human-readable labels.
    """
    experiment_name: str
    data: pd.DataFrame
    strategies: list[str]
    output_dir: Path
    args: Any
    experiment_config: dict | None = None
    resolver: Any = None

    def output_path(self, view: str, metrics: list[str] | None = None,
                    ext: str | None = None) -> Path:
        """Build a consistent output file path.

        Pattern: ``{output_dir}/{view}_{metric_slug}.{ext}``

        If no metrics are given, just ``{view}.{ext}``.
        Truncates to 3 metric tokens; appends ``+N`` for the rest.

        Args:
            view: Descriptive name for this output (e.g. 'overlay',
                'heatmap', 'threshold', 'table').
            metrics: Metric column names to encode in the filename.
            ext: File extension. Defaults to ``args.format`` or 'png'.

        Returns:
            Full Path with parent directories created.
        """
        if ext is None:
            ext = getattr(self.args, 'format', 'png') or 'png'

        if metrics:
            max_parts = 3
            tokens = [_metric_slug(m) for m in metrics[:max_parts]]
            slug = '_'.join(tokens)
            if len(metrics) > max_parts:
                slug += f'_+{len(metrics) - max_parts}'
            name = f'{view}_{slug}.{ext}'
        else:
            name = f'{view}.{ext}'

        path = self.output_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


class AnalysisTool(ABC):
    """Base class for analysis tools.

    Subclasses must set `name` and `description`, and implement `run()`.
    Optionally override `add_args()` for tool-specific CLI arguments
    and `describe_outputs()` to document what files the tool produces.
    """

    name: str = "base_tool"
    description: str = ""

    @classmethod
    def add_args(cls, parser):
        """Add tool-specific CLI arguments to the parser.

        Override in subclasses to register additional argparse arguments.
        """
        pass

    @abstractmethod
    def run(self, context: AnalysisContext) -> None:
        """Execute the analysis tool.

        Args:
            context: AnalysisContext with experiment data and output paths.
        """
        ...

    def describe_outputs(self) -> list[str]:
        """Return descriptions of files this tool produces.

        Override in subclasses to document output files.
        """
        return []


class ToolRegistry:
    """Registry of available analysis tools.

    Tools register via the @ToolRegistry.register decorator. The registry
    stores classes (not instances) and instantiates them on demand.
    """

    _tools: dict[str, type[AnalysisTool]] = {}

    @classmethod
    def register(cls, tool_cls: type[AnalysisTool]) -> type[AnalysisTool]:
        """Decorator to register a tool class."""
        instance = tool_cls()
        cls._tools[instance.name] = tool_cls
        return tool_cls

    @classmethod
    def get(cls, name: str) -> type[AnalysisTool]:
        """Get a tool class by name."""
        if name not in cls._tools:
            available = ', '.join(sorted(cls._tools.keys()))
            raise ValueError(
                f"Unknown analysis tool: '{name}'. Available tools: {available}"
            )
        return cls._tools[name]

    @classmethod
    def list_all(cls) -> list[str]:
        """List all registered tool names."""
        return sorted(cls._tools.keys())

    @classmethod
    def get_all_info(cls) -> list[dict]:
        """Get metadata for all registered tools."""
        info = []
        for name, tool_cls in cls._tools.items():
            instance = tool_cls()
            info.append({
                'name': instance.name,
                'description': instance.description,
                'outputs': instance.describe_outputs(),
            })
        return info
