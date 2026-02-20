"""Analysis tools for experiment metric visualization and exploration.

Provides a registry-based tool system parallel to training hooks.
Tools are auto-discovered via the tools/ subpackage.
"""

from .base import AnalysisTool, AnalysisContext, ToolRegistry
from . import tools  # triggers @ToolRegistry.register decorators

__all__ = ['AnalysisTool', 'AnalysisContext', 'ToolRegistry']
