"""Metric metadata resolution for analysis tools.

Introspects the hook registry to resolve raw metric column names
(e.g., 'training_metrics/loss') to human-readable labels, descriptions,
formulas, and sign conventions from MetricInfo.
"""

from __future__ import annotations

from framework.hooks import MetricInfo


class MetricResolver:
    """Resolve raw metric column names to human-readable metadata.

    Built lazily from the hook registry on first use. Handles both
    exact metric names (e.g., 'training_metrics/loss') and templated
    per-parameter names (e.g., 'norms/norm_transformer.h.0.attn.weight'
    matching the template 'norms/norm_{layer}').

    Usage::

        resolver = MetricResolver()
        resolver.label('training_metrics/loss')       # 'Loss'
        resolver.label('adam_dynamics/amplification_ratio')  # 'Amplification Ratio'
        resolver.label('unknown/metric')               # 'Metric' (fallback)
    """

    def __init__(self):
        self._exact: dict[str, MetricInfo] = {}
        self._prefixes: list[tuple[str, MetricInfo]] = []
        self._built = False

    def _ensure_built(self):
        """Build lookup tables from hook registry on first access."""
        if self._built:
            return
        self._built = True

        from framework.hooks import HookRegistry

        for name in HookRegistry.list_all():
            hook_cls = HookRegistry.get(name)
            hook = hook_cls()
            for mi in hook.describe_metrics():
                full_key = f"{hook.name}/{mi.name}"
                if '{' in full_key:
                    # Template pattern — extract prefix before the variable
                    prefix = full_key.split('{')[0]
                    self._prefixes.append((prefix, mi))
                else:
                    self._exact[full_key] = mi

        # Sort prefixes longest-first for greedy matching
        self._prefixes.sort(key=lambda x: len(x[0]), reverse=True)

    def resolve(self, column_name: str) -> MetricInfo | None:
        """Look up MetricInfo for a column name.

        Tries exact match first, then prefix matching for templated
        per-parameter metrics.
        """
        self._ensure_built()

        if column_name in self._exact:
            return self._exact[column_name]

        for prefix, mi in self._prefixes:
            if column_name.startswith(prefix):
                return mi

        return None

    def label(self, column_name: str) -> str:
        """Short display label suitable for plot axes and legends.

        Returns the MetricInfo.label if available, otherwise generates
        a fallback from the column name by stripping the hook prefix
        and title-casing.
        """
        mi = self.resolve(column_name)
        if mi and mi.label:
            return mi.label
        return _fallback_label(column_name)

    def description(self, column_name: str) -> str:
        """Long description suitable for tooltips or documentation."""
        mi = self.resolve(column_name)
        if mi:
            return mi.description
        return self.label(column_name)

    def formula(self, column_name: str) -> str:
        """Mathematical formula, or empty string."""
        mi = self.resolve(column_name)
        return mi.formula if mi else ''

    def sign_info(self, column_name: str) -> str:
        """Sign convention / range info, or empty string."""
        mi = self.resolve(column_name)
        return mi.sign_info if mi else ''


def _fallback_label(column_name: str) -> str:
    """Generate a display label from a raw column name.

    Strips the hook prefix (first slash segment), replaces underscores
    with spaces, and title-cases the result.
    """
    parts = column_name.split('/')
    # Use the last segment as the base label
    base = parts[-1] if len(parts) > 1 else parts[0]
    return base.replace('_', ' ').title()
