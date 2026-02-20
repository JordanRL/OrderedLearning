"""Shared reference weights loader for hooks that compare against a known solution."""

import warnings

import torch


DEFAULT_REFERENCE_PATH = "weights/{strategy}_final.pt"


class _SafeFormatDict(dict):
    """Dict that returns '{key}' for missing keys, so unresolved
    template variables are preserved rather than raising KeyError."""

    def __missing__(self, key):
        return '{' + key + '}'


class ReferenceWeights:
    """Lazy-loading, context-aware reference weight container.

    Shared by any hooks that need to compare current model state against
    a known solution. Supports template variables in the path (e.g.,
    {strategy}, {curriculum_type}) resolved via set_run_context().

    Weights are lazy-loaded on first access after context resolution,
    and reloaded when the resolved path changes.
    """

    def __init__(self, path_template: str = DEFAULT_REFERENCE_PATH):
        self._path_template = path_template
        self._path = path_template
        self._weights: dict[str, torch.Tensor] | None = None
        self._loaded = False

    @property
    def weights(self) -> dict[str, torch.Tensor] | None:
        """The loaded reference weights, or None if not yet loaded."""
        return self._weights

    @property
    def available(self) -> bool:
        """Whether reference weights have been successfully loaded."""
        return self._weights is not None

    def set_run_context(self, **kwargs):
        """Resolve path template with context variables.

        If the resolved path changes, clears cached weights so they
        reload on next ensure_loaded() call.
        """
        resolved = self._path_template.format_map(_SafeFormatDict(kwargs))
        if resolved != self._path:
            self._path = resolved
            self._weights = None
            self._loaded = False

    def reset(self):
        """Reset to initial state (unresolved template, no cached weights)."""
        self._path = self._path_template
        self._weights = None
        self._loaded = False

    def ensure_loaded(self, device):
        """Lazy-load reference weights on first call after context resolution.

        Skips loading if the path still contains unresolved template variables.
        Checks embedded environment metadata against the current environment
        and warns on reproducibility-relevant differences.
        """
        if self._loaded:
            return

        if not self._path or '{' in self._path:
            return

        self._loaded = True

        try:
            data = torch.load(
                self._path, map_location=device, weights_only=False,
            )
            # Extract state_dict and optional environment metadata
            saved_env = None
            if isinstance(data, dict) and 'model_state_dict' in data:
                saved_env = data.get('environment')
                state_dict = data['model_state_dict']
            elif isinstance(data, dict) and 'state_dict' in data:
                saved_env = data.get('environment')
                state_dict = data['state_dict']
            else:
                # Bare state_dict (no envelope) — legacy format
                state_dict = data

            self._weights = {
                k: v.to(device).float() for k, v in state_dict.items()
            }

            # Report successful load
            n_params = sum(v.numel() for v in self._weights.values())
            from console import OLConsole
            OLConsole().print(
                f"[label]Reference weights loaded:[/label] "
                f"[value]{self._path}[/value] "
                f"[detail]({n_params:,} parameters)[/detail]"
            )

            # Check environment compatibility
            if saved_env is not None:
                from framework.utils import check_environment_compatibility
                diffs = check_environment_compatibility(saved_env)
                if diffs:
                    self._warn_environment_mismatch(diffs)

        except Exception as e:
            warnings.warn(
                f"ReferenceWeights: Failed to load from '{self._path}': {e}"
            )

    def _warn_environment_mismatch(self, diffs: list[str]):
        """Emit console warning about environment differences."""
        from console import OLConsole
        console = OLConsole()
        header = (
            f"Reference weights '{self._path}' were saved in a different "
            f"environment. Solution-dependent metrics may not be comparable:"
        )
        console.print(f"[warning]{header}[/warning]")
        for diff in diffs:
            console.print(f"  [warning]  {diff}[/warning]")

    def resolve_key(self, name: str) -> str | None:
        """Find the matching key in reference weights, handling _orig_mod. prefix.

        Returns the matching key or None if no match found.
        """
        if self._weights is None:
            return None
        if name in self._weights:
            return name
        stripped = name.replace('_orig_mod.', '')
        if stripped in self._weights:
            return stripped
        return None
