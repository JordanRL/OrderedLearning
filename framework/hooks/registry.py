"""Hook registry for auto-discovery and instantiation."""

from ..registry import Registry
from .training_hook import TrainingHook
from .intervention_hook import InterventionHook


class HookRegistry(Registry):
    """Registry of available training hooks.

    Hooks register via the @HookRegistry.register decorator. The registry
    stores classes (not instances) since hooks have mutable state and each
    training run should get fresh instances.
    """

    _items = {}
    _registry_label = "hook"

    @classmethod
    def list_all(cls) -> list[str]:
        """List all non-debug registered hook names."""
        return [
            name for name, hook_cls in cls._items.items()
            if not getattr(hook_cls, 'debug', False)
        ]

    @classmethod
    def list_observers(cls) -> list[str]:
        """List observer (non-intervention, non-debug) hook names."""
        return [
            name for name, hook_cls in cls._items.items()
            if not issubclass(hook_cls, InterventionHook)
            and not getattr(hook_cls, 'debug', False)
        ]

    @classmethod
    def list_interventions(cls) -> list[str]:
        """List intervention (non-debug) hook names."""
        return [
            name for name, hook_cls in cls._items.items()
            if issubclass(hook_cls, InterventionHook)
            and not getattr(hook_cls, 'debug', False)
        ]

    @classmethod
    def list_debug(cls) -> list[str]:
        """List debug hook names."""
        return [
            name for name, hook_cls in cls._items.items()
            if getattr(hook_cls, 'debug', False)
        ]

    @classmethod
    def get_all_info(cls) -> list[dict]:
        """Get metadata for all registered hooks."""
        info = []
        for name, hook_cls in cls._items.items():
            instance = hook_cls()
            info.append({
                'name': instance.name,
                'description': instance.description,
                'hook_points': {hp.name for hp in instance.hook_points},
                'is_intervention': isinstance(instance, InterventionHook),
                'is_debug': instance.debug,
                'needs': instance.needs,
            })
        return info
