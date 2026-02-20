"""Experiment registry for discovering experiments by name.

Used by the unified entry point (run_experiment.py) to look up
experiment runners by name.
"""


class ExperimentRegistry:
    """Registry for experiment runners.

    Experiments register themselves with a name, and can be looked up
    by name from the unified entry point.
    """

    _registry: dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register an experiment runner class.

        Usage:
            @ExperimentRegistry.register("my_experiment")
            class MyRunner(ExperimentRunner):
                ...
        """
        def decorator(runner_cls):
            cls._registry[name] = runner_cls
            return runner_cls
        return decorator

    @classmethod
    def get(cls, name: str):
        """Get a registered experiment runner class by name."""
        if name not in cls._registry:
            available = ', '.join(sorted(cls._registry.keys()))
            raise KeyError(f"Unknown experiment '{name}'. Available: {available}")
        return cls._registry[name]

    @classmethod
    def list_all(cls) -> list[str]:
        """List all registered experiment names."""
        return sorted(cls._registry.keys())

    @classmethod
    def get_all(cls) -> dict[str, type]:
        """Get all registered experiment runner classes."""
        return dict(cls._registry)
