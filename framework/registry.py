"""Registry base class and experiment registry.

Provides a generic Registry pattern that ExperimentRegistry, HookRegistry,
and ToolRegistry all extend. Each registry provides: register() decorator,
get(name) lookup, and list_all() enumeration.

Used by the unified entry point (run_experiment.py) to look up
experiment runners by name.
"""


class Registry:
    """Generic registry base class.

    Subclasses MUST define their own ``_items = {}`` to avoid sharing
    state across registries, and should set ``_registry_label`` for
    descriptive error messages.

    The ``register()`` decorator supports two calling conventions:

    1. ``@MyRegistry.register("name")`` -- name passed as argument.
    2. ``@MyRegistry.register`` -- name read from ``.name`` attribute
       on a temporary instance of the decorated class.
    """

    _items: dict[str, type] = {}
    _registry_label: str = "item"

    @classmethod
    def register(cls, item_or_name=None):
        """Decorator to register a class.

        Usage:
            @MyRegistry.register("my_name")
            class Foo: ...

            @MyRegistry.register
            class Bar:
                name = "bar"
        """
        # Case 1: @Registry.register("name") -- called with a string arg
        if isinstance(item_or_name, str):
            name = item_or_name
            def decorator(registered_cls):
                cls._items[name] = registered_cls
                return registered_cls
            return decorator

        # Case 2: @Registry.register -- applied directly to a class
        if item_or_name is not None and isinstance(item_or_name, type):
            registered_cls = item_or_name
            instance = registered_cls()
            cls._items[instance.name] = registered_cls
            return registered_cls

        # Case 3: @Registry.register() -- called with no args (parens but empty)
        if item_or_name is None:
            def decorator(registered_cls):
                instance = registered_cls()
                cls._items[instance.name] = registered_cls
                return registered_cls
            return decorator

        raise TypeError(
            f"{cls.__name__}.register() expects a string name, "
            f"a class, or no arguments. Got: {type(item_or_name)}"
        )

    @classmethod
    def get(cls, name: str):
        """Get a registered class by name."""
        if name not in cls._items:
            available = ', '.join(sorted(cls._items.keys()))
            raise ValueError(
                f"Unknown {cls._registry_label}: '{name}'. "
                f"Available: {available}"
            )
        return cls._items[name]

    @classmethod
    def list_all(cls) -> list[str]:
        """List all registered names (sorted)."""
        return sorted(cls._items.keys())


class ExperimentRegistry(Registry):
    """Registry for experiment runners.

    Experiments register themselves with a name, and can be looked up
    by name from the unified entry point.
    """

    _items = {}
    _registry_label = "experiment"

    @classmethod
    def get_all(cls) -> dict[str, type]:
        """Get all registered experiment runner classes."""
        return dict(cls._items)
