"""Tests for framework/registry.py — Registry base class and @register decorator."""

import pytest
from framework.registry import Registry


class FreshRegistry(Registry):
    """Isolated registry for testing — avoids polluting real registries."""
    _items = {}
    _registry_label = "test item"


@pytest.fixture(autouse=True)
def clean_registry():
    """Clear the test registry before each test."""
    FreshRegistry._items = {}
    yield
    FreshRegistry._items = {}


class TestRegisterDecorator:

    def test_register_with_string_name(self):
        """@register("name") registers the class under the given name."""
        @FreshRegistry.register("my_item")
        class Foo:
            pass
        assert FreshRegistry.get("my_item") is Foo

    def test_register_bare_reads_name_attribute(self):
        """@register (no parens, no arg) reads .name from a temporary instance."""
        @FreshRegistry.register
        class Bar:
            name = "bar_item"
        assert FreshRegistry.get("bar_item") is Bar

    def test_register_empty_parens(self):
        """@register() with empty parens reads .name from a temporary instance."""
        @FreshRegistry.register()
        class Baz:
            name = "baz_item"
        assert FreshRegistry.get("baz_item") is Baz

    def test_register_invalid_arg_raises(self):
        """Passing a non-string non-class argument raises TypeError."""
        with pytest.raises(TypeError):
            FreshRegistry.register(42)


class TestRegistryLookup:

    def test_get_unknown_raises_value_error(self):
        """Requesting an unregistered name raises ValueError with available list."""
        with pytest.raises(ValueError, match="Unknown test item"):
            FreshRegistry.get("nonexistent")

    def test_list_all_sorted(self):
        """list_all() returns registered names in sorted order."""
        @FreshRegistry.register("zebra")
        class Z: pass

        @FreshRegistry.register("alpha")
        class A: pass

        assert FreshRegistry.list_all() == ["alpha", "zebra"]
