"""Tests for framework/hooks/registry.py — HookRegistry filtering methods."""

from framework.hooks.registry import HookRegistry
from framework.hooks.training_hook import TrainingHook
from framework.hooks.intervention_hook import InterventionHook
from framework.hooks.hook_point import HookPoint


# ---- Test hook classes ----

class _ObserverHook(TrainingHook):
    name = "_test_observer"
    hook_points = {HookPoint.POST_STEP}
    debug = False

    def compute(self, ctx, **state):
        return {}


class _InterventionTestHook(InterventionHook):
    name = "_test_intervention"
    hook_points = {HookPoint.POST_EPOCH}
    debug = False

    def compute(self, ctx, **state):
        return {}

    def intervene(self, ctx, model_ctx, **state):
        pass


class _DebugHook(TrainingHook):
    name = "_test_debug"
    hook_points = {HookPoint.SNAPSHOT}
    debug = True

    def compute(self, ctx, **state):
        return {}


# ---- Setup: register test hooks, cleanup after ----

def _register_test_hooks():
    """Register test hooks and return the names to clean up."""
    names = ['_test_observer', '_test_intervention', '_test_debug']
    HookRegistry._items['_test_observer'] = _ObserverHook
    HookRegistry._items['_test_intervention'] = _InterventionTestHook
    HookRegistry._items['_test_debug'] = _DebugHook
    return names


def _unregister_test_hooks(names):
    for name in names:
        HookRegistry._items.pop(name, None)


class TestHookRegistryFiltering:

    def setup_method(self):
        self._names = _register_test_hooks()

    def teardown_method(self):
        _unregister_test_hooks(self._names)

    def test_list_all_excludes_debug(self):
        """list_all() includes observers and interventions but not debug."""
        all_hooks = HookRegistry.list_all()
        assert '_test_observer' in all_hooks
        assert '_test_intervention' in all_hooks
        assert '_test_debug' not in all_hooks

    def test_list_observers_excludes_interventions_and_debug(self):
        """list_observers() only includes non-intervention, non-debug hooks."""
        observers = HookRegistry.list_observers()
        assert '_test_observer' in observers
        assert '_test_intervention' not in observers
        assert '_test_debug' not in observers

    def test_list_interventions_only_interventions(self):
        """list_interventions() only includes InterventionHook subclasses."""
        interventions = HookRegistry.list_interventions()
        assert '_test_intervention' in interventions
        assert '_test_observer' not in interventions

    def test_list_debug_only_debug(self):
        """list_debug() only includes debug=True hooks."""
        debug = HookRegistry.list_debug()
        assert '_test_debug' in debug
        assert '_test_observer' not in debug

    def test_get_all_info_returns_metadata(self):
        """get_all_info() returns list of dicts with expected keys."""
        info = HookRegistry.get_all_info()
        test_entry = next(
            (i for i in info if i['name'] == '_test_observer'), None
        )
        assert test_entry is not None
        assert 'description' in test_entry
        assert 'hook_points' in test_entry
        assert 'is_intervention' in test_entry
        assert test_entry['is_intervention'] is False
