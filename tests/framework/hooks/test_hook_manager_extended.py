"""Extended tests for HookManager: intervention firing, query methods,
reference weights, JSONL logging, state offloading, and hook-name construction."""

import json
import os

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock

from framework.hooks.hook_point import HookPoint, StepSchedule
from framework.hooks.training_hook import TrainingHook
from framework.hooks.intervention_hook import InterventionHook
from framework.hooks.manager import HookManager
from framework.contexts.run_context import RunContext
from framework.capabilities import HookNeeds


# ---- Test hook helpers ----

class DummyInterventionHook(InterventionHook):
    """Intervention hook that tracks calls to intervene() and compute()."""
    name = "dummy_intervention"
    hook_points = {HookPoint.POST_EPOCH, HookPoint.SNAPSHOT}
    intervention_points = {HookPoint.POST_EPOCH}

    def __init__(self):
        self.intervene_calls = []
        self.compute_calls = []

    def intervene(self, ctx, model_ctx, **state):
        self.intervene_calls.append(ctx.hook_point)
        return {"intervention_metric": 99.0}

    def compute(self, ctx, **state):
        self.compute_calls.append(ctx.hook_point)
        return {"observer_metric": 1.0}


class AllPointInterventionHook(InterventionHook):
    """Intervention hook with intervention_points=None (all points are intervention)."""
    name = "all_point_intervention"
    hook_points = {HookPoint.POST_EPOCH, HookPoint.SNAPSHOT}
    intervention_points = None  # all hook_points are intervention points

    def __init__(self):
        self.intervene_calls = []

    def intervene(self, ctx, model_ctx, **state):
        self.intervene_calls.append(ctx.hook_point)
        return {"all_metric": 1.0}


class WeightModifyingInterventionHook(InterventionHook):
    """Intervention hook that modifies model weights during intervene()."""
    name = "weight_modifier"
    hook_points = {HookPoint.POST_EPOCH}
    intervention_points = {HookPoint.POST_EPOCH}

    def intervene(self, ctx, model_ctx, **state):
        with torch.no_grad():
            for param in model_ctx.model.parameters():
                param.fill_(999.0)
        return {"modified": True}

    def compute(self, ctx, **state):
        return {}


class RefWeightsNeedingHook(TrainingHook):
    """Hook that declares REFERENCE_WEIGHTS need."""
    name = "ref_needer"
    hook_points = {HookPoint.SNAPSHOT}
    needs = HookNeeds.REFERENCE_WEIGHTS

    def compute(self, ctx, **state):
        return {"ref_available": hasattr(self, '_ref') and self._ref is not None}


class GradAccumNeedingHook(TrainingHook):
    """Hook that declares ACCUMULATED_GRADS need at SNAPSHOT."""
    name = "grad_accum_needer"
    hook_points = {HookPoint.SNAPSHOT, HookPoint.POST_STEP}
    needs = HookNeeds.ACCUMULATED_GRADS

    def compute(self, ctx, **state):
        return {}


class PreEpochNeedingInterventionHook(InterventionHook):
    """Intervention hook that declares PRE_EPOCH_STATE need."""
    name = "pre_epoch_needer"
    hook_points = {HookPoint.POST_EPOCH}
    intervention_points = {HookPoint.POST_EPOCH}
    needs = HookNeeds.PRE_EPOCH_STATE

    def intervene(self, ctx, model_ctx, **state):
        return {}

    def compute(self, ctx, **state):
        return {}


class PrevStepGradsHook(TrainingHook):
    """Hook that declares PREV_STEP_GRADS need at POST_STEP."""
    name = "prev_step_grads"
    hook_points = {HookPoint.POST_STEP, HookPoint.SNAPSHOT}
    needs = HookNeeds.PREV_STEP_GRADS

    def compute(self, ctx, **state):
        return {}


class StatefulHook(TrainingHook):
    """Hook with offloadable tensor state."""
    name = "stateful_hook"
    hook_points = {HookPoint.POST_EPOCH}

    def __init__(self):
        self._state_tensor = torch.ones(4)

    def compute(self, ctx, **state):
        return {"state_sum": self._state_tensor.sum().item()}

    def get_state_tensors(self):
        return {"internal": self._state_tensor}

    def set_state_tensors(self, tensors):
        self._state_tensor = tensors["internal"]


class StepInterventionHook(InterventionHook):
    """Intervention hook at POST_STEP with step schedule."""
    name = "step_intervention"
    hook_points = {HookPoint.POST_STEP}
    intervention_points = {HookPoint.POST_STEP}
    step_schedule = StepSchedule(mode='stride', stride=2)

    def intervene(self, ctx, model_ctx, **state):
        return {"stepped": 1.0}

    def compute(self, ctx, **state):
        return {}


class DummyObserver(TrainingHook):
    """Minimal observer for helper tests."""
    name = "dummy_observer"
    hook_points = {HookPoint.POST_EPOCH, HookPoint.SNAPSHOT}

    def compute(self, ctx, **state):
        return {"value": 42.0}


# ---- Fixtures ----

@pytest.fixture
def post_epoch_ctx():
    return RunContext(hook_point=HookPoint.POST_EPOCH, epoch=1)


@pytest.fixture
def snapshot_ctx():
    return RunContext(hook_point=HookPoint.SNAPSHOT, epoch=1)


@pytest.fixture
def tiny_intervention_ctx():
    """Minimal BackpropInterventionContext for testing intervention firing."""
    from framework.contexts.model_context import BackpropInterventionContext

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    criterion = nn.MSELoss()
    loss_fn = lambda m, b: m(b).sum()

    return BackpropInterventionContext(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        loader=None,
        config=None,
        device=torch.device('cpu'),
        loss_fn=loss_fn,
    )


# ---- TestHookNameConstruction ----

class TestHookNameConstruction:

    def test_instantiates_from_registry(self):
        """HookManager with hook_names looks up hooks via HookRegistry."""
        hm = HookManager(hook_names=['training_metrics'], step_metrics_log=None)
        hook = hm.get_hook('training_metrics')
        assert hook is not None
        assert hook.name == 'training_metrics'

    def test_hook_names_with_config(self):
        """Per-hook config kwargs are passed through to instantiation."""
        # training_metrics doesn't take kwargs, but the config lookup path is exercised
        hm = HookManager(
            hook_names=['training_metrics'],
            hook_config={'training_metrics': {}},
            step_metrics_log=None,
        )
        assert hm.get_hook('training_metrics') is not None

    def test_unknown_hook_name_raises(self):
        """Unknown hook name raises an error from the registry."""
        with pytest.raises(Exception):
            HookManager(hook_names=['nonexistent_hook_xyz'], step_metrics_log=None)


# ---- TestReferenceWeightsIntegration ----

class TestReferenceWeightsIntegration:

    def test_initialized_when_hook_needs_them(self):
        """ReferenceWeights created when a hook declares REFERENCE_WEIGHTS need."""
        hook = RefWeightsNeedingHook()
        hm = HookManager(hooks=[hook], step_metrics_log=None)
        assert hm._reference_weights is not None

    def test_set_reference_weights_propagated_to_hook(self):
        """Hook receives the shared ReferenceWeights instance."""
        hook = RefWeightsNeedingHook()
        HookManager(hooks=[hook], step_metrics_log=None)
        assert hasattr(hook, '_ref')
        assert hook._ref is not None

    def test_not_initialized_when_not_needed(self):
        """No ReferenceWeights when no hook needs them."""
        hm = HookManager(hooks=[DummyObserver()], step_metrics_log=None)
        assert hm._reference_weights is None

    def test_reset_all_resets_reference_weights(self):
        """reset_all() resets the shared ReferenceWeights."""
        hook = RefWeightsNeedingHook()
        hm = HookManager(hooks=[hook], step_metrics_log=None)
        # Simulate resolving some context
        hm._reference_weights.set_run_context(strategy='test')
        hm.reset_all()
        # After reset, path should be back to template
        assert '{' in hm._reference_weights._path

    def test_set_run_context_propagates_to_reference_weights(self):
        """set_run_context() resolves template variables in ReferenceWeights."""
        hook = RefWeightsNeedingHook()
        hm = HookManager(hooks=[hook], step_metrics_log=None)
        hm.set_run_context(strategy='stride')
        assert 'stride' in hm._reference_weights._path


# ---- TestQueryMethods ----

class TestQueryMethods:

    def test_needs_grad_accumulation_at_matching_point(self):
        """needs_grad_accumulation_at returns True at hook's registered point."""
        hm = HookManager(hooks=[GradAccumNeedingHook()], step_metrics_log=None)
        assert hm.needs_grad_accumulation_at(HookPoint.SNAPSHOT) is True

    def test_needs_grad_accumulation_at_non_matching_point(self):
        """needs_grad_accumulation_at returns False at unregistered point."""
        hm = HookManager(hooks=[GradAccumNeedingHook()], step_metrics_log=None)
        assert hm.needs_grad_accumulation_at(HookPoint.PRE_EPOCH) is False

    def test_has_interventions_true(self):
        """has_interventions() returns True when intervention hooks exist."""
        hm = HookManager(hooks=[DummyInterventionHook()], step_metrics_log=None)
        assert hm.has_interventions() is True

    def test_has_interventions_false(self):
        """has_interventions() returns False with only observer hooks."""
        hm = HookManager(hooks=[DummyObserver()], step_metrics_log=None)
        assert hm.has_interventions() is False

    def test_has_interventions_at_respects_intervention_points(self):
        """has_interventions_at checks intervention_points, not just hook_points."""
        hook = DummyInterventionHook()  # intervention_points = {POST_EPOCH}
        hm = HookManager(hooks=[hook], step_metrics_log=None)
        # POST_EPOCH is an intervention point
        assert hm.has_interventions_at(HookPoint.POST_EPOCH) is True
        # SNAPSHOT is a hook_point but NOT an intervention point — observer mode
        assert hm.has_interventions_at(HookPoint.SNAPSHOT) is False

    def test_has_interventions_at_all_points(self):
        """intervention_points=None means all hook_points are intervention points."""
        hook = AllPointInterventionHook()
        hm = HookManager(hooks=[hook], step_metrics_log=None)
        assert hm.has_interventions_at(HookPoint.POST_EPOCH) is True
        assert hm.has_interventions_at(HookPoint.SNAPSHOT) is True

    def test_has_active_step_interventions(self):
        """has_active_step_interventions respects step schedule."""
        hook = StepInterventionHook()  # stride=2
        hm = HookManager(hooks=[hook], step_metrics_log=None)

        hm.advance_step()  # step=0
        assert hm.has_active_step_interventions(HookPoint.POST_STEP) is True

        hm.advance_step()  # step=1
        assert hm.has_active_step_interventions(HookPoint.POST_STEP) is False

    def test_needs_pre_epoch_state(self):
        """needs_pre_epoch_state() returns True when a hook declares the need."""
        hm = HookManager(hooks=[PreEpochNeedingInterventionHook()], step_metrics_log=None)
        assert hm.needs_pre_epoch_state() is True

    def test_needs_pre_epoch_state_false(self):
        """needs_pre_epoch_state() returns False when no hook needs it."""
        hm = HookManager(hooks=[DummyObserver()], step_metrics_log=None)
        assert hm.needs_pre_epoch_state() is False

    def test_needs_pre_epoch_state_at(self):
        """needs_pre_epoch_state_at() checks both need and intervention point."""
        hook = PreEpochNeedingInterventionHook()
        hm = HookManager(hooks=[hook], step_metrics_log=None)
        assert hm.needs_pre_epoch_state_at(HookPoint.POST_EPOCH) is True

    def test_needs_prev_step_grads(self):
        """needs_prev_step_grads() returns True when a hook declares the need."""
        hm = HookManager(hooks=[PrevStepGradsHook()], step_metrics_log=None)
        assert hm.needs_prev_step_grads() is True

    def test_needs_prev_step_grads_false(self):
        """needs_prev_step_grads() returns False when no hook needs it."""
        hm = HookManager(hooks=[DummyObserver()], step_metrics_log=None)
        assert hm.needs_prev_step_grads() is False

    def test_needs_prev_step_grads_at(self):
        """needs_prev_step_grads_at() checks at specific hook point."""
        hm = HookManager(hooks=[PrevStepGradsHook()], step_metrics_log=None)
        assert hm.needs_prev_step_grads_at(HookPoint.POST_STEP) is True
        assert hm.needs_prev_step_grads_at(HookPoint.PRE_EPOCH) is False

    def test_needs_prev_step_grads_this_step(self):
        """needs_prev_step_grads_this_step() checks POST_STEP at current step."""
        hm = HookManager(hooks=[PrevStepGradsHook()], step_metrics_log=None)
        hm.advance_step()  # step=0
        assert hm.needs_prev_step_grads_this_step() is True


# ---- TestInterventionFiring ----

class TestInterventionFiring:

    def test_fire_calls_intervene_with_model_ctx(
        self, post_epoch_ctx, tiny_intervention_ctx,
    ):
        """fire() calls intervene() on intervention hooks at their intervention points."""
        hook = DummyInterventionHook()
        hm = HookManager(hooks=[hook], step_metrics_log=None)
        metrics = hm.fire(
            HookPoint.POST_EPOCH, post_epoch_ctx, model_ctx=tiny_intervention_ctx,
        )
        assert len(hook.intervene_calls) == 1
        assert hook.intervene_calls[0] == HookPoint.POST_EPOCH
        assert "dummy_intervention/intervention_metric" in metrics

    def test_guardian_token_saves_and_restores(self, post_epoch_ctx, tiny_intervention_ctx):
        """Guardian token pattern: model state is restored after intervention modifies it."""
        hook = WeightModifyingInterventionHook()
        hm = HookManager(hooks=[hook], step_metrics_log=None)
        model = tiny_intervention_ctx.model

        params_before = [p.clone() for p in model.parameters()]
        hm.fire(HookPoint.POST_EPOCH, post_epoch_ctx, model_ctx=tiny_intervention_ctx)
        params_after = list(model.parameters())

        for p_before, p_after in zip(params_before, params_after):
            assert torch.allclose(p_before, p_after), "Model state not restored after intervention"

    def test_observer_mode_at_non_intervention_points(
        self, tiny_intervention_ctx,
    ):
        """At non-intervention points, compute() is called instead of intervene()."""
        hook = DummyInterventionHook()  # intervention_points = {POST_EPOCH}
        hm = HookManager(hooks=[hook], step_metrics_log=None)
        snapshot_ctx = RunContext(hook_point=HookPoint.SNAPSHOT, epoch=1)
        metrics = hm.fire(
            HookPoint.SNAPSHOT, snapshot_ctx, model_ctx=tiny_intervention_ctx,
        )
        assert len(hook.compute_calls) == 1
        assert len(hook.intervene_calls) == 0
        assert "dummy_intervention/observer_metric" in metrics

    def test_fire_without_model_ctx_skips_interventions(self, post_epoch_ctx):
        """When model_ctx=None, intervention hooks don't fire intervene()."""
        hook = DummyInterventionHook()
        hm = HookManager(hooks=[hook], step_metrics_log=None)
        hm.fire(HookPoint.POST_EPOCH, post_epoch_ctx, model_ctx=None)
        assert len(hook.intervene_calls) == 0


# ---- TestEpochGatingLowerBound ----

class TestEpochGatingLowerBound:

    def test_hook_inactive_before_min_epoch(self):
        """Epoch-gated hook does not fire when epoch < min_epoch."""
        class LateStartHook(TrainingHook):
            name = "late_start"
            hook_points = {HookPoint.POST_EPOCH}
            loop_points = {'epoch': {HookPoint.POST_EPOCH: (5, 10)}}

            def compute(self, ctx, **state):
                return {"val": 1.0}

        hm = HookManager(
            hooks=[LateStartHook()], step_metrics_log=None, loop_type='epoch',
        )
        ctx = RunContext(hook_point=HookPoint.POST_EPOCH, epoch=3)
        metrics = hm.fire(HookPoint.POST_EPOCH, ctx)
        assert metrics == {}


# ---- TestJSONLStepMetrics ----

class TestJSONLStepMetrics:

    def test_flush_writes_jsonl_file(self, tmp_path):
        """flush_step_metrics() writes accumulated metrics to a JSONL file."""
        log_path = str(tmp_path / "raw_logs" / "step_metrics.jsonl")

        class StepHook(TrainingHook):
            name = "step_hook"
            hook_points = {HookPoint.POST_STEP}

            def compute(self, ctx, **state):
                return {"val": 1.0}

        hm = HookManager(hooks=[StepHook()], step_metrics_log=log_path)

        # Fire a few POST_STEP events to populate buffer
        for step in range(3):
            hm.advance_step()
            ctx = RunContext(hook_point=HookPoint.POST_STEP, epoch=0, step=step)
            hm.fire(HookPoint.POST_STEP, ctx)

        hm.flush_step_metrics(epoch=0)

        assert os.path.exists(log_path)
        with open(log_path) as f:
            line = f.readline()
            record = json.loads(line)
        assert record["epoch"] == 0
        assert "step_hook/val" in record

    def test_flush_sinks_closes_jsonl_file(self, tmp_path):
        """flush_sinks() closes the JSONL file handle."""
        log_path = str(tmp_path / "raw_logs" / "step_metrics.jsonl")

        class StepHook(TrainingHook):
            name = "step_hook"
            hook_points = {HookPoint.POST_STEP}

            def compute(self, ctx, **state):
                return {"val": 1.0}

        hm = HookManager(hooks=[StepHook()], step_metrics_log=log_path)

        hm.advance_step()
        ctx = RunContext(hook_point=HookPoint.POST_STEP, epoch=0, step=0)
        hm.fire(HookPoint.POST_STEP, ctx)
        hm.flush_step_metrics(epoch=0)
        assert hm._step_metrics_log_file is not None

        hm.flush_sinks()
        assert hm._step_metrics_log_file is None

    def test_multiple_flushes_append(self, tmp_path):
        """Multiple flush_step_metrics calls append to the same file."""
        log_path = str(tmp_path / "raw_logs" / "step_metrics.jsonl")

        class StepHook(TrainingHook):
            name = "step_hook"
            hook_points = {HookPoint.POST_STEP}

            def compute(self, ctx, **state):
                return {"val": 1.0}

        hm = HookManager(hooks=[StepHook()], step_metrics_log=log_path)

        for epoch in range(2):
            hm.advance_step()
            ctx = RunContext(hook_point=HookPoint.POST_STEP, epoch=epoch, step=0)
            hm.fire(HookPoint.POST_STEP, ctx)
            hm.flush_step_metrics(epoch=epoch)

        with open(log_path) as f:
            lines = f.readlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["epoch"] == 0
        assert json.loads(lines[1])["epoch"] == 1


# ---- TestStateOffloading ----

class TestStateOffloading:

    def test_offload_calls_set_state_tensors(self):
        """_offload_states() reads tensors via get_state_tensors and writes back via set_state_tensors."""
        hook = StatefulHook()
        hm = HookManager(hooks=[hook], step_metrics_log=None)
        # Verify the tensor is the original one before offload
        original_tensor = hook._state_tensor
        hm._offload_states()
        # After offload, set_state_tensors was called — verify the tensor
        # is still functional (contiguous CPU tensor with correct values)
        tensors = hook.get_state_tensors()
        assert tensors["internal"].device == torch.device('cpu')
        assert torch.allclose(tensors["internal"], torch.ones(4))

    def test_restore_states_with_none_device_is_noop(self):
        """_restore_states(None) is a no-op (no error, no device movement)."""
        hook = StatefulHook()
        hm = HookManager(hooks=[hook], step_metrics_log=None)
        hm._restore_states(None)  # should not raise
        tensors = hook.get_state_tensors()
        assert tensors["internal"].device == torch.device('cpu')
