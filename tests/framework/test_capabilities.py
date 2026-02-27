"""Tests for framework/capabilities.py — Flag enums, requirements, capability matching."""

import pytest
from framework.capabilities import (
    TrainingParadigm, ModelCapability, GradientAvailability,
    HookNeeds, HookRequirements, TrainingCapabilities,
)


class TestTrainingParadigm:
    """Flag enum bitwise behavior."""

    def test_single_flag_identity(self):
        """Single paradigm flags are distinct."""
        assert TrainingParadigm.BACKPROP != TrainingParadigm.ADVERSARIAL

    def test_flag_combination(self):
        """Flags can be combined with bitwise OR."""
        combined = TrainingParadigm.BACKPROP | TrainingParadigm.ADVERSARIAL
        assert TrainingParadigm.BACKPROP in combined
        assert TrainingParadigm.ADVERSARIAL in combined
        assert TrainingParadigm.LOCAL_LEARNING not in combined

    def test_all_paradigms_defined(self):
        """All six paradigms exist."""
        expected = {'BACKPROP', 'ADVERSARIAL', 'LOCAL_LEARNING', 'NESTED', 'ROLLOUT', 'EVOLUTIONARY'}
        actual = {m.name for m in TrainingParadigm}
        assert expected == actual


class TestTrainingCapabilities:
    """TrainingCapabilities.satisfies() and describe_unsatisfied()."""

    def test_satisfies_all_none_requirements(self):
        """All-None requirements are universally satisfied."""
        caps = TrainingCapabilities(
            paradigm=TrainingParadigm.BACKPROP,
            model_capabilities=ModelCapability.PARAMETERS,
            gradient_availability=GradientAvailability.GLOBAL_GRADIENTS,
        )
        reqs = HookRequirements()  # all None
        assert caps.satisfies(reqs) is True

    def test_satisfies_matching_paradigm(self):
        """Capability with matching paradigm passes."""
        caps = TrainingCapabilities(
            paradigm=TrainingParadigm.BACKPROP,
            model_capabilities=ModelCapability.PARAMETERS | ModelCapability.GLOBAL_LOSS,
            gradient_availability=GradientAvailability.GLOBAL_GRADIENTS,
        )
        reqs = HookRequirements(paradigm=TrainingParadigm.BACKPROP)
        assert caps.satisfies(reqs) is True

    def test_rejects_wrong_paradigm(self):
        """Non-overlapping paradigm requirement fails."""
        caps = TrainingCapabilities(
            paradigm=TrainingParadigm.BACKPROP,
            model_capabilities=ModelCapability.PARAMETERS,
            gradient_availability=GradientAvailability.GLOBAL_GRADIENTS,
        )
        reqs = HookRequirements(paradigm=TrainingParadigm.ADVERSARIAL)
        assert caps.satisfies(reqs) is False

    def test_rejects_missing_model_capability(self):
        """Missing required model capability fails."""
        caps = TrainingCapabilities(
            paradigm=TrainingParadigm.BACKPROP,
            model_capabilities=ModelCapability.PARAMETERS,
            gradient_availability=GradientAvailability.GLOBAL_GRADIENTS,
        )
        reqs = HookRequirements(
            model_capabilities=ModelCapability.PARAMETERS | ModelCapability.ATTENTION,
        )
        assert caps.satisfies(reqs) is False

    def test_satisfies_model_capabilities_superset(self):
        """Capabilities that are a superset of requirements pass."""
        caps = TrainingCapabilities(
            paradigm=TrainingParadigm.BACKPROP,
            model_capabilities=ModelCapability.PARAMETERS | ModelCapability.EMBEDDINGS | ModelCapability.ATTENTION,
            gradient_availability=GradientAvailability.GLOBAL_GRADIENTS,
        )
        reqs = HookRequirements(model_capabilities=ModelCapability.PARAMETERS)
        assert caps.satisfies(reqs) is True

    def test_satisfies_model_capabilities_exact_match(self):
        """Capabilities that exactly match requirements pass."""
        caps = TrainingCapabilities(
            paradigm=TrainingParadigm.BACKPROP,
            model_capabilities=ModelCapability.PARAMETERS | ModelCapability.ATTENTION,
            gradient_availability=GradientAvailability.GLOBAL_GRADIENTS,
        )
        reqs = HookRequirements(
            model_capabilities=ModelCapability.PARAMETERS | ModelCapability.ATTENTION,
        )
        assert caps.satisfies(reqs) is True

    def test_satisfies_matching_gradient_availability(self):
        """Matching gradient availability passes."""
        caps = TrainingCapabilities(
            paradigm=TrainingParadigm.BACKPROP,
            model_capabilities=ModelCapability.PARAMETERS,
            gradient_availability=GradientAvailability.GLOBAL_GRADIENTS,
        )
        reqs = HookRequirements(gradient_availability=GradientAvailability.GLOBAL_GRADIENTS)
        assert caps.satisfies(reqs) is True

    def test_rejects_wrong_gradient_availability(self):
        """Non-overlapping gradient availability requirement fails."""
        caps = TrainingCapabilities(
            paradigm=TrainingParadigm.BACKPROP,
            model_capabilities=ModelCapability.PARAMETERS,
            gradient_availability=GradientAvailability.GLOBAL_GRADIENTS,
        )
        reqs = HookRequirements(gradient_availability=GradientAvailability.POLICY_GRADIENTS)
        assert caps.satisfies(reqs) is False

    def test_describe_unsatisfied_model_capabilities(self):
        """describe_unsatisfied returns reason for missing model capabilities."""
        caps = TrainingCapabilities(
            paradigm=TrainingParadigm.BACKPROP,
            model_capabilities=ModelCapability.PARAMETERS,
            gradient_availability=GradientAvailability.GLOBAL_GRADIENTS,
        )
        reqs = HookRequirements(
            model_capabilities=ModelCapability.PARAMETERS | ModelCapability.ATTENTION,
        )
        reasons = caps.describe_unsatisfied(reqs)
        assert len(reasons) == 1
        assert "model capabilities" in reasons[0]

    def test_describe_unsatisfied_gradient_availability(self):
        """describe_unsatisfied returns reason for gradient mismatch."""
        caps = TrainingCapabilities(
            paradigm=TrainingParadigm.BACKPROP,
            model_capabilities=ModelCapability.PARAMETERS,
            gradient_availability=GradientAvailability.GLOBAL_GRADIENTS,
        )
        reqs = HookRequirements(gradient_availability=GradientAvailability.POLICY_GRADIENTS)
        reasons = caps.describe_unsatisfied(reqs)
        assert len(reasons) == 1
        assert "gradients" in reasons[0]

    def test_describe_unsatisfied_paradigm(self):
        """describe_unsatisfied returns human-readable reason for paradigm mismatch."""
        caps = TrainingCapabilities(
            paradigm=TrainingParadigm.EVOLUTIONARY,
            model_capabilities=ModelCapability.PARAMETERS,
            gradient_availability=GradientAvailability.NONE,
        )
        reqs = HookRequirements(paradigm=TrainingParadigm.BACKPROP)
        reasons = caps.describe_unsatisfied(reqs)
        assert len(reasons) == 1
        assert "paradigm" in reasons[0]

    def test_describe_unsatisfied_empty_when_satisfied(self):
        """describe_unsatisfied returns empty list when all requirements met."""
        caps = TrainingCapabilities(
            paradigm=TrainingParadigm.BACKPROP,
            model_capabilities=ModelCapability.PARAMETERS,
            gradient_availability=GradientAvailability.GLOBAL_GRADIENTS,
        )
        reqs = HookRequirements(paradigm=TrainingParadigm.BACKPROP)
        assert caps.describe_unsatisfied(reqs) == []


class TestHookNeeds:
    """HookNeeds flag composition."""

    def test_none_is_falsy(self):
        """NONE is the zero flag (falsy)."""
        assert not HookNeeds.NONE
        assert HookNeeds.NONE.value == 0

    def test_combined_needs_membership(self):
        """Combined flag correctly reports membership."""
        needs = HookNeeds.ACCUMULATED_GRADS | HookNeeds.REFERENCE_WEIGHTS
        assert HookNeeds.ACCUMULATED_GRADS in needs
        assert HookNeeds.REFERENCE_WEIGHTS in needs
        assert HookNeeds.PREV_STEP_GRADS not in needs
