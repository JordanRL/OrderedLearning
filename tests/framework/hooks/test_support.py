"""Tests for framework/hooks/support.py — demand-driven construction helpers."""

from unittest.mock import MagicMock

import torch
import torch.nn as nn
from torch.optim import SGD

from framework.hooks.support import (
    build_intervention_context_if_needed,
    setup_grad_accumulation_if_needed,
    capture_prev_step_grads_if_needed,
    capture_pre_epoch_state_if_needed,
)
from framework.hooks.hook_point import HookPoint
from framework.contexts import BackpropInterventionContext


class TestBuildInterventionContextIfNeeded:

    def test_returns_none_when_no_hook_manager(self):
        """Returns None when hook_manager is None."""
        result = build_intervention_context_if_needed(
            None, HookPoint.POST_STEP, 0,
        )
        assert result is None

    def test_returns_none_when_no_interventions_active(self):
        """Returns None when no intervention hooks are active at the point."""
        hm = MagicMock()
        hm.has_active_step_interventions.return_value = False
        result = build_intervention_context_if_needed(
            hm, HookPoint.POST_STEP, 0,
        )
        assert result is None

    def test_returns_context_when_interventions_active(self):
        """Returns BackpropInterventionContext when interventions are active."""
        hm = MagicMock()
        hm.has_active_step_interventions.return_value = True
        model = nn.Linear(4, 2)
        optimizer = SGD(model.parameters(), lr=0.01)
        result = build_intervention_context_if_needed(
            hm, HookPoint.POST_STEP, 0,
            model=model, optimizer=optimizer, scheduler=None,
            criterion=None, loader=None, config=None,
            device=torch.device('cpu'),
        )
        assert isinstance(result, BackpropInterventionContext)

    def test_uses_components_kwargs(self):
        """Uses components.build_intervention_context_kwargs() when provided."""
        hm = MagicMock()
        hm.has_interventions_at.return_value = True
        model = nn.Linear(4, 2)
        optimizer = SGD(model.parameters(), lr=0.01)

        components = MagicMock()
        components.build_intervention_context_kwargs.return_value = {
            'model': model,
            'optimizer': optimizer,
            'scheduler': None,
            'criterion': None,
            'loader': None,
            'config': None,
            'device': torch.device('cpu'),
            'pre_epoch_state': None,
            'current_batch': None,
            'profiler': None,
            'loss_fn': None,
            'grad_scaler': None,
        }
        result = build_intervention_context_if_needed(
            hm, HookPoint.SNAPSHOT, 0,
            components=components, loader=None, config=None,
            device=torch.device('cpu'),
        )
        assert isinstance(result, BackpropInterventionContext)
        components.build_intervention_context_kwargs.assert_called_once()


class TestSetupGradAccumulationIfNeeded:

    def test_returns_none_when_not_needed(self):
        """Returns (None, 0) when no hooks need accumulation."""
        model = nn.Linear(4, 2)
        result = setup_grad_accumulation_if_needed(
            None, model, None, is_snapshot=False, record_trajectory=False,
        )
        assert result == (None, 0)

    def test_returns_accumulator_for_trajectory_snapshot(self):
        """Returns accumulator when record_trajectory and is_snapshot."""
        model = nn.Linear(4, 2)
        accum, count = setup_grad_accumulation_if_needed(
            None, model, None, is_snapshot=True, record_trajectory=True,
        )
        assert accum is not None
        assert count == 0


class TestCapturePrevStepGradsIfNeeded:

    def test_returns_none_when_no_hook_manager(self):
        """Returns None when hook_manager is None."""
        model = nn.Linear(4, 2)
        assert capture_prev_step_grads_if_needed(None, model, 0) is None

    def test_returns_none_when_not_needed(self):
        """Returns None when hook_manager says no prev step grads needed."""
        hm = MagicMock()
        hm.needs_prev_step_grads_this_step.return_value = False
        model = nn.Linear(4, 2)
        assert capture_prev_step_grads_if_needed(hm, model, 0) is None

    def test_captures_grads_when_needed(self):
        """Returns gradient dict when grads exist and needed."""
        hm = MagicMock()
        hm.needs_prev_step_grads_this_step.return_value = True
        model = nn.Linear(4, 2)
        # Create gradients via forward-backward
        loss = model(torch.randn(2, 4)).sum()
        loss.backward()
        result = capture_prev_step_grads_if_needed(hm, model, 0)
        assert isinstance(result, dict)
        assert len(result) > 0


class TestCapturePreEpochStateIfNeeded:

    def test_returns_none_when_no_hook_manager(self):
        """Returns None when hook_manager is None."""
        model = nn.Linear(4, 2)
        optimizer = SGD(model.parameters(), lr=0.01)
        assert capture_pre_epoch_state_if_needed(None, model, optimizer, 0) is None

    def test_returns_none_when_not_needed(self):
        """Returns None when neither POST_EPOCH nor SNAPSHOT need pre-epoch state."""
        hm = MagicMock()
        hm.needs_pre_epoch_state_at.return_value = False
        model = nn.Linear(4, 2)
        optimizer = SGD(model.parameters(), lr=0.01)
        assert capture_pre_epoch_state_if_needed(hm, model, optimizer, 0) is None

    def test_captures_state_when_needed(self):
        """Returns state dict with model/optimizer/rng keys when needed."""
        hm = MagicMock()
        hm.needs_pre_epoch_state_at.return_value = True
        model = nn.Linear(4, 2)
        optimizer = SGD(model.parameters(), lr=0.01)
        result = capture_pre_epoch_state_if_needed(hm, model, optimizer, 0)
        assert isinstance(result, dict)
        assert 'model' in result
        assert 'optimizer' in result
        assert 'rng_cpu' in result
