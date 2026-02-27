"""Meta-learning training strategies.

Implements MAML (with optional first_order mode for FOMAML) and Reptile.
Both operate on MetaLearningComponents and TaskSampler data.

The inner/outer loop structure is encapsulated entirely within train_step():
the trainer sees a standard step-based interface.
"""

from __future__ import annotations

from typing import Any

import torch

from .strategy_runner import StrategyRunner, StepResult
from ..capabilities import TrainingParadigm


class _ParameterizedModule:
    """Thin wrapper for torch.func.functional_call.

    Makes functional parameter application transparent to task_loss_fn,
    which can call this object the same way it would call a model:
        loss = task_loss_fn(parameterized_module, batch)

    Internally dispatches to torch.func.functional_call with the
    adapted parameters.
    """

    def __init__(self, module: torch.nn.Module, params: dict[str, torch.Tensor]):
        self._module = module
        self._params = params

    def __call__(self, *args, **kwargs):
        return torch.func.functional_call(self._module, self._params, args, kwargs)

    def parameters(self):
        """Yield parameter tensors (for compatibility)."""
        return self._params.values()

    def named_parameters(self):
        """Yield (name, param) pairs (for compatibility)."""
        return self._params.items()


class MAMLStep(StrategyRunner):
    """Model-Agnostic Meta-Learning (and FOMAML with first_order=True).

    Each train_step (one meta-step):
    1. Sample tasks from the TaskSampler
    2. For each task:
       a. Clone meta-parameters as tensors (for functional_call)
       b. Run K inner gradient steps on support set using functional_call
          (differentiable parameter updates, unless first_order=True)
       c. Compute meta-loss on query set with adapted parameters
    3. Average meta-losses, backward through inner loop
    4. Clip meta-gradients, meta-optimizer step

    Setup kwargs (via runner.get_strategy_kwargs()):
        inner_lr: float          -- inner loop learning rate (default 0.01)
        inner_steps: int         -- K inner gradient steps (default 5)
        tasks_per_step: int      -- number of tasks per meta-step (default 4)
        first_order: bool        -- True for FOMAML (default False)
    """

    paradigm = TrainingParadigm.NESTED

    def setup(self, *, components, config, device, **kwargs) -> None:
        self.model = components.model
        self.optimizer = components.optimizer
        self.device = device
        self._components = components
        self.task_loss_fn = components.task_loss_fn
        self.task_sampler = components.data

        if self.task_loss_fn is None:
            raise ValueError(
                "MAMLStep requires a task_loss_fn. "
                "Ensure the experiment provides task_loss_fn in MetaLearningComponents."
            )

        # Meta-learning hyperparameters (from strategy kwargs)
        self.inner_lr = kwargs.get('inner_lr', 0.01)
        self.inner_steps = kwargs.get('inner_steps', 5)
        self.tasks_per_step = kwargs.get('tasks_per_step', 4)
        self.first_order = kwargs.get('first_order', False)

        # AMP setup
        self._scaler = components.grad_scaler
        self._use_amp = self._scaler is not None
        self._autocast_device = device.type if device.type in ('cuda', 'cpu') else 'cpu'

        # Strategy caches (read by MetaLearningComponents for hooks)
        self._cached_meta_gradients: dict[str, torch.Tensor] | None = None
        self._cached_inner_gradients: list[dict[str, torch.Tensor]] | None = None
        self._cached_task_losses: list[float] | None = None

    def _inner_loop(self, task_batch):
        """Run K inner gradient steps on the support set.

        Uses torch.func.functional_call for differentiable parameter
        updates. When first_order=True, detaches gradients after each
        inner step (FOMAML).

        Returns:
            (meta_loss, inner_grads)
            meta_loss: loss on query set with adapted params (has grad_fn)
            inner_grads: last inner step gradients (detached), or None
        """
        # Start with meta-parameters as leaf tensors for autograd
        params = {
            name: p.clone()
            for name, p in self.model.named_parameters()
        }

        inner_grads = None

        for k in range(self.inner_steps):
            parameterized = _ParameterizedModule(self.model, params)

            with torch.autocast(device_type=self._autocast_device, enabled=self._use_amp):
                support_loss = self.task_loss_fn(parameterized, task_batch.support)

            # Compute gradients w.r.t. current adapted params
            grads = torch.autograd.grad(
                support_loss, params.values(),
                create_graph=not self.first_order,
            )

            # Update adapted parameters
            updated_params = {}
            for (name, p), g in zip(params.items(), grads):
                if self.first_order:
                    g = g.detach()
                updated_params[name] = p - self.inner_lr * g
            params = updated_params

            # Cache last inner gradients (detached for hooks)
            inner_grads = {
                name: g.detach()
                for name, g in zip(params.keys(), grads)
            }

        # Compute meta-loss on query set with adapted params
        parameterized = _ParameterizedModule(self.model, params)
        with torch.autocast(device_type=self._autocast_device, enabled=self._use_amp):
            meta_loss = self.task_loss_fn(parameterized, task_batch.query)

        return meta_loss, inner_grads

    def train_step(self, step: int, batch: Any = None) -> StepResult:
        """Execute one meta-step: sample tasks, inner adapt, meta-update."""
        tasks = self.task_sampler.sample(self.tasks_per_step)

        self.optimizer.zero_grad()

        total_meta_loss = torch.tensor(0.0, device=self.device)
        all_inner_grads = []
        task_losses = []

        for task_batch in tasks:
            meta_loss, inner_grads = self._inner_loop(task_batch)
            total_meta_loss = total_meta_loss + meta_loss
            task_losses.append(float(meta_loss.detach()))
            if inner_grads is not None:
                all_inner_grads.append(inner_grads)

        # Average meta-loss across tasks
        avg_meta_loss = total_meta_loss / len(tasks)

        # Backward through meta-loss to get meta-gradients
        if self._scaler:
            self._scaler.scale(avg_meta_loss).backward()
            self._scaler.unscale_(self.optimizer)
        else:
            avg_meta_loss.backward()

        self._components.clip_gradients()

        if self._scaler:
            self._scaler.step(self.optimizer)
            self._scaler.update()
        else:
            self.optimizer.step()

        # Cache state for hooks (via MetaLearningComponents.build_gradient_state)
        self._cached_meta_gradients = {
            name: p.grad.detach().clone()
            for name, p in self.model.named_parameters()
            if p.grad is not None
        }
        self._cached_inner_gradients = all_inner_grads if all_inner_grads else None
        self._cached_task_losses = task_losses

        return StepResult(
            metrics={
                'loss': avg_meta_loss.detach(),
                'mean_task_loss': sum(task_losses) / len(task_losses),
                'task_loss_std': (
                    torch.tensor(task_losses).std().item()
                    if len(task_losses) > 1 else 0.0
                ),
                'num_tasks': len(tasks),
            },
        )

    @property
    def name(self) -> str:
        if self.first_order:
            return "FOMAMLStep"
        return "MAMLStep"


class ReptileStep(StrategyRunner):
    """Reptile meta-learning.

    Each train_step (one meta-step):
    1. Sample tasks from the TaskSampler
    2. For each task:
       a. Snapshot the meta-model parameters
       b. Run K inner SGD steps on support set (standard in-place updates)
       c. Compute parameter displacement: adapted - initial
       d. Restore initial parameters
    3. Average displacements across tasks => meta-gradient direction
    4. Apply meta-gradient to meta-model via the meta-optimizer

    Reptile does NOT need torch.func.functional_call — it uses standard
    in-place SGD for the inner loop, with no computational graph.

    Setup kwargs (via runner.get_strategy_kwargs()):
        inner_lr: float          -- inner loop learning rate (default 0.01)
        inner_steps: int         -- K inner SGD steps (default 5)
        tasks_per_step: int      -- tasks per meta-step (default 4)
    """

    paradigm = TrainingParadigm.NESTED

    def setup(self, *, components, config, device, **kwargs) -> None:
        self.model = components.model
        self.optimizer = components.optimizer
        self.device = device
        self._components = components
        self.task_loss_fn = components.task_loss_fn
        self.task_sampler = components.data

        if self.task_loss_fn is None:
            raise ValueError(
                "ReptileStep requires a task_loss_fn. "
                "Ensure the experiment provides task_loss_fn in MetaLearningComponents."
            )

        # Meta-learning hyperparameters
        self.inner_lr = kwargs.get('inner_lr', 0.01)
        self.inner_steps = kwargs.get('inner_steps', 5)
        self.tasks_per_step = kwargs.get('tasks_per_step', 4)

        # Strategy caches
        self._cached_meta_gradients: dict[str, torch.Tensor] | None = None
        self._cached_inner_gradients: list[dict[str, torch.Tensor]] | None = None
        self._cached_task_losses: list[float] | None = None

    def _inner_loop(self, task_batch):
        """Run K standard SGD steps on the support set.

        No computational graph needed — Reptile computes meta-gradients
        as the direction from initial to adapted parameters.

        Returns:
            (displacement, final_loss)
            displacement: dict of {name: adapted - initial} tensors
            final_loss: loss at the final inner step (for reporting)
        """
        # Snapshot initial parameters
        initial_params = {
            name: p.detach().clone()
            for name, p in self.model.named_parameters()
        }

        # Standard in-place SGD inner loop (no graph needed for meta-gradient)
        final_loss = 0.0
        for k in range(self.inner_steps):
            self.model.zero_grad()
            support_loss = self.task_loss_fn(self.model, task_batch.support)
            support_loss.backward()

            # Manual SGD step (in-place, no graph)
            with torch.no_grad():
                for name, p in self.model.named_parameters():
                    if p.grad is not None:
                        p.sub_(self.inner_lr * p.grad)

            final_loss = float(support_loss.detach())

        # Compute displacement: adapted - initial
        displacement = {
            name: (p.detach() - initial_params[name])
            for name, p in self.model.named_parameters()
        }

        # Restore initial parameters (before next task)
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                p.copy_(initial_params[name])

        return displacement, final_loss

    def train_step(self, step: int, batch: Any = None) -> StepResult:
        """Execute one Reptile meta-step."""
        tasks = self.task_sampler.sample(self.tasks_per_step)

        # Collect per-task displacements
        displacements = []
        task_losses = []

        for task_batch in tasks:
            displacement, final_loss = self._inner_loop(task_batch)
            displacements.append(displacement)
            task_losses.append(final_loss)

        # Average displacements across tasks
        param_names = list(displacements[0].keys())
        avg_displacement = {}
        for name in param_names:
            avg_displacement[name] = torch.stack(
                [d[name] for d in displacements]
            ).mean(dim=0)

        # Apply meta-gradient via optimizer:
        # Set param.grad = -displacement (negative because optimizer does
        # p -= lr * grad, and we want p += lr * displacement)
        self.optimizer.zero_grad()
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if name in avg_displacement:
                    p.grad = -avg_displacement[name]

        self._components.clip_gradients()
        self.optimizer.step()

        # Cache state for hooks
        self._cached_meta_gradients = {
            name: avg_displacement[name].detach()
            for name in param_names
        }
        self._cached_inner_gradients = None
        self._cached_task_losses = task_losses

        mean_loss = sum(task_losses) / len(task_losses)
        return StepResult(
            metrics={
                'loss': mean_loss,
                'mean_task_loss': mean_loss,
                'task_loss_std': (
                    torch.tensor(task_losses).std().item()
                    if len(task_losses) > 1 else 0.0
                ),
                'num_tasks': len(tasks),
                'meta_grad_norm': sum(
                    d.norm().item() ** 2 for d in avg_displacement.values()
                ) ** 0.5,
            },
        )

    @property
    def name(self) -> str:
        return "ReptileStep"
