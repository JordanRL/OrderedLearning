"""BackpropInterventionContext: mutable SAPI for intervention hooks in backprop training."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from console import OLConsole
from console.utils import apply_style
from .profiler import CheckpointProfiler
from ..utils import create_accumulator, accumulate, finalize


class BackpropInterventionContext:
    """Mutable SAPI for intervention hooks in single-model backprop training.

    Provides controlled operations for hooks that need to modify training
    state, such as saving/restoring checkpoints and running extra training
    epochs. Hooks should use these methods instead of directly manipulating
    the model, optimizer, or scheduler.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler,
        criterion: nn.Module,
        loader,
        config,
        device: torch.device,
        pre_epoch_state: dict | None = None,
        current_batch=None,
        profiler: CheckpointProfiler | None = None,
        loss_fn: callable | None = None,
        get_shuffled_loader_fn: callable | None = None,
        grad_scaler=None,
    ):
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._criterion = criterion
        self._loader = loader
        self._config = config
        self._device = device
        self._pre_epoch_state = pre_epoch_state
        self._current_batch = current_batch
        self._checkpoints: dict[str, dict] = {}
        self._next_token_id = 0
        self.profiler = profiler or CheckpointProfiler(enabled=False)
        self._loss_fn = loss_fn
        self._get_shuffled_loader_fn = get_shuffled_loader_fn
        self._grad_scaler = grad_scaler

    @property
    def model(self) -> nn.Module:
        """The model (read-only access)."""
        return self._model

    @property
    def device(self) -> torch.device:
        """The device the model is on."""
        return self._device

    def compute_batch_gradients(
        self, batch=None,
    ) -> dict[str, torch.Tensor]:
        """Forward-backward on a single batch, returning per-parameter gradients.

        Does NOT step the optimizer. Saves and restores ``param.grad`` internally
        so the caller's gradient state is not corrupted.

        Args:
            batch: Input batch. Falls back to ``self._current_batch`` when None.
                   Passed to ``loss_fn(model, batch)``.

        Returns:
            Dict of parameter_name -> gradient tensor (on device).

        Raises:
            RuntimeError: If no batch is available and none was passed.
            RuntimeError: If no loss_fn was provided.
        """
        if batch is None:
            batch = self._current_batch
        if batch is None:
            raise RuntimeError(
                "compute_batch_gradients() requires a batch — either pass one "
                "or construct BackpropInterventionContext with current_batch."
            )
        if self._loss_fn is None:
            raise RuntimeError(
                "compute_batch_gradients() requires a loss_fn. "
                "Ensure the experiment's get_loss_fn() returns a callable."
            )

        p = self.profiler

        # Save existing param.grad so we can restore it after
        with p.section('cbg_save_grads'):
            saved_grads: dict[str, torch.Tensor | None] = {}
            for name, param in self._model.named_parameters():
                saved_grads[name] = param.grad.detach().clone() if param.grad is not None else None

        # Forward-backward
        with p.section('cbg_forward_backward'):
            self._model.train()
            self._optimizer.zero_grad()
            loss = self._loss_fn(self._model, batch)
            loss.backward()

        # Collect gradients
        with p.section('cbg_collect_grads'):
            grads: dict[str, torch.Tensor] = {}
            for name, param in self._model.named_parameters():
                if param.grad is not None:
                    grads[name] = param.grad.detach().clone()

        # Restore original param.grad
        with p.section('cbg_restore_grads'):
            for name, param in self._model.named_parameters():
                param.grad = saved_grads[name]

        return grads

    def restore_pre_epoch(self):
        """Restore model and optimizer to the state before the training epoch.

        Only available when a hook includes HookNeeds.PRE_EPOCH_STATE in its
        needs, which causes the training loop to save the pre-epoch checkpoint.

        Raises:
            RuntimeError: If no pre-epoch state was saved.
        """
        if self._pre_epoch_state is None:
            raise RuntimeError(
                "No pre-epoch state available. Include HookNeeds.PRE_EPOCH_STATE "
                "in the hook's needs to have the training loop save it."
            )
        self._model.load_state_dict(
            {k: v.to(self._device) for k, v in self._pre_epoch_state['model'].items()}
        )
        self._optimizer.load_state_dict(self._pre_epoch_state['optimizer'])
        if 'rng_cpu' in self._pre_epoch_state:
            torch.random.set_rng_state(self._pre_epoch_state['rng_cpu'])
            if self._pre_epoch_state.get('rng_cuda') is not None:
                torch.cuda.set_rng_state(self._pre_epoch_state['rng_cuda'], self._device)

    def apply_perturbation(self, direction: dict[str, torch.Tensor], scale: float):
        """Perturb model weights: theta <- theta + scale * direction.

        Used for finite-difference approximations (e.g., Hessian-vector products).
        The caller is responsible for saving/restoring state around this.

        Args:
            direction: Dict of parameter_name -> perturbation tensor.
            scale: Scalar multiplier for the perturbation.
        """
        with torch.no_grad():
            for name, param in self._model.named_parameters():
                if name in direction:
                    param.add_(direction[name], alpha=scale)

    def compute_gradients(self) -> dict[str, torch.Tensor]:
        """Run forward-backward over the training data without stepping the optimizer.

        Equivalent to run_training_epoch(self._loader, step=False).

        Returns:
            Dict of parameter_name -> mean gradient tensor (on GPU).
        """
        return self.run_training_epoch(self._loader, step=False)

    def save_checkpoint(self, full: bool = True) -> str:
        """Save training state for later restoration.

        Args:
            full: If True, saves everything to CPU (model, optimizer, scheduler,
                  grads). Use when the hook calls optimizer.step() between save
                  and restore.  If False, saves only model params and RNG state
                  on GPU — much faster for hooks that only need forward/backward.

        Returns:
            Opaque token to pass to restore_checkpoint().
        """
        p = self.profiler
        token = f"ckpt_{self._next_token_id}"
        self._next_token_id += 1

        with p.section('rng_save'):
            rng_cpu = torch.random.get_rng_state()
            rng_cuda = torch.cuda.get_rng_state(self._device) if self._device.type == 'cuda' else None

        if full:
            with p.section('full_model_save'):
                model_state = {k: v.cpu().clone() for k, v in self._model.state_dict().items()}
            with p.section('full_optimizer_save'):
                opt_state = copy.deepcopy(self._optimizer.state_dict())
            with p.section('full_scheduler_save'):
                sched_state = self._scheduler.state_dict()
            with p.section('full_grads_save'):
                grads = {
                    name: param.grad.detach().cpu().clone() if param.grad is not None else None
                    for name, param in self._model.named_parameters()
                }
            self._checkpoints[token] = {
                'full': True,
                'model': model_state,
                'optimizer': opt_state,
                'scheduler': sched_state,
                'grads': grads,
                'rng_cpu': rng_cpu,
                'rng_cuda': rng_cuda,
            }
        else:
            with p.section('light_params_save'):
                params = {
                    name: param.data.clone()
                    for name, param in self._model.named_parameters()
                }
            self._checkpoints[token] = {
                'full': False,
                'params': params,
                'rng_cpu': rng_cpu,
                'rng_cuda': rng_cuda,
            }

        return token

    def restore_checkpoint(self, token: str):
        """Restore training state from a saved checkpoint.

        Args:
            token: Token returned by save_checkpoint().
        """
        if token not in self._checkpoints:
            raise ValueError(f"Unknown checkpoint token: '{token}'")
        cp = self._checkpoints[token]
        p = self.profiler

        if cp['full']:
            with p.section('full_model_restore'):
                self._model.load_state_dict(
                    {k: v.to(self._device) for k, v in cp['model'].items()}
                )
            with p.section('full_optimizer_restore'):
                self._optimizer.load_state_dict(cp['optimizer'])
            with p.section('full_scheduler_restore'):
                self._scheduler.load_state_dict(cp['scheduler'])
            with p.section('full_grads_restore'):
                saved_grads = cp['grads']
                for name, param in self._model.named_parameters():
                    if name in saved_grads:
                        g = saved_grads[name]
                        param.grad = g.to(self._device) if g is not None else None
        else:
            with p.section('light_params_restore'):
                saved_params = cp['params']
                for name, param in self._model.named_parameters():
                    param.data.copy_(saved_params[name])

        with p.section('rng_restore'):
            torch.random.set_rng_state(cp['rng_cpu'])
            if cp['rng_cuda'] is not None:
                torch.cuda.set_rng_state(cp['rng_cuda'], self._device)

    def discard_checkpoint(self, token: str):
        """Free memory for a checkpoint that is no longer needed.

        Args:
            token: Token returned by save_checkpoint().
        """
        self._checkpoints.pop(token, None)

    _TASK_BATCHES = "_hook_batches"

    def run_training_epoch(
        self,
        loader,
        step: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Run one training epoch and return accumulated gradients.

        Args:
            loader: Data loader/iterator to train on.
            step: If True, call optimizer.step() after each batch
                  (normal training). If False, only compute gradients
                  without updating weights.

        Returns:
            Dict of parameter_name -> mean gradient tensor (on GPU).

        Raises:
            RuntimeError: If no loss_fn was provided.
        """
        if self._loss_fn is None:
            raise RuntimeError(
                "run_training_epoch() requires a loss_fn. "
                "Ensure the experiment's get_loss_fn() returns a callable."
            )

        console = OLConsole()
        try:
            loader_len = len(loader)
        except TypeError:
            loader_len = None
        console.create_progress_task(
            self._TASK_BATCHES,
            apply_style("Hook batches", "status"),
            total=loader_len,
        )

        accum, count = create_accumulator(self._model)
        self._model.train()

        for batch in loader:
            # Handle both SparseModularDataset (returns tensor directly)
            # and TensorDataset (returns tuple of tensors)
            if isinstance(batch, (list, tuple)):
                batch = batch[0]

            self._optimizer.zero_grad()
            loss = self._loss_fn(self._model, batch)
            loss.backward()

            count = accumulate(accum, self._model, count)

            if step:
                self._optimizer.step()

            console.update_progress_task(self._TASK_BATCHES, advance=1)

        console.remove_progress_task(self._TASK_BATCHES)
        return finalize(accum, count, to_cpu=False)

    def get_shuffled_loader(self):
        """Create an iterator with randomly permuted training data.

        Delegates to the experiment runner's get_shuffled_loader_fn callback,
        since the runner knows its data format. Falls back to error if no
        callback was provided.

        Returns:
            New iterator with the same data in a random order.
        """
        if self._get_shuffled_loader_fn is not None:
            return self._get_shuffled_loader_fn(self._loader, self._config)
        raise RuntimeError(
            "get_shuffled_loader() requires a get_shuffled_loader_fn callback. "
            "The experiment runner should provide this via "
            "build_intervention_context_kwargs()."
        )
