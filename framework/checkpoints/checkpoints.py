"""Checkpoint utilities for the training framework.

Provides save_checkpoint, validate_checkpoint, and EmergencyCheckpoint,
used by the Trainer classes in framework.trainers.
"""

from __future__ import annotations

from ..utils import get_environment_info


def _get_rng_states() -> dict:
    """Capture current RNG states for all relevant backends."""
    import random
    import numpy as np
    import torch

    states = {
        'torch': torch.random.get_rng_state(),
        'python': random.getstate(),
        'numpy': np.random.get_state(),
    }
    if torch.cuda.is_available():
        states['torch_cuda'] = torch.cuda.get_rng_state_all()
    return states


def save_checkpoint(experiment_dir, components, step_or_epoch,
                    training_state=None) -> None:
    """Save training checkpoint (includes environment metadata and RNG states)."""
    import os
    import torch
    ckpt_dir = os.path.join(experiment_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f'checkpoint_{step_or_epoch}.pt')
    torch.save({
        'step': step_or_epoch,
        'components': components.state_dict(),
        'rng_states': _get_rng_states(),
        'training_state': training_state,
        'environment': get_environment_info(),
    }, path)


def _compare_state(live, saved) -> bool:
    """Recursively compare two state structures for bit-identity.

    Handles tensors (torch.equal), numpy arrays, dicts, lists/tuples,
    and scalars. Returns True if every value matches exactly.
    """
    import numpy as np
    import torch

    if type(live) is not type(saved):
        return False
    if isinstance(live, torch.Tensor):
        if live.shape != saved.shape or live.dtype != saved.dtype:
            return False
        return torch.equal(live.cpu(), saved.cpu())
    if isinstance(live, np.ndarray):
        return np.array_equal(live, saved)
    if isinstance(live, dict):
        if live.keys() != saved.keys():
            return False
        return all(_compare_state(live[k], saved[k]) for k in live)
    if isinstance(live, (list, tuple)):
        if len(live) != len(saved):
            return False
        return all(_compare_state(a, b) for a, b in zip(live, saved))
    return live == saved


def validate_checkpoint(experiment_dir, components, step_or_epoch,
                        training_state=None) -> None:
    """Compare current training state against a saved checkpoint."""
    import os
    import torch
    from console import OLConsole

    console = OLConsole()
    path = os.path.join(experiment_dir, 'checkpoints', f'checkpoint_{step_or_epoch}.pt')

    if not os.path.exists(path):
        console.print_warning(f"No checkpoint found at step/epoch {step_or_epoch}")
        return

    checkpoint = torch.load(path, map_location='cpu', weights_only=False)

    current_state = components.state_dict()
    saved_state = checkpoint['components']

    components_ok = _compare_state(current_state, saved_state)

    rng_ok = True
    has_rng = 'rng_states' in checkpoint
    if has_rng:
        rng_ok = _compare_state(_get_rng_states(), checkpoint['rng_states'])

    state_ok = True
    has_state = checkpoint.get('training_state') is not None
    if has_state and training_state is not None:
        state_ok = _compare_state(training_state, checkpoint['training_state'])

    all_ok = components_ok and rng_ok and state_ok

    def _mark(ok):
        return '[metric.improved]OK[/metric.improved]' if ok else '[metric.degraded]MISMATCH[/metric.degraded]'

    rng_display = f"  rng {_mark(rng_ok)}" if has_rng else ""
    state_display = f"  state {_mark(state_ok)}" if has_state else ""
    label = step_or_epoch
    if all_ok:
        console.print_complete(
            f"Checkpoint {label}: [metric.improved]bit-identical[/metric.improved]"
            f"  (components {_mark(components_ok)}{rng_display}{state_display})"
        )
    else:
        console.print_error(
            f"Checkpoint {label}: non-identical"
            f"  (components {_mark(components_ok)}{rng_display}{state_display})"
        )


class EmergencyCheckpoint:
    """Holds in-memory snapshot at a known-good training boundary.

    Loops call capture() at safe restart points (epoch start for epoch_loop,
    periodic intervals for step_loop). Emergency handlers call save() to
    write the snapshot to disk.

    Automatically installs a SIGTERM handler so that pod preemption,
    ``docker stop``, and job scheduler termination trigger a clean
    checkpoint save before exit.
    """

    def __init__(self, experiment_dir, hook_manager=None):
        self._experiment_dir = experiment_dir
        self._hook_manager = hook_manager
        self._state = None
        self._step_or_epoch = None
        self._install_signal_handler()

    def _install_signal_handler(self):
        """Register SIGTERM handler to save emergency checkpoint on termination."""
        import signal
        signal.signal(signal.SIGTERM, self._handle_sigterm)

    def _handle_sigterm(self, signum, frame):
        """SIGTERM handler — save checkpoint and exit cleanly."""
        import signal
        # Reset to default so a second SIGTERM terminates immediately
        signal.signal(signal.SIGTERM, signal.SIG_DFL)

        from console import OLConsole
        console = OLConsole()
        console.print_warning(
            f"SIGTERM received. "
            f"Saving emergency checkpoint from {self._step_or_epoch}..."
        )
        path = self.save()
        if path:
            console.print_complete(f"Emergency checkpoint saved: {path}")
        raise SystemExit(0)

    def capture(self, components, step_or_epoch, runner=None):
        """Snapshot current training state to CPU memory."""
        self._step_or_epoch = step_or_epoch
        self._state = {
            'components': components.capture_for_emergency(),
            'rng_states': _get_rng_states(),
            'training_state': runner.save_training_state() if runner else None,
            'environment': get_environment_info(),
        }

    def save(self):
        """Write captured snapshot to disk and flush sinks. Returns path or None."""
        import os
        import torch

        if self._state is None:
            return None
        ckpt_dir = os.path.join(self._experiment_dir, 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        path = os.path.join(ckpt_dir, f'emergency_{self._step_or_epoch}.pt')
        torch.save({'step': self._step_or_epoch, **self._state}, path)
        if self._hook_manager:
            self._hook_manager.flush_sinks()
        return path

    @property
    def step_or_epoch(self):
        return self._step_or_epoch
