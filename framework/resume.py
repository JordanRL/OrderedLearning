"""Resume support for checkpoint-based training continuation.

Provides detection of resumable state from output directories,
config loading from saved experiment_config.json, and checkpoint
restoration of model, optimizer, scheduler, and RNG states.
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
from dataclasses import dataclass


@dataclass
class ResumeInfo:
    """Information needed to resume training from a checkpoint.

    Attributes:
        checkpoint_path: Path to the .pt checkpoint file, or None if the
            next strategy should start fresh.
        start_step_or_epoch: Training resumes from this + 1.
        completed_strategies: Strategy names to skip entirely.
        config_path: Path to experiment_config.json used for config loading.
    """
    checkpoint_path: str | None
    start_step_or_epoch: int
    completed_strategies: list[str]
    config_path: str


# Flags that are safe to use alongside --resume. Everything else is
# rejected automatically — no experiment-specific knowledge needed.
RESUME_SAFE_FLAGS = frozenset({
    # Resume / identification
    '--resume', '--strategy', '--output-dir', '--target',
    # Display / runtime
    '--live', '--silent', '--no-console-output', '--no-determinism',
    # Checkpointing
    '--save-checkpoints', '--validate-checkpoints',
    # Hook flags
    '--hooks', '--with-hooks', '--hook-csv', '--hook-jsonl',
    '--hook-wandb', '--hook-config', '--profile-hooks',
    '--hooks-list', '--hooks-describe', '--hook-offload-state',
    # Help
    '--help', '-h',
})


def check_resume_conflicts(argv: list[str]) -> list[str]:
    """Return any --flags in argv that are not resume-safe.

    Positional args and flag values are ignored — only tokens
    starting with '--' (or '-h') are checked.
    """
    conflicts = []
    for token in argv:
        if token.startswith('--') and token not in RESUME_SAFE_FLAGS:
            # Strip =value from --flag=value forms
            flag = token.split('=', 1)[0]
            if flag not in RESUME_SAFE_FLAGS:
                conflicts.append(flag)
    return conflicts


def find_latest_checkpoint(experiment_dir: str) -> tuple[str, int] | None:
    """Find the most recent checkpoint in an experiment directory.

    Globs checkpoints/checkpoint_*.pt, extracts the step/epoch number
    from the filename, and returns (path, number) for the highest.
    Returns None if no checkpoints exist.
    """
    pattern = os.path.join(experiment_dir, 'checkpoints', 'checkpoint_*.pt')
    files = glob.glob(pattern)
    if not files:
        return None

    best_path = None
    best_num = -1
    for path in files:
        basename = os.path.basename(path)
        match = re.search(r'checkpoint_(\d+)\.pt$', basename)
        if match:
            num = int(match.group(1))
            if num > best_num:
                best_num = num
                best_path = path

    if best_path is None:
        return None
    return (best_path, best_num)


def detect_resume_state(experiment_name: str, strategy: str,
                        output_dir: str, runner_cls) -> ResumeInfo:
    """Detect resumable state from the output directory structure.

    For a single strategy: looks for the strategy's output dir,
    checks for completion (summary.json) or an in-progress checkpoint.

    For 'all' strategies: walks the canonical strategy order, classifying
    each as complete, in-progress, or not-started.
    """
    from console import OLConsole
    console = OLConsole()

    if strategy != 'all':
        return _detect_single_strategy(experiment_name, strategy, output_dir)

    return _detect_all_strategies(experiment_name, output_dir, runner_cls,
                                  console)


def _detect_single_strategy(experiment_name, strategy, output_dir):
    """Detect resume state for a single named strategy."""
    from console import OLConsole
    console = OLConsole()

    strategy_dir = os.path.join(output_dir, experiment_name, strategy)

    if not os.path.isdir(strategy_dir):
        console.print_error(
            f"No experiment data found to resume from: {strategy_dir}"
        )
        sys.exit(1)

    config_path = os.path.join(strategy_dir, 'experiment_config.json')
    if not os.path.exists(config_path):
        console.print_error(
            f"No experiment config found to resume from: {config_path}"
        )
        sys.exit(1)

    # Already complete?
    summary_path = os.path.join(strategy_dir, 'summary.json')
    if os.path.exists(summary_path):
        console.print_error(
            f"Strategy '{strategy}' is already complete "
            f"(summary.json exists in {strategy_dir})"
        )
        sys.exit(1)

    # Find checkpoint
    result = find_latest_checkpoint(strategy_dir)
    if result is None:
        console.print_error(
            f"No checkpoints found in {strategy_dir}/checkpoints/ — "
            "cannot resume without a checkpoint"
        )
        sys.exit(1)

    checkpoint_path, step_or_epoch = result
    return ResumeInfo(
        checkpoint_path=checkpoint_path,
        start_step_or_epoch=step_or_epoch,
        completed_strategies=[],
        config_path=config_path,
    )


def _detect_all_strategies(experiment_name, output_dir, runner_cls, console):
    """Detect resume state when strategy='all'."""

    # Find any existing experiment_config.json to load config for strategy list
    config_path = _find_any_config(experiment_name, output_dir)
    if config_path is None:
        console.print_error(
            f"No experiment data found to resume from in {output_dir}/"
        )
        sys.exit(1)

    # Load config to get canonical strategy order
    config = load_config_from_output(config_path, runner_cls)
    config.experiment_name = experiment_name
    config.strategy = 'all'
    temp_runner = runner_cls(config=config)
    canonical_strategies = temp_runner.get_strategies()

    completed = []
    resume_target = None

    for strat in canonical_strategies:
        strat_dir = os.path.join(output_dir, experiment_name, strat)

        if not os.path.isdir(strat_dir):
            # Not started — stop scanning
            break

        summary_path = os.path.join(strat_dir, 'summary.json')
        if os.path.exists(summary_path):
            # Complete
            completed.append(strat)
            continue

        # Has checkpoints but no summary — in progress
        result = find_latest_checkpoint(strat_dir)
        if result is not None:
            checkpoint_path, step_or_epoch = result
            resume_target = (strat, checkpoint_path, step_or_epoch)
        # Either way, stop scanning (rest are not started)
        break

    # Edge: all complete
    if len(completed) == len(canonical_strategies):
        console.print(
            "[panel.success]All strategies are already complete.[/panel.success]"
        )
        sys.exit(0)

    # Edge: nothing at all
    if not completed and resume_target is None:
        console.print_error(
            f"No experiment data found to resume from in {output_dir}/"
        )
        sys.exit(1)

    # Build ResumeInfo
    if resume_target is not None:
        _, checkpoint_path, step_or_epoch = resume_target
        return ResumeInfo(
            checkpoint_path=checkpoint_path,
            start_step_or_epoch=step_or_epoch,
            completed_strategies=completed,
            config_path=config_path,
        )
    else:
        # All completed so far, next one starts fresh
        return ResumeInfo(
            checkpoint_path=None,
            start_step_or_epoch=0,
            completed_strategies=completed,
            config_path=config_path,
        )


def _find_any_config(experiment_name, output_dir):
    """Find any experiment_config.json for the given experiment name."""
    pattern = os.path.join(output_dir, experiment_name, '*',
                           'experiment_config.json')
    configs = glob.glob(pattern)
    if configs:
        return sorted(configs)[0]
    return None


def load_config_from_output(config_path: str, runner_cls) -> 'BaseConfig':
    """Load experiment config from a saved experiment_config.json.

    Reads the JSON, gets the config class from runner_cls.config_class,
    and constructs the dataclass filtering to only known fields.
    """
    with open(config_path, 'r') as f:
        data = json.load(f)

    config_cls = runner_cls.config_class
    valid_fields = set(config_cls.__dataclass_fields__.keys())
    filtered = {k: v for k, v in data.items() if k in valid_fields}
    return config_cls(**filtered)


def load_checkpoint(path: str, model, optimizer, scheduler, runner):
    """Restore training state from a checkpoint file.

    Loads model, optimizer, and scheduler state dicts, restores RNG
    states if present, and calls runner.load_training_state() for
    experiment-specific state.

    Security note: uses weights_only=False because checkpoints contain
    optimizer state dicts, RNG states, and experiment-specific training
    state that require unpickling. Only load checkpoints you have
    generated yourself or trust completely.
    """
    import torch

    checkpoint = torch.load(path, map_location='cpu', weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    if 'rng_states' in checkpoint:
        _set_rng_states(checkpoint['rng_states'])

    if checkpoint.get('training_state') is not None:
        runner.load_training_state(checkpoint['training_state'])


def _set_rng_states(states: dict):
    """Restore RNG states for all backends (inverse of _get_rng_states)."""
    import random
    import numpy as np
    import torch

    torch.random.set_rng_state(states['torch'])
    random.setstate(states['python'])
    np.random.set_state(states['numpy'])
    if 'torch_cuda' in states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states['torch_cuda'])
