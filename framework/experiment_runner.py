"""Experiment runner hierarchy.

ExperimentRunner is the ABC. It declares loop_type and provides lifecycle
methods that the loop functions (step_loop, epoch_loop) call. Experiments
never write the training loop — they implement building blocks.

Task-specific base classes:
- LMRunner: GPT-2 language model experiments (step-based)
- GrokkingRunner: synthetic grokking experiments (epoch-based)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Any
import json
import os

import torch
import torch.nn as nn
import torch.optim as optim

from console import OLConsole
from .config import BaseConfig
from .eval_result import EvalResult
from .strategy_runner import StrategyRunner, StepResult
from .utils import (
    set_seeds, set_determinism, snapshot_params,
    get_environment_info, check_environment_compatibility,
)
from .models import MODEL_CONFIGS, get_lr_scheduler
from .cli import add_eval_target_args
from . import display


class ExperimentRunner(ABC):
    """Base class for all experiments.

    Never writes the training loop. Declares loop_type to select which
    loop function runs it. Provides lifecycle methods that the loop calls.
    """

    config_class = BaseConfig
    loop_type: str = 'step'   # 'step' or 'epoch' — framework dispatches
    hook_sets: dict[str, list[str]] = {
        'none': [],
        'minimal': [],
        'observers': [],
        'interventions': [],
        'full': [],
    }
    live_metrics: dict[str, dict[str, str]] = {}  # group -> {label: metric_key}
    arg_aliases: dict[str, str] = {}  # config field name -> argparse dest name

    def __init__(self, config: BaseConfig, **kwargs):
        self.config = config
        self.console = OLConsole()
        self.device = self._select_device()

    @staticmethod
    def _select_device() -> torch.device:
        """Select best available device: CUDA > MPS > CPU."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    # === CLI / Factory classmethods ===

    @classmethod
    def add_args(cls, parser):
        """Add experiment-specific CLI args to parser. Override in subclasses."""
        pass

    @classmethod
    def build_config(cls, args):
        """Build config dataclass from parsed CLI args. Must override."""
        raise NotImplementedError(
            f"{cls.__name__} must implement build_config()"
        )

    @classmethod
    def build_runner(cls, config, args):
        """Factory to create runner instance from config and parsed args."""
        return cls(config=config)

    # === Strategy enumeration ===

    @abstractmethod
    def get_strategies(self) -> list[str]:
        """Return list of strategy names to iterate over."""
        ...

    # === Component creation ===

    @abstractmethod
    def create_model(self) -> nn.Module:
        """Create and return a model on self.device."""
        ...

    @abstractmethod
    def create_strategy(self, strategy_name: str) -> StrategyRunner:
        """Create the StrategyRunner for this strategy name."""
        ...

    @abstractmethod
    def create_data(self, strategy_name: str) -> Any:
        """Create the data source for this strategy."""
        ...

    def create_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Create optimizer. Override for custom setup."""
        return optim.AdamW(
            model.parameters(),
            lr=getattr(self.config, 'lr', 1e-4),
            weight_decay=getattr(self.config, 'weight_decay', 0.01),
        )

    def create_scheduler(self, optimizer: optim.Optimizer, total_steps: int):
        """Create LR scheduler. Override for custom setup."""
        warmup = getattr(self.config, 'warmup_steps', 0)
        return get_lr_scheduler(optimizer, warmup, total_steps)

    def create_criterion(self) -> nn.Module | None:
        """Create loss function. Override for epoch-based experiments."""
        return None

    def get_loss_fn(self, criterion) -> callable | None:
        """Return a loss function: (model, batch) -> loss_scalar.

        Used by ModelDataContext for intervention hooks that need to compute
        gradients on arbitrary batch formats. Returns None to use legacy
        hardcoded behavior (mod-arithmetic [a, b, result] format).
        """
        return None

    # === Configuration ===

    def get_total_steps(self) -> int:
        """Total training steps (step-based experiments)."""
        return getattr(self.config, 'steps', 1000)

    def get_total_epochs(self) -> int:
        """Total training epochs (epoch-based experiments)."""
        return getattr(self.config, 'epochs', 100)

    # === Lifecycle callbacks ===

    def setup_condition(self, strategy_name: str):
        """Called before each strategy run. Seeds, prep, etc."""
        set_seeds(self.config.seed)
        set_determinism(enabled=not self.config.no_determinism)

    def teardown_condition(self, strategy_name: str):
        """Called after each strategy run."""
        pass

    def wire_hooks(self, strategy_name: str, strategy, hook_manager):
        """Wire hook-to-strategy dependencies after setup.

        Called by the training loop after strategy creation and hook_manager
        reset. Override in subclasses that need to connect hooks to strategies
        (e.g., adaptive-stride wiring the transition callback).
        """
        pass

    def save_training_state(self) -> dict | None:
        """Return experiment-specific state to include in checkpoints.

        Called by the training loop before saving a checkpoint. Override to
        save custom state (e.g., curriculum phase, data iterator position,
        strategy-specific counters) that the framework doesn't know about.

        Returns a dict of serializable state, or None if nothing to save.
        """
        return None

    def load_training_state(self, state: dict) -> None:
        """Restore experiment-specific state from a checkpoint.

        Called by the training loop on resume. Receives the dict that was
        returned by save_training_state() when the checkpoint was created.
        """
        pass

    def get_epoch_loader(self, data: Any, epoch: int):
        """Return the DataLoader for this epoch. Override for alternating loaders."""
        return data

    def get_strategy_kwargs(self, strategy_name: str, model, optimizer, data) -> dict:
        """Return extra kwargs for strategy.setup().

        Override to provide tokenizer, criterion, selector, curriculum, etc.
        """
        return {}

    # === Evaluation ===

    def test_validate(self, model: nn.Module, step_or_epoch: int) -> EvalResult | None:
        """Basic validation measure (test accuracy, test loss, target probabilities).

        Override in subclasses to provide test-set evaluation.
        """
        return None

    def train_validate(self, model: nn.Module, step_or_epoch: int) -> EvalResult | None:
        """Training process validation (train accuracy, phase metrics).

        Override in subclasses to provide training-side evaluation.
        """
        return None

    def evaluate(self, model: nn.Module, step_or_epoch: int) -> EvalResult | None:
        """Evaluate model. Called at eval_every intervals and at start/end.

        Composes test_validate() and train_validate(). Subclasses should
        override those methods rather than this one.
        """
        test = self.test_validate(model, step_or_epoch)
        train = self.train_validate(model, step_or_epoch)
        return EvalResult.merge(test, train)

    def should_stop(self, step_or_epoch: int, eval_result: EvalResult | None) -> bool:
        """Check for early stopping. Default: defer to eval_result."""
        if eval_result is not None and eval_result.should_stop:
            return True
        return False

    # === Display ===

    def display_banner(self):
        """Display experiment-level header. Called once at start."""
        pass

    def display_condition_start(self, strategy_name: str):
        """Display condition header. Called once per strategy."""
        display.display_condition_header(strategy_name)

    def display_eval(self, step_or_epoch: int, eval_result: EvalResult,
                     strategy_name: str):
        """Display periodic evaluation results."""
        display.display_eval_update(
            step_or_epoch, eval_result,
            context_label=strategy_name,
        )

    def display_post_step(self, step: int, post_info: dict):
        """Display strategy post-step info (e.g. phase transition)."""
        if post_info and post_info.get('phase_transition'):
            display.display_phase_transition(
                post_info['old_phase'],
                post_info['new_phase'],
            )

    def display_final(self, strategy_name: str, init_eval: EvalResult,
                      final_eval: EvalResult):
        """Display final results for one strategy."""
        display.display_final_results(
            strategy_name, init_eval, final_eval,
        )

    def display_comparison(self, all_results: dict):
        """Display cross-strategy comparison. Called once at end."""
        if len(all_results) > 1:
            display.display_comparison_table(all_results)

    # === Results ===

    def build_summary(self, strategy_name: str, init_eval: EvalResult,
                      final_eval: EvalResult, total: int, **context) -> dict:
        """Build summary dict for one strategy run.

        The training loop passes extra context via keyword arguments:
            model: the trained model (for parameter count)
            planned_total: planned steps/epochs before early stopping
            duration: wall clock seconds for the training loop
            early_stopped: whether training ended before planned_total
            global_step: total gradient steps (epoch loop only)
        """
        model = context.get('model')
        planned_total = context.get('planned_total')
        duration = context.get('duration')
        early_stopped = context.get('early_stopped', False)

        summary = {
            'experiment': self.config.experiment_name or '',
            'strategy': strategy_name,
        }

        # --- Timing ---
        if duration is not None:
            timing = {'duration_seconds': round(duration, 2)}
            if duration > 0:
                if self.loop_type == 'epoch':
                    global_step = context.get('global_step', 0)
                    if global_step > 0:
                        timing['steps_per_second'] = round(global_step / duration, 2)
                else:
                    timing['steps_per_second'] = round(total / duration, 2)
            summary['timing'] = timing

        # --- Model ---
        if model is not None:
            raw = model._orig_mod if hasattr(model, '_orig_mod') else model
            param_count = sum(p.numel() for p in raw.parameters())
            model_info = {'parameters': param_count}
            if hasattr(self, 'model_config'):
                model_info['architecture'] = dict(self.model_config)
            summary['model'] = model_info

        # --- Training ---
        counter = 'epochs' if self.loop_type == 'epoch' else 'steps'
        training = {}
        if planned_total is not None:
            training[f'planned_{counter}'] = planned_total
        training[f'actual_{counter}'] = total
        training['early_stopped'] = early_stopped
        if self.loop_type == 'epoch' and context.get('global_step') is not None:
            training['total_gradient_steps'] = context['global_step']
        for key in ('seed', 'lr', 'batch_size', 'weight_decay'):
            val = getattr(self.config, key, None)
            if val is not None:
                training[key] = val
        training['device'] = str(self.device)
        summary['training'] = training

        # --- Evaluation ---
        init_metrics = init_eval.metrics if init_eval else None
        final_metrics = final_eval.metrics if final_eval else None
        summary['init_eval'] = init_metrics
        summary['final_eval'] = final_metrics

        # --- Deltas ---
        if init_metrics and final_metrics:
            deltas = {}
            for key in final_metrics:
                if key in init_metrics:
                    init_val = init_metrics[key]
                    final_val = final_metrics[key]
                    delta = {
                        'init': init_val,
                        'final': final_val,
                        'change': round(final_val - init_val, 6),
                    }
                    if init_val != 0:
                        delta['ratio'] = round(final_val / init_val, 4)
                    deltas[key] = delta
            if deltas:
                summary['deltas'] = deltas

        return summary

    # === Output management ===

    def prepare_output_dir(self, strategy_name: str = None) -> str:
        """Create and return the output directory."""
        name = self.config.experiment_name or "experiment"
        if strategy_name:
            experiment_dir = os.path.join(self.config.output_dir, name, strategy_name)
        else:
            experiment_dir = os.path.join(self.config.output_dir, name)
        os.makedirs(experiment_dir, exist_ok=True)

        # Check for pre-existing run data and environment compatibility
        summary_path = os.path.join(experiment_dir, 'summary.json')
        config_path = os.path.join(experiment_dir, 'experiment_config.json')
        if os.path.exists(config_path):
            self._check_environment_compatibility(config_path)

        if os.path.exists(summary_path):
            self.console.print_warning(
                f"Output directory contains a completed previous run (summary.json exists). "
                f"Files in {experiment_dir} will be overwritten."
            )
        elif os.path.exists(config_path):
            self.console.print_warning(
                f"Output directory contains data from an incomplete previous run. "
                f"Files in {experiment_dir} will be overwritten. "
                f"Use --resume to continue from a checkpoint instead."
            )

        return experiment_dir

    def _check_environment_compatibility(self, config_path: str):
        """Check current environment against a saved experiment_config.json.

        If the saved config contains an 'environment' block and the current
        environment differs in reproducibility-relevant ways, prints an error
        with details and exits before any data is written to disk.
        """
        import sys

        try:
            with open(config_path, 'r') as f:
                saved_config = json.load(f)
        except (json.JSONDecodeError, OSError):
            return  # Can't read config — skip check

        saved_env = saved_config.get('environment')
        if not saved_env:
            return  # No environment block — skip check

        warnings = check_environment_compatibility(saved_env)
        if not warnings:
            return

        self.console.print()
        self.console.print_error(
            f"Environment mismatch with existing data in {config_path}"
        )
        self.console.print()
        self.console.print("[bold]Differences:[/bold]")
        for w in warnings:
            self.console.print(f"  [metric.degraded]•[/metric.degraded] {w}")
        self.console.print()
        self.console.print(
            "[label]To proceed, either:[/label]\n"
            f"  1. Delete [path]{config_path}[/path] to accept the new environment\n"
            "  2. Update the current environment to match the existing data"
        )
        self.console.print()
        sys.exit(1)

    def save_config(self, experiment_dir: str, extra: dict = None):
        """Save experiment configuration to JSON (includes environment info)."""
        config_path = os.path.join(experiment_dir, 'experiment_config.json')
        config_dict = asdict(self.config) if hasattr(self.config, '__dataclass_fields__') else {}
        if extra:
            config_dict.update(extra)
        config_dict['environment'] = get_environment_info()
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        return config_path

    def save_summary(self, experiment_dir: str, summary: dict):
        """Save summary.json."""
        summary_path = os.path.join(experiment_dir, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        self.console.print(f"[label]Summary saved to:[/label] [path]{summary_path}[/path]")
        return summary_path

    def save_trajectory(self, experiment_dir: str, trajectory: list | None):
        """Save traj.pt if trajectory recording was enabled."""
        if trajectory is not None and trajectory:
            traj_path = os.path.join(experiment_dir, 'traj.pt')
            torch.save(trajectory, traj_path)
            self.console.print(
                f"[label]Trajectory ({len(trajectory)} snapshots) saved to:[/label] "
                f"[path]{traj_path}[/path]"
            )
            return traj_path
        return None

    def save_final_model(self, experiment_dir: str, model: nn.Module, strategy_name: str):
        """Save final model weights.

        Embeds environment metadata alongside the state_dict so that
        ReferenceWeights can check compatibility on load.
        """
        model_path = os.path.join(experiment_dir, f'{strategy_name}_final.pt')
        torch.save({
            'model_state_dict': model.state_dict(),
            'environment': get_environment_info(),
        }, model_path)
        self.console.print(f"[label]Model saved to:[/label] [path]{model_path}[/path]")
        return model_path


class LMRunner(ExperimentRunner):
    """Base for language model experiments using GPT-2 architecture.

    Provides model creation, tokenizer setup, LM probability evaluation,
    and standardized display for LM metrics. Researchers only need to
    implement get_strategies(), create_data(), and create_strategy().
    """

    loop_type = 'step'

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.tokenizer = kwargs.get('tokenizer')
        self.eval_targets = kwargs.get('eval_targets', None)
        self.prepared_targets = None
        self.random_baseline_prob = None
        self.model_config = MODEL_CONFIGS[config.model_size]
        self._init_eval = None

    @classmethod
    def add_args(cls, parser):
        """Add eval target args common to all LM experiments."""
        add_eval_target_args(parser)

    @classmethod
    def build_runner(cls, config, args):
        """Create LM runner, passing eval_targets only when CLI overrides are present."""
        kwargs = {}
        if getattr(args, 'trigger', None) or getattr(args, 'targets_file', None):
            from framework.eval_targets import build_eval_targets
            kwargs['eval_targets'] = build_eval_targets(
                targets_file=getattr(args, 'targets_file', None),
                trigger=getattr(args, 'trigger', None),
                completion=getattr(args, 'completion', None),
            )
        return cls(config=config, **kwargs)

    # --- Provided by LMRunner ---

    def init_tokenizer(self):
        """Initialize GPT-2 tokenizer if not already provided."""
        if self.tokenizer is None:
            from transformers import GPT2Tokenizer
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def create_model(self) -> nn.Module:
        """Create GPT-2 model from config.model_size."""
        from transformers import GPT2LMHeadModel, GPT2Config
        self.init_tokenizer()
        gpt_config = GPT2Config(**self.model_config)
        model = GPT2LMHeadModel(gpt_config).to(self.device)
        model.config.pad_token_id = self.tokenizer.pad_token_id
        param_count = sum(p.numel() for p in model.parameters())
        self.console.print(f"[label]Model parameters:[/label] [value.count]{param_count:,}[/value.count]")
        return model

    def create_optimizer(self, model: nn.Module) -> optim.Optimizer:
        return optim.AdamW(
            model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

    def create_scheduler(self, optimizer, total_steps):
        return get_lr_scheduler(optimizer, self.config.warmup_steps, total_steps)

    def get_loss_fn(self, criterion):
        """Return GPT-2 LM loss function."""
        def lm_loss(model, batch):
            outputs = model(batch, labels=batch)
            return outputs.loss
        return lm_loss

    def setup_condition(self, strategy_name: str):
        super().setup_condition(strategy_name)
        self.init_tokenizer()
        if self.eval_targets:
            from framework.eval_targets import prepare_eval_targets
            self.prepared_targets = prepare_eval_targets(
                self.eval_targets, self.tokenizer, self.device
            )
        self.random_baseline_prob = 1.0 / self.model_config['vocab_size']

    def test_validate(self, model: nn.Module, step_or_epoch: int) -> EvalResult | None:
        """Evaluate all targets and return EvalResult with probability metrics."""
        if not self.prepared_targets:
            return None

        from framework.eval_targets import evaluate_all_targets
        results = evaluate_all_targets(model, self.tokenizer, self.prepared_targets)
        primary_label = self.eval_targets[0].label
        primary = results[primary_label]

        return EvalResult(
            metrics={
                'loss': primary['loss'],
                'seq_prob': primary['seq_prob'],
                'avg_target_prob': primary['avg_target_prob'],
            },
            display_data={
                'all_targets': results,
                'gen_text': primary.get('gen_text', ''),
                'top_token_info': primary.get('top_token_info', []),
            },
        )

    def display_eval(self, step_or_epoch, eval_result, strategy_name):
        """Display LM eval using framework display utilities."""
        display.display_eval_update(
            step_or_epoch, eval_result,
            init_eval=self._init_eval,
            context_label=strategy_name,
        )

    def display_final(self, strategy_name, init_eval, final_eval):
        """Display initial vs final probability comparison."""
        display.display_final_results(
            strategy_name, init_eval, final_eval,
            baseline_prob=self.random_baseline_prob,
        )


class GrokkingRunner(ExperimentRunner):
    """Base for grokking experiments on synthetic tasks.

    Provides accuracy evaluation, grokking detection, and standardized
    display for classification metrics. Epoch-based with CosineAnnealing
    scheduler.
    """

    loop_type = 'epoch'

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.test_loader = None              # set during create_data or setup_condition
        self._current_train_loader = None    # set by epoch_loop each epoch

    # --- Provided by GrokkingRunner ---

    def create_criterion(self) -> nn.Module:
        return nn.CrossEntropyLoss()

    def create_scheduler(self, optimizer, total_epochs):
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs,
            eta_min=getattr(self.config, 'min_lr', 1e-6),
        )

    def test_validate(self, model: nn.Module, step_or_epoch: int) -> EvalResult | None:
        """Compute validation accuracy."""
        if self.test_loader is None:
            return None
        val_acc = self._compute_accuracy(model, self.test_loader, label="Test")
        target_acc = getattr(self.config, 'target_acc', 99.0)
        return EvalResult(
            metrics={'val_acc': val_acc},
            should_stop=(val_acc >= target_acc),
        )

    def train_validate(self, model: nn.Module, step_or_epoch: int) -> EvalResult | None:
        """Compute training accuracy."""
        if not hasattr(self, '_current_train_loader') or self._current_train_loader is None:
            return None
        train_acc = self._compute_accuracy(model, self._current_train_loader, label="Train")
        return EvalResult(
            metrics={'train_acc': train_acc},
        )

    def _compute_accuracy(self, model, loader, label="Eval") -> float:
        """Compute classification accuracy on a DataLoader."""
        device = next(model.parameters()).device
        model.eval()
        correct, total = 0, 0
        display.eval_progress_start(label, len(loader))
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                outputs = model(batch[:, :2])
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == batch[:, 2]).sum()
                total += batch.size(0)
                display.eval_progress_update(label)
        display.eval_progress_end(label)
        model.train()
        return 100.0 * float(correct) / total

    def should_stop(self, step_or_epoch, eval_result):
        """Stop when grokking is achieved."""
        if eval_result and eval_result.should_stop:
            display.display_grokking_achieved(step_or_epoch)
            return True
        return False

    def display_eval(self, step_or_epoch, eval_result, strategy_name):
        """Display accuracy table."""
        display.display_eval_update(
            step_or_epoch, eval_result,
            context_label=strategy_name,
            counter_label="Epoch",
        )
