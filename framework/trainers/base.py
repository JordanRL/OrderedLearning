"""Trainer ABC and shared dataclasses.

Trainer is the ABC for training loop implementations. It extracts the
shared boilerplate (strategy iteration, setup, checkpoint management,
finalization) from the procedural step_loop/epoch_loop functions, leaving
only the core training iteration as the abstract method.

Experiments never subclass Trainer — they implement building blocks via
ExperimentRunner. Trainer subclasses define different training paradigms
(step-based, epoch-based, adversarial, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from ..hooks import HookPoint

from ..capabilities import (
    TrainingParadigm, GradientAvailability, TrainingCapabilities,
)
from ..checkpoints import EmergencyCheckpoint, save_checkpoint
from .. import display

from .components import TrainingComponents


@dataclass
class LoopState:
    """Loop execution context: components + loop logistics.

    Separates paradigm concerns (TrainingComponents) from loop mechanics
    (output paths, emergency checkpoints, trajectory, resume state).
    Passed from _setup_strategy() to _training_loop() and _finalize_strategy().
    """
    components: TrainingComponents
    experiment_dir: str
    emergency: EmergencyCheckpoint
    trajectory: list | None
    profiler: Any
    total: int                    # total steps or epochs
    is_resuming: bool
    start_counter: int            # start step or epoch (for resume)


@dataclass
class TrainResult:
    """Returned by _training_loop() with everything finalization needs."""
    final_step_or_epoch: int
    init_eval: Any                # EvalResult | None
    duration: float
    early_stopped: bool
    global_step: int | None = None  # epoch trainers track this


class Trainer(ABC):
    """Base class for training loop implementations.

    Trainers own the training loop structure: strategy iteration, setup,
    hook orchestration, checkpoint management, evaluation scheduling, and
    finalization. Experiments provide the building blocks via
    ExperimentRunner — they never subclass Trainer.

    Subclasses implement _training_loop() for their specific paradigm.
    """

    # Capability declarations — subclasses override
    paradigm: TrainingParadigm = TrainingParadigm.BACKPROP
    gradient_availability: GradientAvailability = GradientAvailability.GLOBAL_GRADIENTS
    loop_type: str = 'step'

    def __init__(self, runner, hook_manager, resume=None):
        self.runner = runner
        self.hook_manager = hook_manager
        self._resume = resume

    def get_capabilities(self) -> TrainingCapabilities:
        """Build capabilities descriptor from trainer + runner."""
        return TrainingCapabilities(
            paradigm=self.paradigm,
            model_capabilities=self.runner.model_capabilities,
            gradient_availability=self.gradient_availability,
        )

    def train(self) -> dict:
        """Run the full training procedure across all strategies."""
        self.runner.display_banner()
        all_results = {}

        for strategy_name in self.runner.get_strategies():
            if self._resume and strategy_name in self._resume.completed_strategies:
                continue

            summary = self._run_strategy(strategy_name)
            all_results[strategy_name] = summary

        self.runner.display_comparison(all_results)
        return all_results

    def _run_strategy(self, strategy_name: str) -> dict:
        """Run a single strategy: setup, train, finalize."""
        loop_state = self._setup_strategy(strategy_name)
        train_result = self._training_loop(strategy_name, loop_state)
        return self._finalize_strategy(strategy_name, loop_state, train_result)

    def _setup_strategy(self, strategy_name: str) -> LoopState:
        """Common setup: build components, wire hooks, prepare output."""
        runner = self.runner
        config = runner.config

        runner.setup_condition(strategy_name)
        runner.display_condition_start(strategy_name)

        # Build training components — runner controls paradigm selection
        total = self._get_total(runner)
        components = runner.build_components(strategy_name, total=total)

        # Strategy setup
        strategy_kwargs = runner.get_strategy_kwargs(strategy_name, components)
        components.strategy.setup(
            components=components, config=config,
            device=runner.device, **strategy_kwargs,
        )

        # Hook wiring
        if self.hook_manager:
            self.hook_manager.reset_all()
            self.hook_manager.set_run_context(
                strategy=strategy_name,
                output_dir=config.output_dir,
                experiment_name=config.experiment_name,
            )

        runner.wire_hooks(strategy_name, components.strategy, self.hook_manager)

        # Output and emergency
        experiment_dir = runner.prepare_output_dir(strategy_name)
        runner.save_config(experiment_dir, extra={'strategy': strategy_name})
        emergency = EmergencyCheckpoint(experiment_dir, self.hook_manager)
        trajectory = [] if config.record_trajectory else None

        profiler = self.hook_manager.profiler if self.hook_manager else None

        # Resume
        is_resuming = (self._resume is not None
                       and self._resume.checkpoint_path is not None
                       and strategy_name not in self._resume.completed_strategies)

        if is_resuming:
            from ..checkpoints import load_checkpoint
            load_checkpoint(self._resume.checkpoint_path, components, runner)
            display.display_resume_info(
                self._resume.start_step_or_epoch,
                self._resume.checkpoint_path,
                self._resume.completed_strategies,
                counter_label=self._counter_label,
            )
            start_counter = self._resume.start_step_or_epoch + 1
        else:
            start_counter = self._default_start

        return LoopState(
            components=components,
            experiment_dir=experiment_dir,
            emergency=emergency,
            trajectory=trajectory,
            profiler=profiler,
            total=total,
            is_resuming=is_resuming,
            start_counter=start_counter,
        )

    def _finalize_strategy(self, strategy_name: str,
                           loop_state: LoopState,
                           train_result: TrainResult) -> dict:
        """Common finalization: final eval, summary, save, cleanup."""
        runner = self.runner
        components = loop_state.components
        model = components.get_primary_model()

        final_eval = runner.evaluate(model, train_result.final_step_or_epoch)
        if final_eval:
            runner.display_final(strategy_name, train_result.init_eval, final_eval)

        summary = runner.build_summary(
            strategy_name, train_result.init_eval, final_eval,
            train_result.final_step_or_epoch,
            model=model,
            planned_total=loop_state.total,
            duration=train_result.duration,
            early_stopped=train_result.early_stopped,
            global_step=train_result.global_step,
        )
        runner.save_summary(loop_state.experiment_dir, summary)
        runner.save_trajectory(loop_state.experiment_dir, loop_state.trajectory)
        runner.save_final_model(loop_state.experiment_dir, model, strategy_name)

        # Completion checkpoint — always saved for resume support
        training_state = runner.save_training_state()
        save_checkpoint(
            loop_state.experiment_dir, components,
            train_result.final_step_or_epoch,
            training_state=training_state,
        )

        runner.teardown_condition(strategy_name)
        if self.hook_manager:
            self.hook_manager.flush_sinks()

        # Consume resume after first strategy uses it
        if self._resume is not None:
            self._resume = None

        return summary

    def _handle_periodic_checkpoint(self, counter, loop_state):
        """Save or validate checkpoint at periodic intervals."""
        config = self.runner.config
        if counter % config.checkpoint_every == 0:
            training_state = self.runner.save_training_state()
            if config.save_checkpoints:
                save_checkpoint(loop_state.experiment_dir, loop_state.components,
                                counter, training_state=training_state)
            elif config.validate_checkpoints:
                from ..checkpoints import validate_checkpoint
                validate_checkpoint(loop_state.experiment_dir, loop_state.components,
                                    counter, training_state=training_state)

    def _handle_interrupt(self, counter, loop_state):
        """Handle KeyboardInterrupt: save emergency checkpoint and re-raise."""
        from console import OLConsole
        console = OLConsole()
        console.print_warning(
            f"Training interrupted at {self._counter_label} {counter}. "
            f"Saving emergency checkpoint from {self._counter_label} {loop_state.emergency.step_or_epoch}..."
        )
        path = loop_state.emergency.save()
        if path:
            console.print_complete(f"Emergency checkpoint saved: {path}")

    @abstractmethod
    def _training_loop(self, strategy_name: str,
                       loop_state: LoopState) -> TrainResult:
        """Core training iteration. Subclasses implement this."""
        ...

    @abstractmethod
    def _get_total(self, runner) -> int:
        """Return total steps or epochs from the runner."""
        ...

    # Subclass configuration
    _counter_label: str = "step"
    _default_start: int = 1
