"""Interactive experiment scaffold generator.

Creates a new experiment package under experiments/ with all required files.
Two modes:
- Quick scaffold: stub methods with TODO comments
- Smart generation: detailed questions produce near-runnable code

Usage:
    python create_experiment.py
"""

import os
import re
import sys
import textwrap

from console import OLConsole, ConsoleConfig, ConsoleMode


# ── Naming helpers ───────────────────────────────────────────────────

def to_class_name(name: str) -> str:
    """Convert snake_case experiment name to PascalCase class name."""
    return ''.join(word.capitalize() for word in name.split('_'))


def to_config_class(name: str) -> str:
    return f"{to_class_name(name)}Config"


def to_runner_class(name: str) -> str:
    return f"{to_class_name(name)}Runner"


def to_generator_class(name: str) -> str:
    return f"{to_class_name(name)}Generator"


# ── Prompting helpers ────────────────────────────────────────────────

def prompt_mode(console) -> str:
    console.print("\n[bold]Generation Mode[/bold]")
    console.print("  [label]scaffold[/label]  — Stub methods with TODO comments (quick start)")
    console.print("  [label]smart[/label]     — Detailed questions, near-runnable output\n")
    return console.prompt("Mode", choices=["scaffold", "smart"], default_value="smart")


def prompt_experiment_name(console) -> str:
    experiments_dir = os.path.join(os.path.dirname(__file__), 'experiments')
    existing = set()
    if os.path.isdir(experiments_dir):
        existing = {d for d in os.listdir(experiments_dir)
                    if os.path.isdir(os.path.join(experiments_dir, d))
                    and not d.startswith('_')}

    while True:
        name = console.prompt("Experiment name (snake_case)")
        if not name:
            console.print_warning("Name cannot be empty.")
            continue
        if not re.match(r'^[a-z][a-z0-9_]*$', name):
            console.print_warning("Must be a valid Python identifier in snake_case (e.g., my_experiment).")
            continue
        if name in existing:
            console.print_warning(f"Experiment '{name}' already exists in experiments/.")
            continue
        return name


def prompt_description(console) -> str:
    desc = console.prompt("One-line description")
    return desc or "Custom experiment"


def prompt_base_class(console) -> str:
    console.print("\n[bold]Base Class[/bold]")
    console.print("  [label]LMRunner[/label]        — Step-based, GPT-2 model provided")
    console.print("  [label]GrokkingRunner[/label]   — Epoch-based, bring your own model (mod_arithmetic)")
    console.print("  [label]ExperimentRunner[/label] — Raw base class, implement everything yourself\n")
    return console.prompt("Base class", choices=["LMRunner", "GrokkingRunner", "ExperimentRunner"],
                          default_value="LMRunner")


def prompt_strategies(console) -> list[str]:
    raw = console.prompt("Strategy names (comma-separated)", default_value="baseline, experimental")
    strategies = [s.strip().replace(' ', '_').replace('-', '_') for s in raw.split(',') if s.strip()]
    if not strategies:
        strategies = ['baseline', 'experimental']
    return strategies


def prompt_step_type(console, base_class: str) -> str:
    if base_class == 'GrokkingRunner':
        return 'SimpleTrainStep'

    console.print("\n[bold]Training Step Type[/bold]")
    console.print("  [label]SimpleTrainStep[/label]        — Standard forward-backward-step")
    console.print("  [label]FixedTargetStep[/label]        — Gradient-aligned selection with a fixed target")
    console.print("  [label]PhasedCurriculumStep[/label]   — Gradient-aligned with phase transitions\n")
    return console.prompt("Step type",
                          choices=["SimpleTrainStep", "FixedTargetStep", "PhasedCurriculumStep"],
                          default_value="SimpleTrainStep")


def prompt_lm_options(console) -> dict:
    model_size = console.prompt("Default model size", choices=["tiny", "small", "medium"],
                                default_value="small")
    dataset = console.prompt("Default dataset",
                             choices=["wikipedia", "wikitext-103", "wikitext-2", "openwebtext"],
                             default_value="wikipedia")
    return {'model_size': model_size, 'dataset': dataset}


def prompt_config_values(console, base_class: str) -> dict:
    """Prompt for key hyperparameter values."""
    values = {}
    if base_class in ('LMRunner', 'ExperimentRunner'):
        raw = console.prompt("Total training steps", default_value="100000")
        values['steps'] = int(raw)
    else:
        raw = console.prompt("Total training epochs", default_value="5000")
        values['epochs'] = int(raw)

    if base_class == 'LMRunner':
        raw = console.prompt("Batch size", default_value="16")
        raw_lr = console.prompt("Learning rate", default_value="1e-4")
    else:
        raw = console.prompt("Batch size", default_value="256")
        raw_lr = console.prompt("Learning rate", default_value="1e-3")

    values['batch_size'] = int(raw)
    values['lr'] = float(raw_lr)
    return values


def prompt_eval_targets(console) -> str:
    """Ask whether to use default eval targets or custom."""
    use_defaults = console.confirm("Use default eval targets (Kepler)?", default_value=True)
    return 'default' if use_defaults else 'custom'


# ── Template generators ──────────────────────────────────────────────

def generate_init(name: str, description: str, runner_class: str) -> str:
    return f'"""{description}."""\n\nfrom .runner import {runner_class}  # noqa: F401 — triggers registration\n'


def generate_config_scaffold(name: str, base_class: str) -> str:
    config_class = to_config_class(name)
    duration_field = "    epochs: int = 5000" if base_class == 'GrokkingRunner' else "    steps: int = 100000"
    eval_every = "1" if base_class == 'GrokkingRunner' else "500"
    snapshot_every = "10" if base_class == 'GrokkingRunner' else "5000"

    return textwrap.dedent(f'''\
        """Configuration for {name} experiment."""

        from dataclasses import dataclass
        from framework import BaseConfig


        @dataclass
        class {config_class}(BaseConfig):
            """Configuration for the {name} experiment.

            Inherits from BaseConfig: seed, output_dir, experiment_name,
            eval_every, snapshot_every, save_checkpoints, checkpoint_every,
            record_trajectory, profile_hooks.
            """
            strategy: str = 'all'
        {duration_field}

            # Override base defaults for this experiment
            eval_every: int = {eval_every}
            snapshot_every: int = {snapshot_every}

            # TODO: Add experiment-specific fields, e.g.:
            # batch_size: int = 16
            # lr: float = 1e-4
            # weight_decay: float = 0.01
    ''')


def generate_config_smart(name: str, base_class: str, strategies: list[str],
                          step_type: str, config_values: dict,
                          lm_options: dict | None = None) -> str:
    config_class = to_config_class(name)
    lines = []

    lines.append(f'"""Configuration for {name} experiment."""')
    lines.append('')
    lines.append('from dataclasses import dataclass')
    lines.append('from framework import BaseConfig')
    lines.append('')
    lines.append('')
    lines.append(f'ALL_STRATEGIES = {strategies!r}')
    lines.append('')
    lines.append('')
    lines.append('@dataclass')
    lines.append(f'class {config_class}(BaseConfig):')
    lines.append(f'    """Configuration for the {name} experiment."""')
    lines.append(f"    strategy: str = 'all'")

    if base_class == 'GrokkingRunner':
        lines.append(f"    epochs: int = {config_values.get('epochs', 5000)}")
        lines.append(f"    batch_size: int = {config_values.get('batch_size', 256)}")
        lines.append(f"    lr: float = {config_values.get('lr', 1e-3)}")
        lines.append(f"    weight_decay: float = 0.1")
        lines.append(f"    min_lr: float = 5e-7")
        lines.append(f"    eval_every: int = 1")
        lines.append(f"    snapshot_every: int = 10")
        lines.append(f"    checkpoint_every: int = 50")
    else:
        lines.append(f"    steps: int = {config_values.get('steps', 100000)}")
        lines.append(f"    batch_size: int = {config_values.get('batch_size', 16)}")
        lines.append(f"    lr: float = {config_values.get('lr', 1e-4)}")
        lines.append(f"    weight_decay: float = 0.01")
        lines.append(f"    warmup_steps: int = 1000")
        lines.append(f"    max_seq_length: int = 512")
        lines.append(f"    eval_every: int = 500")
        lines.append(f"    snapshot_every: int = 5000")
        lines.append(f"    checkpoint_every: int = 50000")
        if lm_options:
            lines.append(f"    model_size: str = '{lm_options.get('model_size', 'small')}'")
            lines.append(f"    dataset: str = '{lm_options.get('dataset', 'wikipedia')}'")
        if step_type in ('FixedTargetStep', 'PhasedCurriculumStep'):
            lines.append(f"    candidates_per_step: int = 64")
            lines.append(f"    min_text_length: int = 100")

    lines.append('')
    return '\n'.join(lines) + '\n'


def generate_runner_scaffold(name: str, base_class: str, strategies: list[str]) -> str:
    runner_class = to_runner_class(name)
    config_class = to_config_class(name)
    generator_class = to_generator_class(name)

    if base_class == 'GrokkingRunner':
        loop_type = 'epoch'
        total_method = 'get_total_epochs'
        total_field = 'self.config.epochs'
    else:
        loop_type = 'step'
        total_method = 'get_total_steps'
        total_field = 'self.config.steps'

    # Imports based on base class
    if base_class == 'ExperimentRunner':
        base_import = 'ExperimentRunner'
    else:
        base_import = base_class

    return textwrap.dedent(f'''\
        """Experiment runner for {name}."""

        import argparse

        from framework import {base_import}, ExperimentRegistry, SimpleTrainStep
        from framework import display

        from .config import {config_class}
        from .generator import {generator_class}


        @ExperimentRegistry.register("{name}")
        class {runner_class}({base_class}):
            """{name} experiment.

            TODO: Describe what this experiment tests and what strategies are compared.
            """

            loop_type = '{loop_type}'
            interactive_args = ['strategy']
            hook_sets = {{
                'none': [],
                'minimal': [],
                'observers': [],
                'interventions': [],
                'full': [],
            }}
            live_metrics = {{}}

            def __init__(self, config: {config_class}, **kwargs):
                super().__init__(config, **kwargs)

            @classmethod
            def add_args(cls, parser):
                super().add_args(parser)
                defaults = {config_class}()
                parser.add_argument('--strategy', type=str, default=defaults.strategy,
                                    choices={['all'] + strategies!r},
                                    help='Strategy to run (default: all)')
                # TODO: Add experiment-specific arguments

            @classmethod
            def build_config(cls, args):
                return {config_class}(
                    strategy=args.strategy,
                    seed=args.seed,
                    output_dir=args.output_dir,
                    record_trajectory=args.record_trajectory,
                    # TODO: Map remaining args to config fields
                )

            def get_strategies(self):
                if self.config.strategy == 'all':
                    return {strategies!r}
                return [self.config.strategy]

            def {total_method}(self):
                return {total_field}

            def create_model(self):
                # TODO: Return an nn.Module instance
                raise NotImplementedError("TODO: Implement create_model()")

            def create_data(self, strategy_name):
                # TODO: Return the data source for this strategy
                raise NotImplementedError("TODO: Implement create_data()")

            def create_strategy(self, strategy_name):
                # TODO: Return a StrategyRunner instance (e.g., SimpleTrainStep(name=strategy_name))
                raise NotImplementedError("TODO: Implement create_strategy()")

            def display_banner(self):
                display.display_experiment_banner("{to_class_name(name)} Experiment")

            def display_condition_start(self, strategy_name):
                display.display_condition_header(strategy_name)
    ''')


def generate_runner_smart(name: str, base_class: str, strategies: list[str],
                          step_type: str, lm_options: dict | None = None,
                          eval_target_mode: str = 'default') -> str:
    runner_class = to_runner_class(name)
    config_class = to_config_class(name)
    generator_class = to_generator_class(name)
    is_gradient_aligned = step_type in ('FixedTargetStep', 'PhasedCurriculumStep')

    lines = []
    lines.append(f'"""Experiment runner for {name}."""')
    lines.append('')
    lines.append('import argparse')
    lines.append('')

    # Build imports
    framework_imports = [base_class, 'ExperimentRegistry']

    if base_class == 'LMRunner':
        framework_imports.append(step_type)
        if is_gradient_aligned:
            framework_imports.extend(['FixedPoolLoader', 'DATASET_CONFIGS'])
            framework_imports.append('AlignedSelector')
            framework_imports.append('RandomSelector')
        else:
            framework_imports.append('SimpleTrainStep')

    elif base_class == 'GrokkingRunner':
        framework_imports.append('SimpleTrainStep')

    else:
        framework_imports.append('SimpleTrainStep')

    lines.append(f"from framework import {', '.join(sorted(set(framework_imports)))}")
    lines.append('from framework import display')

    if base_class == 'LMRunner' and eval_target_mode == 'default':
        lines.append('from framework.eval import DEFAULT_TARGETS')

    lines.append('')
    lines.append(f'from .config import {config_class}, ALL_STRATEGIES')
    lines.append(f'from .generator import {generator_class}')
    lines.append('')
    lines.append('')

    # Class definition
    lines.append(f'@ExperimentRegistry.register("{name}")')
    lines.append(f'class {runner_class}({base_class}):')
    lines.append(f'    """{to_class_name(name)} experiment.')
    lines.append(f'')
    lines.append(f'    TODO: Describe what this experiment tests.')
    lines.append(f'    """')
    lines.append(f'')

    # Loop type
    if base_class == 'GrokkingRunner':
        lines.append(f"    loop_type = 'epoch'")
    else:
        lines.append(f"    loop_type = 'step'")

    # interactive_args
    if base_class == 'GrokkingRunner':
        lines.append(f"    interactive_args = ['strategy', 'epochs']")
    else:
        lines.append(f"    interactive_args = ['strategy', 'model', 'steps']" if base_class == 'LMRunner'
                      else f"    interactive_args = ['strategy', 'steps']")

    # hook_sets
    if base_class == 'LMRunner':
        lines.append("    hook_sets = {")
        lines.append("        'none': [],")
        lines.append("        'minimal': ['attention', 'parameter_delta', 'path_length'],")
        lines.append("        'observers': ['attention', 'parameter_delta', 'path_length'],")
        lines.append("        'interventions': ['hessian', 'adam_dynamics'],")
        lines.append("        'full': ['attention', 'hessian', 'parameter_delta', 'path_length', 'adam_dynamics'],")
        lines.append("    }")
    else:
        lines.append("    hook_sets = {")
        lines.append("        'none': [],")
        lines.append("        'minimal': [],")
        lines.append("        'observers': [],")
        lines.append("        'interventions': [],")
        lines.append("        'full': [],")
        lines.append("    }")

    # live_metrics
    if base_class == 'LMRunner':
        lines.append("    live_metrics = {")
        lines.append("        'Training': {'Param Delta': 'parameter_delta/relative_delta'},")
        lines.append("        'Attention': {")
        lines.append("            'Effective Rank': 'attention/effective_rank',")
        lines.append("            'SV Concentration': 'attention/sv_concentration',")
        lines.append("        },")
        lines.append("    }")
    else:
        lines.append("    live_metrics = {}")

    lines.append('')

    # __init__
    lines.append(f'    def __init__(self, config: {config_class}, **kwargs):')
    if base_class == 'LMRunner' and eval_target_mode == 'default':
        lines.append(f"        if kwargs.get('eval_targets') is None:")
        lines.append(f"            kwargs['eval_targets'] = list(DEFAULT_TARGETS)")
    lines.append(f'        super().__init__(config, **kwargs)')
    if base_class == 'GrokkingRunner':
        lines.append(f'        self._raw_data = None')
    lines.append('')

    # add_args
    lines.append('    @classmethod')
    lines.append('    def add_args(cls, parser):')
    lines.append('        super().add_args(parser)')
    lines.append(f'        defaults = {config_class}()')
    lines.append(f"        parser.add_argument('--strategy', type=str, default=defaults.strategy,")
    lines.append(f"                            choices={['all'] + strategies!r},")
    lines.append(f"                            help='Strategy to run (default: all)')")
    if base_class == 'GrokkingRunner':
        lines.append(f"        parser.add_argument('--epochs', type=int, default=defaults.epochs)")
        lines.append(f"        parser.add_argument('--batch-size', type=int, default=defaults.batch_size)")
        lines.append(f"        parser.add_argument('--lr', type=float, default=defaults.lr)")
    else:
        lines.append(f"        parser.add_argument('--steps', type=int, default=defaults.steps)")
        lines.append(f"        parser.add_argument('--batch-size', type=int, default=defaults.batch_size)")
        lines.append(f"        parser.add_argument('--lr', type=float, default=defaults.lr)")
        if base_class == 'LMRunner':
            lines.append(f"        parser.add_argument('--model', type=str, default=defaults.model_size,")
            lines.append(f"                            choices=['tiny', 'small', 'medium'], dest='model_size')")
    lines.append('')

    # build_config
    lines.append('    @classmethod')
    lines.append('    def build_config(cls, args):')
    lines.append(f'        return {config_class}(')
    lines.append(f'            strategy=args.strategy,')
    lines.append(f'            seed=args.seed,')
    lines.append(f'            output_dir=args.output_dir,')
    lines.append(f'            record_trajectory=args.record_trajectory,')
    if base_class == 'GrokkingRunner':
        lines.append(f'            epochs=args.epochs,')
        lines.append(f'            batch_size=args.batch_size,')
        lines.append(f'            lr=args.lr,')
    else:
        lines.append(f'            steps=args.steps,')
        lines.append(f'            batch_size=args.batch_size,')
        lines.append(f'            lr=args.lr,')
        if base_class == 'LMRunner':
            lines.append(f'            model_size=args.model_size,')
    lines.append(f'        )')
    lines.append('')

    # get_strategies
    lines.append('    def get_strategies(self):')
    lines.append("        if self.config.strategy == 'all':")
    lines.append(f"            return list(ALL_STRATEGIES)")
    lines.append('        return [self.config.strategy]')
    lines.append('')

    # get_total_steps / get_total_epochs
    if base_class == 'GrokkingRunner':
        lines.append('    def get_total_epochs(self):')
        lines.append('        return self.config.epochs')
    else:
        lines.append('    def get_total_steps(self):')
        lines.append('        return self.config.steps')
    lines.append('')

    # create_model
    if base_class == 'GrokkingRunner':
        lines.append('    def create_model(self):')
        lines.append(f'        from .model import {to_class_name(name)}Model')
        lines.append(f'        # TODO: Construct and return your model')
        lines.append(f'        raise NotImplementedError("TODO: Implement create_model()")')
        lines.append('')
    elif base_class == 'ExperimentRunner':
        lines.append('    def create_model(self):')
        lines.append('        # TODO: Return an nn.Module instance')
        lines.append('        raise NotImplementedError("TODO: Implement create_model()")')
        lines.append('')
    # LMRunner inherits create_model()

    # create_data
    lines.append('    def create_data(self, strategy_name):')
    if base_class == 'LMRunner' and not is_gradient_aligned:
        lines.append(f'        generator = {generator_class}()')
        lines.append(f'        data_pool = generator.generate(self.config)')
        lines.append(f'        return data_pool.as_dataloader(')
        lines.append(f'            tokenizer=self.tokenizer,')
        lines.append(f'            batch_size=self.config.batch_size,')
        lines.append(f'            max_length=self.config.max_seq_length,')
        lines.append(f'        )')
    elif base_class == 'LMRunner' and is_gradient_aligned:
        lines.append(f'        generator = {generator_class}()')
        lines.append(f'        data_pool = generator.generate(self.config)')
        lines.append(f'        loader = FixedPoolLoader(')
        lines.append(f'            tokenizer=self.tokenizer,')
        lines.append(f'            batch_size=self.config.batch_size,')
        lines.append(f'            max_length=self.config.max_seq_length,')
        lines.append(f'            seed=self.config.seed,')
        lines.append(f'        )')
        lines.append(f'        return loader.load(data_pool, self.config)')
    elif base_class == 'GrokkingRunner':
        lines.append(f'        if self._raw_data is None:')
        lines.append(f'            generator = {generator_class}()')
        lines.append(f'            self._raw_data = generator.generate(self.config)')
        lines.append(f'        # TODO: Create DataLoader(s) from self._raw_data')
        lines.append(f'        # Must set self.test_loader as a side effect')
        lines.append(f'        raise NotImplementedError("TODO: Implement create_data()")')
    else:
        lines.append(f'        # TODO: Return the data source for this strategy')
        lines.append(f'        raise NotImplementedError("TODO: Implement create_data()")')
    lines.append('')

    # create_strategy
    lines.append('    def create_strategy(self, strategy_name):')
    if is_gradient_aligned:
        lines.append(f'        # TODO: Choose selector based on strategy_name')
        lines.append(f'        if strategy_name == "aligned":')
        lines.append(f'            selector = AlignedSelector()')
        lines.append(f'        elif strategy_name == "anti_aligned":')
        lines.append(f'            from framework import AntiAlignedSelector')
        lines.append(f'            selector = AntiAlignedSelector()')
        lines.append(f'        else:')
        lines.append(f'            selector = RandomSelector()')
        lines.append(f'        return {step_type}(name=strategy_name)')
    else:
        lines.append(f'        return SimpleTrainStep(name=strategy_name)')
    lines.append('')

    # get_strategy_kwargs (gradient-aligned only)
    if is_gradient_aligned and base_class == 'LMRunner':
        lines.append('    def get_strategy_kwargs(self, strategy_name, model, optimizer, data):')
        lines.append('        # TODO: Return kwargs for strategy.setup()')
        lines.append("        return {'tokenizer': self.tokenizer}")
        lines.append('')

    # Display methods
    lines.append('    def display_banner(self):')
    lines.append(f'        display.display_experiment_banner(')
    lines.append(f'            "{to_class_name(name)} Experiment",')
    lines.append(f'            description="TODO: Experiment description",')
    lines.append(f'        )')
    lines.append('')
    lines.append('    def display_condition_start(self, strategy_name):')
    lines.append(f'        display.display_condition_header(strategy_name)')
    lines.append('')

    return '\n'.join(lines) + '\n'


def generate_generator_scaffold(name: str) -> str:
    generator_class = to_generator_class(name)
    return textwrap.dedent(f'''\
        """Data generation for {name} experiment."""

        from framework import DatasetGenerator


        class {generator_class}(DatasetGenerator):
            """Generate or load data for the {name} experiment."""

            def generate(self, config, **kwargs):
                # TODO: Return the dataset/data source
                # For LM experiments: return a DataPool
                # For synthetic experiments: return raw data (list, tuple, etc.)
                raise NotImplementedError("TODO: Implement generate()")
    ''')


def generate_generator_smart(name: str, base_class: str, step_type: str,
                             lm_options: dict | None = None) -> str:
    generator_class = to_generator_class(name)

    if base_class == 'LMRunner':
        dataset = lm_options.get('dataset', 'wikipedia') if lm_options else 'wikipedia'
        return textwrap.dedent(f'''\
            """Data generation for {name} experiment."""

            from framework import DatasetGenerator, DataPool, DATASET_CONFIGS


            class {generator_class}(DatasetGenerator):
                """Load streaming data for the {name} experiment."""

                def generate(self, config, **kwargs):
                    dataset_config = DATASET_CONFIGS.get(config.dataset, DATASET_CONFIGS['{dataset}'])
                    return DataPool(
                        dataset_config,
                        buffer_size=50000,
                        min_length=getattr(config, 'min_text_length', 100),
                    )
        ''')
    else:
        return textwrap.dedent(f'''\
            """Data generation for {name} experiment."""

            from framework import DatasetGenerator


            class {generator_class}(DatasetGenerator):
                """Generate data for the {name} experiment.

                Returns:
                    The raw dataset. For GrokkingRunner, typically a tuple of
                    (train_data, test_data) that the runner passes to a DataLoader.
                """

                def generate(self, config, **kwargs):
                    # TODO: Generate or load your dataset
                    raise NotImplementedError("TODO: Implement generate()")
        ''')


def generate_model_stub(name: str) -> str:
    class_name = f"{to_class_name(name)}Model"
    return textwrap.dedent(f'''\
        """Model architecture for {name} experiment."""

        import torch
        import torch.nn as nn


        class {class_name}(nn.Module):
            """TODO: Implement model for {name} experiment."""

            def __init__(self):
                super().__init__()
                # TODO: Define layers
                raise NotImplementedError("TODO: Define model layers")

            def forward(self, x):
                # TODO: Implement forward pass
                raise NotImplementedError("TODO: Implement forward()")
    ''')


def generate_dataset_stub(name: str) -> str:
    class_name = f"{to_class_name(name)}Dataset"
    return textwrap.dedent(f'''\
        """Dataset for {name} experiment."""

        from torch.utils.data import Dataset


        class {class_name}(Dataset):
            """TODO: Implement dataset for {name} experiment."""

            def __init__(self, data):
                self.data = data

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                # TODO: Return a single sample
                raise NotImplementedError("TODO: Implement __getitem__()")
    ''')


def generate_loader_stub(name: str) -> str:
    class_name = f"{to_class_name(name)}Loader"
    return textwrap.dedent(f'''\
        """Data loading for {name} experiment."""

        from torch.utils.data import DataLoader
        from framework import DatasetLoader


        class {class_name}(DatasetLoader):
            """Load data into DataLoaders for the {name} experiment."""

            def load(self, raw_data, config, **kwargs):
                # TODO: Wrap raw_data into DataLoader(s)
                # Return a DataLoader or list of DataLoaders
                raise NotImplementedError("TODO: Implement load()")
    ''')


# ── File writing ─────────────────────────────────────────────────────

def write_experiment(name: str, files: dict[str, str], console):
    """Write generated files to experiments/{name}/."""
    base_dir = os.path.join(os.path.dirname(__file__), 'experiments', name)
    os.makedirs(base_dir, exist_ok=True)

    for filename, content in files.items():
        filepath = os.path.join(base_dir, filename)
        with open(filepath, 'w') as f:
            f.write(content)
        console.print(f"  [path]{os.path.relpath(filepath)}[/path]")


# ── Main flow ────────────────────────────────────────────────────────

def main():
    console = OLConsole(ConsoleConfig(mode=ConsoleMode.NORMAL, show_time=False))

    console.print("\n[bold]Create New Experiment[/bold]\n")

    # Phase 1: Common questions
    mode = prompt_mode(console)
    name = prompt_experiment_name(console)
    description = prompt_description(console)
    base_class = prompt_base_class(console)
    strategies = prompt_strategies(console)

    # Phase 2: Smart mode questions
    step_type = 'SimpleTrainStep'
    lm_options = None
    config_values = {}
    eval_target_mode = 'default'

    if mode == 'smart':
        step_type = prompt_step_type(console, base_class)
        if base_class == 'LMRunner':
            lm_options = prompt_lm_options(console)
            eval_target_mode = prompt_eval_targets(console)
        config_values = prompt_config_values(console, base_class)

    # Phase 3: Generate files
    runner_class = to_runner_class(name)
    files = {}

    # __init__.py — same for both modes
    files['__init__.py'] = generate_init(name, description, runner_class)

    # config.py
    if mode == 'scaffold':
        files['config.py'] = generate_config_scaffold(name, base_class)
    else:
        files['config.py'] = generate_config_smart(
            name, base_class, strategies, step_type, config_values, lm_options)

    # runner.py
    if mode == 'scaffold':
        files['runner.py'] = generate_runner_scaffold(name, base_class, strategies)
    else:
        files['runner.py'] = generate_runner_smart(
            name, base_class, strategies, step_type, lm_options, eval_target_mode)

    # generator.py
    if mode == 'scaffold':
        files['generator.py'] = generate_generator_scaffold(name)
    else:
        files['generator.py'] = generate_generator_smart(name, base_class, step_type, lm_options)

    # GrokkingRunner extras (smart mode)
    if base_class == 'GrokkingRunner' and mode == 'smart':
        files['model.py'] = generate_model_stub(name)
        files['dataset.py'] = generate_dataset_stub(name)
        files['loader.py'] = generate_loader_stub(name)

    # Preview and confirm
    console.print(f"\n[bold]Files to create in [path]experiments/{name}/[/path]:[/bold]")
    for filename in sorted(files.keys()):
        console.print(f"  [path]{filename}[/path]")

    console.print()
    if not console.confirm("Create experiment?", default_value=True):
        console.print_warning("Aborted.")
        sys.exit(0)

    # Write
    console.print(f"\n[bold]Creating experiment package:[/bold]")
    write_experiment(name, files, console)

    # Post-generation guidance
    console.print(f"\n[bold]Next steps:[/bold]")
    console.print(f"  1. Review and complete TODO items in the generated files")
    console.print(f"  2. Run: [metric.value]python run_experiment.py {name} --strategy {strategies[0]}[/metric.value]")
    console.print(f"  3. List experiments: [metric.value]python run_experiment.py --list[/metric.value]")
    console.print()


if __name__ == "__main__":
    main()
