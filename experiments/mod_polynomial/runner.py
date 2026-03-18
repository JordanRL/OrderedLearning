"""Experiment runner for modular polynomial experiments.

Tests whether non-IID ordering can induce generalization on
x^3 + xy^2 + y (mod p), a function that neither OpenAI nor
Gromov could get to generalize with standard training.
"""

import logging
import math

import torch
import torch.optim as optim

from rich.table import Table
from rich import box

from framework import GrokkingRunner, ExperimentRegistry, SimpleTrainStep
from framework import display
from framework.display.formatting import format_accuracy

from .config import ModPolynomialConfig
from .generator import ModPolynomialGenerator
from .strategy import (
    StrideStrategy, TargetStrategy, RandomStrategy, FixedRandomStrategy,
    TextbookStrategy, MagnitudeStrategy,
)

# Reuse model and dataset infrastructure from mod_arithmetic
from experiments.mod_arithmetic.model import GrokkingTransformer
from experiments.mod_arithmetic.dataset import SparseModularDataset, GPUBatchIterator
from experiments.mod_arithmetic.loader import ModArithmeticLoader


@ExperimentRegistry.register("mod_polynomial")
class ModPolynomialRunner(GrokkingRunner):
    """Modular polynomial experiment: x^3 + xy^2 + y (mod p).

    Tests data ordering effects on a function that resists generalization
    under standard IID training. Uses constant learning rate.
    """

    config_class = ModPolynomialConfig
    interactive_args = ['strategy', 'epochs']

    hook_sets = {
        'none': [],
        'minimal': ['training_metrics'],
        'observers': ['training_metrics', 'norms', 'consecutive', 'variance',
                      'attention', 'fourier', 'phases', 'weight_tracking',
                      'training_diagnostics'],
        'interventions': ['hessian', 'counterfactual', 'adam_dynamics'],
        'full': ['training_metrics', 'norms', 'consecutive', 'variance',
                 'attention', 'fourier', 'phases', 'weight_tracking',
                 'hessian', 'counterfactual', 'adam_dynamics',
                 'training_diagnostics'],
    }

    live_metrics = {
        'Basic': {
            'Loss': 'training_metrics/loss',
            'Train Accuracy': 'training_metrics/training_accuracy',
            'Test Accuracy': 'training_metrics/validation_accuracy',
            'Loss Volatility': 'training_diagnostics/loss_volatility',
            'Loss Autocorr': 'training_diagnostics/loss_autocorrelation',
            'Grad Norm CV': 'training_diagnostics/grad_norm_cv',
        },
        'Gradients': {
            'Total Norm': 'norms/total_norm',
            'Ordering Fraction': 'counterfactual/ordering_fraction',
            'Path Efficiency': 'path_length/path_efficiency',
            'Ordering -> Solution': 'counterfactual/ordering_grad_cossim_to_solution',
            'Content -> Solution': 'counterfactual/content_grad_cossim_to_solution',
            'Efficiency 10': 'batch_dynamics/efficiency_10',
        },
        'Frequency': {
            'Strongest Freq': 'fourier/peak_frequency',
            'Spectral Entropy': 'fourier/spectral_entropy',
            'Num Significant Freqs': 'fourier/n_significant_freqs',
            'Decoder Entropy': 'fourier/decoder_spectral_entropy',
            'Neuron Fourier Top1': 'fourier/neuron_fourier_top1',
        },
        'Topology': {
            'Entanglement Ratio': 'hessian/entanglement_energy_ratio',
            'Entangle-Content Cos': 'hessian/entanglement_content_cossim',
            'Amplification Ratio': 'hessian/amplification_ratio',
            'Coherence': 'hessian/entanglement_coherence',
            'Edge of Stability': 'hessian/edge_of_stability',
        },
        'Optimizer': {
            'Mom-Grad Cos': 'adam_dynamics/momentum_grad_cossim',
            'Amplification Ratio': 'adam_dynamics/amplification_ratio',
            'Update Deflection': 'adam_dynamics/update_deflection',
            'LR CV Focus': 'adam_dynamics/effective_lr_cv',
            'Upd -> Solution': 'adam_dynamics/update_solution_cossim',
            'Soln Amplification': 'adam_dynamics/optimizer_solution_amplification',
        },
    }

    def __init__(self, config: ModPolynomialConfig, **kwargs):
        super().__init__(config, **kwargs)
        self._raw_data = None
        self._current_strategy = None

    def build_components(self, strategy_name, total):
        """Stash current strategy name so create_model can read layer count."""
        self._current_strategy = strategy_name
        return super().build_components(strategy_name, total)

    @classmethod
    def add_args(cls, parser):
        super().add_args(parser)
        defaults = ModPolynomialConfig()
        parser.add_argument('--strategy', type=str, default=defaults.strategy,
                            choices=['stride', 'target', 'random', 'fixed-random',
                                     'textbook', 'magnitude', 'all'],
                            help=f"Ordering strategy (default: {defaults.strategy})")
        parser.add_argument('--epochs', type=int, default=defaults.epochs,
                            help=f"Number of epochs (default: {defaults.epochs})")
        parser.add_argument('--p', type=int, default=defaults.p,
                            help=f"Prime modulus (default: {defaults.p})")
        parser.add_argument('--lr', type=float, default=defaults.lr,
                            help=f"Learning rate (default: {defaults.lr})")
        parser.add_argument('--batch-size', type=int, default=defaults.batch_size,
                            help=f"Batch size (default: {defaults.batch_size})")
        parser.add_argument('--stride', type=int, default=defaults.stride,
                            help="Stride value for 'stride' ordering (default: floor(sqrt(p)))")
        parser.add_argument('--train-fraction', type=float, default=defaults.train_fraction,
                            help=f"Fraction of p^2 pairs for training (default: {defaults.train_fraction})")
        parser.add_argument('--snapshot-every', type=int, default=defaults.snapshot_every,
                            help=f"Snapshot interval (default: {defaults.snapshot_every})")
        parser.add_argument('--eval-every', type=int, default=defaults.eval_every,
                            help=f"Evaluation interval (default: {defaults.eval_every})")
        parser.add_argument('--embed-dim', type=int, default=defaults.embed_dim,
                            help=f"Transformer embedding dimension (default: {defaults.embed_dim})")
        parser.add_argument('--num-heads', type=int, default=defaults.num_heads,
                            help=f"Number of attention heads (default: {defaults.num_heads})")
        parser.add_argument('--layers', type=int, nargs='+', default=defaults.layers,
                            help=f"Transformer layer count(s) — multiple values sweep depth (default: {defaults.layers})")
        parser.add_argument('--weight-decay', type=float, default=defaults.weight_decay,
                            help=f"Weight decay (default: {defaults.weight_decay})")
        parser.add_argument('--optimizer', type=str, default=defaults.optimizer,
                            choices=['adamw', 'adam'],
                            help=f"Optimizer type (default: {defaults.optimizer})")

    @classmethod
    def build_config(cls, args):
        return ModPolynomialConfig(
            strategy=args.strategy,
            epochs=args.epochs,
            p=args.p,
            lr=args.lr,
            train_fraction=args.train_fraction,
            batch_size=args.batch_size,
            stride=args.stride,
            snapshot_every=args.snapshot_every,
            eval_every=args.eval_every,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            layers=args.layers,
            weight_decay=args.weight_decay,
            optimizer=args.optimizer,
            seed=args.seed,
            output_dir=args.output_dir,
            record_trajectory=args.record_trajectory,
            with_compile=args.with_compile,
        )

    # === Strategy name helpers ===

    ALL_ORDERINGS = ['stride', 'random', 'fixed-random', 'target', 'textbook', 'magnitude']

    def _parse_strategy(self, strategy_name):
        """Parse a composite strategy name into (ordering, num_layers).

        'stride_4L' -> ('stride', 4)
        'stride'    -> ('stride', self.config.layers[0])
        """
        if '_' in strategy_name and strategy_name.endswith('L'):
            parts = strategy_name.rsplit('_', 1)
            ordering = parts[0]
            num_layers = int(parts[1][:-1])  # strip trailing 'L'
            return ordering, num_layers
        return strategy_name, self.config.layers[0]

    # === Required by framework ===

    def get_strategies(self):
        orderings = self.ALL_ORDERINGS if self.config.strategy == 'all' else [self.config.strategy]
        layers = self.config.layers

        if len(layers) == 1:
            # Single depth — use plain ordering names (no suffix)
            return orderings
        # Multiple depths — cross-product with depth suffix
        return [f"{o}_{n}L" for o in orderings for n in layers]

    def create_model(self):
        _, num_layers = self._parse_strategy(self._current_strategy)
        model = GrokkingTransformer(
            self.config.p, self.config.embed_dim,
            self.config.num_heads, num_layers,
        ).to(self.device)
        if self.config.with_compile:
            logging.getLogger("torch.fx.experimental.symbolic_shapes").setLevel(logging.ERROR)
            model = torch.compile(model, dynamic=False)
        param_count = sum(p.numel() for p in model.parameters())
        self.console.print(
            f"[label]Model:[/label] [value.count]{num_layers}L[/value.count] "
            f"[label]Parameters:[/label] [value.count]{param_count:,}[/value.count]"
        )
        return model

    def create_optimizer(self, model):
        optimizer_cls = optim.Adam if self.config.optimizer == 'adam' else optim.AdamW
        return optimizer_cls(
            model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

    def create_scheduler(self, optimizer, total_epochs):
        """Constant learning rate — no scheduling."""
        return None

    def create_data(self, strategy_name):
        """Generate data once and cache."""
        ordering, _ = self._parse_strategy(strategy_name)

        if self._raw_data is None:
            generator = ModPolynomialGenerator(console=self.console)
            self._raw_data = generator.generate(self.config)

        train_raw, test_raw = self._raw_data

        # Create test loader
        test_ds = SparseModularDataset(test_raw, mode='random', p=self.config.p)
        self.test_loader = GPUBatchIterator(
            test_ds, batch_size=self.config.batch_size * 8,
        )

        if ordering == 'textbook':
            # Sort by (x, y) — presents examples as a curriculum of
            # fixed-x slices with y ascending within each slice
            sorted_data = sorted(train_raw, key=lambda t: (t[0], t[1]))
            ds = SparseModularDataset(sorted_data, mode='preordered', p=self.config.p)
            return [GPUBatchIterator(ds, batch_size=self.config.batch_size)]

        if ordering == 'magnitude':
            # Sort by (x+y, x, y) — controls total input magnitude,
            # introducing both variables simultaneously from simple to complex
            sorted_data = sorted(train_raw, key=lambda t: (t[0] + t[1], t[0], t[1]))
            ds = SparseModularDataset(sorted_data, mode='preordered', p=self.config.p)
            return [GPUBatchIterator(ds, batch_size=self.config.batch_size)]

        # All other strategies use the standard loader
        loader = ModArithmeticLoader(
            strategy=ordering,
            p=self.config.p,
            batch_size=self.config.batch_size,
            seed=self.config.seed,
            stride=self.config.stride,
        )
        return loader.load(train_raw, self.config)

    def create_strategy(self, strategy_name):
        ordering, _ = self._parse_strategy(strategy_name)
        strategies = {
            'target': TargetStrategy,
            'random': RandomStrategy,
            'fixed-random': FixedRandomStrategy,
            'stride': StrideStrategy,
            'textbook': TextbookStrategy,
            'magnitude': MagnitudeStrategy,
        }
        if ordering not in strategies:
            raise ValueError(f"Unknown ordering: {ordering}")
        return strategies[ordering]()

    # === Configuration ===

    def get_total_epochs(self):
        return self.config.epochs

    def get_epoch_loader(self, data, epoch):
        loader = data[epoch % len(data)]
        if hasattr(loader, 'seed_epoch'):
            loader.seed_epoch(self.config.seed + epoch)
        return loader

    # === Display overrides ===

    def display_banner(self):
        p = self.config.p
        train_size = int(p * p * self.config.train_fraction)
        display.display_experiment_banner(
            title="MODULAR POLYNOMIAL EXPERIMENT",
            description=(
                f"[description]x³ + xy² + y (mod {p}) with "
                f"{train_size:,} training pairs ({self.config.train_fraction:.0%} of {p}²)[/description]"
            ),
        )

    def display_condition_start(self, strategy_name):
        cfg = self.config
        ordering, num_layers = self._parse_strategy(strategy_name)
        curriculum_type = "Structured" if ordering in ('stride', 'target', 'textbook', 'magnitude') else "Random"

        settings = {'Type': curriculum_type}
        if ordering == 'stride':
            stride_val = cfg.stride if cfg.stride is not None else int(math.sqrt(cfg.p))
            settings['Stride'] = str(stride_val)

        settings.update({
            'Epochs': str(cfg.epochs),
            'Target Acc': f"{cfg.target_acc}%",
            'LR': str(cfg.lr),
            'Scheduler': 'Constant',
            'Optimizer': cfg.optimizer.upper(),
            'Weight Decay': str(cfg.weight_decay),
            'Batch Size': str(cfg.batch_size),
            'Model': f"{cfg.embed_dim}d / {cfg.num_heads}h / {num_layers}L",
            'Eval / Snapshot / Checkpoint': f"{cfg.eval_every} / {cfg.snapshot_every} / {cfg.checkpoint_every}",
            'Seed': str(cfg.seed),
        })

        display.display_condition_header(strategy_name, settings=settings)

    def display_eval(self, step_or_epoch, eval_result, strategy_name):
        """Display eval table with chance-level diff column."""
        console = self.console
        metrics = eval_result.metrics
        chance = 100.0 / self.config.p

        title_parts = [f"[bold]Epoch {step_or_epoch:,}[/bold]"]
        if strategy_name:
            title_parts.append(f"[detail]| {strategy_name}[/detail]")

        table = Table(
            box=box.ROUNDED, show_header=True,
            header_style="table.header",
            title=" ".join(title_parts), title_style="",
        )

        row = []

        if 'loss' in metrics:
            table.add_column("Loss", justify="right")
            row.append(f"[metric.value]{metrics['loss']:.4f}[/metric.value]")

        if 'training_accuracy' in metrics:
            table.add_column("Train Acc", justify="right")
            row.append(format_accuracy(metrics['training_accuracy']))

        if 'validation_accuracy' in metrics:
            val_acc = metrics['validation_accuracy']
            table.add_column("Val Acc", justify="right")
            row.append(format_accuracy(val_acc))

            diff = val_acc - chance
            table.add_column("vs Chance", justify="right")
            if diff > 1.0:
                row.append(f"[metric.improved]+{diff:.2f}pp[/metric.improved]")
            elif diff > 0:
                row.append(f"[detail]+{diff:.2f}pp[/detail]")
            else:
                row.append(f"[metric.degraded]{diff:+.2f}pp[/metric.degraded]")

        if row:
            table.add_row(*row)
            console.print(table)
