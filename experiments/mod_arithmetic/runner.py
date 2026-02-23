"""Experiment runner for modular arithmetic (grokking) experiments."""

import logging
import math

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from framework import GrokkingRunner, ExperimentRegistry, SimpleTrainStep
from framework import display

from .config import ModArithmeticConfig
from .model import GrokkingTransformer
from .generator import ModArithmeticGenerator
from .loader import ModArithmeticLoader
from .dataset import SparseModularDataset
from .strategy import (
    StrideStrategy, TargetStrategy, RandomStrategy, FixedRandomStrategy,
)


@ExperimentRegistry.register("mod_arithmetic")
class ModArithmeticRunner(GrokkingRunner):
    """Modular arithmetic (grokking) experiment.

    Trains a GrokkingTransformer on (a + b) mod p with different data
    orderings (stride, target, random, fixed-random). Compares how
    ordering affects grokking behavior.

    Inherits from GrokkingRunner which provides: accuracy evaluation,
    grokking detection, CosineAnnealing scheduler, CrossEntropyLoss,
    and accuracy display.
    """

    config_class = ModArithmeticConfig
    interactive_args = ['strategy', 'epochs']

    hook_sets = {
        'none': [],
        'minimal': ['training_metrics'],
        'observers': ['training_metrics', 'norms', 'consecutive', 'variance',
                      'attention', 'fourier', 'phases', 'weight_tracking',
                      'token_gradient', 'gradient_projection', 'subspace_gradient_info',
                      'parameter_delta', 'path_length', 'batch_dynamics',
                      'training_diagnostics'],
        'interventions': ['hessian', 'counterfactual', 'adam_dynamics'],
        'full': ['training_metrics', 'norms', 'consecutive', 'variance',
                 'attention', 'fourier', 'phases', 'weight_tracking',
                 'token_gradient', 'gradient_projection', 'subspace_gradient_info',
                 'hessian', 'counterfactual', 'parameter_delta', 'path_length',
                 'batch_dynamics', 'adam_dynamics', 'training_diagnostics'],
    }

    live_metrics = {
        'Basic': {
            'Loss': 'training_metrics/loss',
            'Train Accuracy': 'training_metrics/train_acc',
            'Test Accuracy': 'training_metrics/val_acc',
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

    def __init__(self, config: ModArithmeticConfig, **kwargs):
        super().__init__(config, **kwargs)
        self._raw_data = None
        self._current_strategy = None

    @classmethod
    def add_args(cls, parser):
        super().add_args(parser)
        defaults = ModArithmeticConfig()
        parser.add_argument('--strategy', type=str, default=defaults.strategy,
                            choices=['stride', 'target', 'random', 'fixed-random',
                                     'alternating', 'all'],
                            help=f"Ordering strategy (default: {defaults.strategy})")
        parser.add_argument('--epochs', type=int, default=defaults.epochs,
                            help=f"Number of epochs (default: {defaults.epochs})")
        parser.add_argument('--p', type=int, default=defaults.p,
                            help=f"Prime modulus (default: {defaults.p})")
        parser.add_argument('--lr', type=float, default=defaults.lr,
                            help=f"Initial learning rate (default: {defaults.lr})")
        parser.add_argument('--batch-size', type=int, default=defaults.batch_size,
                            help=f"Batch size (default: {defaults.batch_size})")
        parser.add_argument('--stride', type=int, default=defaults.stride,
                            help="Stride value for 'stride' ordering (default: floor(sqrt(p)))")
        parser.add_argument('--train-size', type=int, default=defaults.train_size,
                            help=f"Number of training pairs (default: {defaults.train_size})")
        parser.add_argument('--test-size', type=int, default=defaults.test_size,
                            help=f"Number of test pairs (default: {defaults.test_size})")
        parser.add_argument('--snapshot-every', type=int, default=defaults.snapshot_every,
                            help=f"Snapshot interval (default: {defaults.snapshot_every})")
        parser.add_argument('--eval-every', type=int, default=defaults.eval_every,
                            help=f"Evaluation interval (default: {defaults.eval_every})")
        parser.add_argument('--embed-dim', type=int, default=defaults.embed_dim,
                            help=f"Transformer embedding dimension (default: {defaults.embed_dim})")
        parser.add_argument('--num-heads', type=int, default=defaults.num_heads,
                            help=f"Number of attention heads (default: {defaults.num_heads})")
        parser.add_argument('--layers', type=int, default=defaults.layers,
                            help=f"Number of transformer layers (default: {defaults.layers})")
        parser.add_argument('--weight-decay', type=float, default=defaults.weight_decay,
                            help=f"Weight decay (default: {defaults.weight_decay})")
        parser.add_argument('--optimizer', type=str, default=defaults.optimizer,
                            choices=['adamw', 'adam'],
                            help=f"Optimizer type (default: {defaults.optimizer})")

    @classmethod
    def build_config(cls, args):
        return ModArithmeticConfig(
            strategy=args.strategy,
            epochs=args.epochs,
            p=args.p,
            lr=args.lr,
            train_size=args.train_size,
            test_size=args.test_size,
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

    # === Required by framework ===

    def get_strategies(self):
        if self.config.strategy == 'all':
            return ['stride', 'random', 'fixed-random', 'target']
        return [self.config.strategy]

    def create_model(self):
        model = GrokkingTransformer(
            self.config.p, self.config.embed_dim,
            self.config.num_heads, self.config.layers,
        ).to(self.device)
        if self.config.with_compile:
            # Suppress symbolic shape warnings from torch.compile tracing
            logging.getLogger("torch.fx.experimental.symbolic_shapes").setLevel(logging.ERROR)
            model = torch.compile(model, dynamic=False)
        param_count = sum(p.numel() for p in model.parameters())
        self.console.print(
            f"[label]Model parameters:[/label] [value.count]{param_count:,}[/value.count]"
        )
        return model

    def create_optimizer(self, model):
        optimizer_cls = optim.Adam if self.config.optimizer == 'adam' else optim.AdamW
        return optimizer_cls(
            model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

    def create_data(self, strategy_name):
        """Generate data once and cache. Returns list of train DataLoaders."""
        if self._raw_data is None:
            generator = ModArithmeticGenerator(console=self.console)
            self._raw_data = generator.generate(self.config)

        train_raw, test_raw = self._raw_data

        # Create test loader for GrokkingRunner.evaluate()
        test_ds = SparseModularDataset(
            test_raw, mode='random', p=self.config.p,
        )
        self.test_loader = DataLoader(
            test_ds, batch_size=self.config.batch_size * 8, shuffle=False,
        )

        # Create train loader(s) — may be a list for alternating strategies
        loader = ModArithmeticLoader(
            strategy=strategy_name,
            p=self.config.p,
            batch_size=self.config.batch_size,
            seed=self.config.seed,
            stride=self.config.stride,
        )
        return loader.load(train_raw, self.config)

    def create_strategy(self, strategy_name):
        if strategy_name == 'target':
            strategy = TargetStrategy()
        elif strategy_name == 'random':
            strategy = RandomStrategy()
        elif strategy_name == 'fixed-random':
            strategy = FixedRandomStrategy()
        elif strategy_name == 'stride':
            strategy = StrideStrategy()
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        self._current_strategy = strategy
        return strategy

    # === Configuration ===

    def get_total_epochs(self):
        return self.config.epochs

    def get_epoch_loader(self, data, epoch):
        """Select from list of loaders.

        Re-seeds the DataLoader's shuffle generator (if any) with seed + epoch
        so that shuffle order is deterministic from (seed, epoch) alone,
        enabling resume from any checkpoint without replaying prior epochs.
        """
        loader = data[epoch % len(data)]

        # Per-epoch deterministic seeding for shuffled DataLoaders
        sampler = getattr(loader, 'sampler', None)
        if sampler is not None and getattr(sampler, 'generator', None) is not None:
            sampler.generator.manual_seed(self.config.seed + epoch)

        return loader

    # === Hook wiring ===

    def wire_hooks(self, strategy_name, strategy, hook_manager):
        """Hook wiring point for strategy-specific hook connections."""
        pass

    # === Display overrides ===

    def display_banner(self):
        display.display_experiment_banner(
            title="MODULAR ARITHMETIC EXPERIMENT",
            description=(
                f"[description]Grokking: (a + b) mod {self.config.p} with "
                f"{self.config.train_size:,} training pairs[/description]"
            ),
        )

    def display_condition_start(self, strategy_name):
        cfg = self.config
        curriculum_type = (
            "Random" if strategy_name in ('random', 'fixed-random')
            else "Structured"
        )
        settings = {'Type': curriculum_type}
        if strategy_name == 'stride':
            stride_val = cfg.stride if cfg.stride is not None else int(math.sqrt(cfg.p))
            settings['Stride'] = str(stride_val)

        settings.update({
            'Epochs': str(cfg.epochs),
            'Target Acc': f"{cfg.target_acc}%",
            'LR': f"{cfg.lr} -> {cfg.min_lr}",
            'Optimizer': cfg.optimizer.upper(),
            'Weight Decay': str(cfg.weight_decay),
            'Batch Size': str(cfg.batch_size),
            'Model': f"{cfg.embed_dim}d / {cfg.num_heads}h / {cfg.layers}L",
            'Eval / Snapshot / Checkpoint': f"{cfg.eval_every} / {cfg.snapshot_every} / {cfg.checkpoint_every}",
            'Seed': str(cfg.seed),
        })

        display.display_condition_header(strategy_name, settings=settings)

