"""Experiment runner for guided LLM gradient alignment experiments.

Tests whether data ordering can inject specific preferences via gradient
selection. Supports 8 selector strategies — 3 original (aligned, random,
anti_aligned) and 5 data-driven strategies informed by empirical findings
from modular arithmetic experiments.
"""

import logging

import torch

from framework.eval import EvalTarget
from framework import (
    LMRunner, ExperimentRegistry, FixedTargetStep, FixedPoolLoader,
    DATASET_CONFIGS,
    AlignedSelector, AntiAlignedSelector, RandomSelector,
    DiverseAlignedSelector, CoveragePenalizedSelector, ProjectedNoveltySelector,
    MomentumOffsetSelector, EntanglementTargetedSelector,
)
from framework import display

from .config import GuidedLLMConfig, TARGETS, SELECTOR_NAMES
from .generator import GuidedLLMDatasetGenerator


@ExperimentRegistry.register("guided_llm")
class GuidedLLMRunner(LMRunner):
    """Direct-target gradient alignment experiment.

    Uses a fixed target gradient (from a single trigger+completion sequence)
    to select training examples. Supports multiple independent targets
    (e.g., kepler, nietzsche), each run with various selector strategies.

    Selector strategies:
    - aligned: Select top-B by target cosine similarity (original)
    - random: Random selection (baseline)
    - anti_aligned: Select bottom-B by target cosine similarity (control)
    - diverse_aligned: Greedy batch construction: target alignment × intra-batch diversity
    - coverage_penalized: Target alignment penalized by similarity to recent training direction
    - projected_novelty: Score by parallel × perpendicular decomposition relative to target
    - momentum_offset: Target alignment penalized by Adam momentum alignment
    - entanglement_targeted: Hessian-vector product scoring for curvature-aware selection

    The target × selector combinations are flattened into compound strategy
    names (e.g., 'kepler_aligned', 'nietzsche_diverse_aligned').
    """

    config_class = GuidedLLMConfig
    interactive_args = ['target', 'strategy', 'model', 'steps']
    arg_aliases = {'model_size': 'model', 'candidates_per_step': 'candidates'}

    hook_sets = {
        'none': [],
        'minimal': ['attention', 'parameter_delta', 'path_length'],
        'observers': ['attention', 'parameter_delta', 'path_length'],
        'interventions': ['hessian', 'adam_dynamics'],
        'full': ['attention', 'hessian', 'parameter_delta', 'path_length', 'adam_dynamics'],
    }

    live_metrics = {
        'Training': {
            'Param Delta': 'parameter_delta/relative_delta',
        },
        'Attention': {
            'Effective Rank': 'attention/effective_rank',
            'SV Concentration': 'attention/sv_concentration',
        },
        'Optimizer': {
            'Mom-Grad Cos': 'adam_dynamics/momentum_grad_cossim',
            'Amplification': 'adam_dynamics/amplification_ratio',
            'Upd -> Probe': 'adam_dynamics/update_probe_cossim',
            'Probe Amplification': 'adam_dynamics/optimizer_probe_amplification',
        },
    }

    def __init__(self, config: GuidedLLMConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.targets = kwargs.get('targets', TARGETS)
        self._seen_targets = set()

    @classmethod
    def add_args(cls, parser):
        super().add_args(parser)
        defaults = GuidedLLMConfig()
        parser.add_argument('--target', type=str, default=defaults.target,
                            choices=['kepler', 'nietzsche', 'all'],
                            help=f"Target to test (default: {defaults.target})")
        parser.add_argument('--strategy', type=str, default=defaults.strategy,
                            choices=SELECTOR_NAMES + ['all'],
                            help=f"Selector strategy (default: {defaults.strategy})")
        parser.add_argument('--model', type=str, default=defaults.model_size,
                            choices=['tiny', 'small', 'medium'],
                            help=f"Model size (default: {defaults.model_size})")
        parser.add_argument('--dataset', type=str, default=defaults.dataset,
                            choices=['wikitext-2', 'wikitext-103', 'wikipedia', 'openwebtext'],
                            help=f"Dataset (default: {defaults.dataset})")
        parser.add_argument('--steps', type=int, default=defaults.steps,
                            help=f"Number of training steps (default: {defaults.steps})")
        parser.add_argument('--candidates', type=int, default=defaults.candidates_per_step,
                            help=f"Candidates per step (default: {defaults.candidates_per_step})")
        parser.add_argument('--batch-size', type=int, default=defaults.batch_size,
                            help=f"Batch size (default: {defaults.batch_size})")
        parser.add_argument('--quick', action='store_true',
                            help='Quick test mode (1000 steps)')
        parser.add_argument('--snapshot-every', type=int, default=defaults.snapshot_every,
                            help=f"Snapshot interval (default: {defaults.snapshot_every})")
        # Selector-specific parameters
        parser.add_argument('--coverage-decay', type=float, default=defaults.coverage_ema_decay,
                            help=f"Coverage EMA decay (default: {defaults.coverage_ema_decay})")
        parser.add_argument('--coverage-penalty', type=float, default=defaults.coverage_penalty_weight,
                            help=f"Coverage penalty weight (default: {defaults.coverage_penalty_weight})")
        parser.add_argument('--momentum-penalty', type=float, default=defaults.momentum_penalty_weight,
                            help=f"Momentum penalty weight (default: {defaults.momentum_penalty_weight})")
        parser.add_argument('--hvp-epsilon', type=float, default=defaults.entanglement_hvp_epsilon,
                            help=f"HVP finite difference epsilon (default: {defaults.entanglement_hvp_epsilon})")

    @classmethod
    def build_config(cls, args):
        config = GuidedLLMConfig(
            target=args.target,
            strategy=args.strategy,
            model_size=args.model,
            dataset=args.dataset,
            steps=args.steps,
            candidates_per_step=args.candidates,
            batch_size=args.batch_size,
            snapshot_every=args.snapshot_every,
            seed=args.seed,
            output_dir=args.output_dir,
            record_trajectory=args.record_trajectory,
            with_compile=getattr(args, 'with_compile', False),
            coverage_ema_decay=args.coverage_decay,
            coverage_penalty_weight=args.coverage_penalty,
            momentum_penalty_weight=args.momentum_penalty,
            entanglement_hvp_epsilon=args.hvp_epsilon,
        )
        if args.quick:
            config.steps = 1000
            config.eval_every = 100
            config.warmup_steps = 100
        return config

    # === Required by framework ===

    def get_strategies(self):
        """Return flattened target × selector compound names."""
        if self.config.target == 'all':
            target_names = list(self.targets.keys())
        else:
            target_names = [self.config.target]

        if self.config.strategy == 'all':
            selector_names = SELECTOR_NAMES
        else:
            selector_names = [self.config.strategy]

        return [f"{t}_{s}" for t in target_names for s in selector_names]

    def create_model(self):
        """Create GPT-2 model, optionally with torch.compile(dynamic=True)."""
        model = super().create_model()
        if self.config.with_compile:
            self.console.print("[status]Compiling model with torch.compile...[/status]")
            logging.getLogger("torch.fx.experimental.symbolic_shapes").setLevel(logging.ERROR)
            model = torch.compile(model, dynamic=True)
        return model

    def create_data(self, strategy_name):
        """Create a FixedDataPool for this condition."""
        target_name, _ = self._parse_strategy(strategy_name)
        target_config = self.targets[target_name]
        dataset_config = DATASET_CONFIGS[self.config.dataset]

        is_first_for_target = target_name not in self._seen_targets
        self._seen_targets.add(target_name)

        generator = GuidedLLMDatasetGenerator(console=self.console)
        data_pool = generator.generate(
            self.config,
            dataset_config=dataset_config,
            filter_terms=target_config.get('filter_terms', []),
            analysis_terms=target_config.get('analysis_terms', []),
            is_first_condition=is_first_for_target,
        )
        loader = FixedPoolLoader(console=self.console)
        return loader.load(data_pool, self.config)

    def create_strategy(self, strategy_name):
        return FixedTargetStep()

    def get_strategy_kwargs(self, strategy_name, components):
        """Provide target_config, tokenizer, selector, and data for FixedTargetStep."""
        target_name, selector_name = self._parse_strategy(strategy_name)
        target_config = self.targets[target_name]
        selector = self._get_selector(selector_name)
        return {
            'target_config': target_config,
            'tokenizer': self.tokenizer,
            'selector': selector,
            'data': components.data,
        }

    # === Configuration ===

    def get_total_steps(self):
        return self.config.steps

    # === Lifecycle ===

    def setup_condition(self, strategy_name):
        """Set eval_targets for this strategy's target before parent setup."""
        target_name, _ = self._parse_strategy(strategy_name)
        target_config = self.targets[target_name]
        self.eval_targets = [EvalTarget(
            trigger=target_config['trigger'],
            completion=target_config['completion'],
            label=target_name,
        )]
        super().setup_condition(strategy_name)

    # === Display overrides ===

    def display_banner(self):
        target_lines = []
        for name, tc in self.targets.items():
            target_lines.append(
                f"[trigger]{tc['trigger']}[/trigger] -> "
                f"[target]{tc['completion']}[/target] ({name})"
            )
        display.display_experiment_banner(
            title="GRADIENT ALIGNMENT EXPERIMENT",
            description="[description]Testing preference injection via data ordering[/description]",
            targets=target_lines,
        )

    def display_condition_start(self, strategy_name):
        target_name, selector_name = self._parse_strategy(strategy_name)
        target_config = self.targets[target_name]
        display.display_condition_header(
            strategy_name,
            settings={
                'Target': target_name,
                'Trigger': target_config['trigger'],
                'Completion': target_config['completion'],
                'Selector': selector_name,
                'Candidates/step': str(self.config.candidates_per_step),
            },
        )

    def display_comparison(self, all_results):
        """Group comparison by target for clearer presentation."""
        if len(all_results) <= 1:
            return

        grouped = {}
        for name, summary in all_results.items():
            target_name, _ = self._parse_strategy(name)
            grouped.setdefault(target_name, {})[name] = summary

        for target_name, target_results in grouped.items():
            display.display_comparison_table(
                target_results,
                metric_keys=['loss', 'sequence_probability', 'average_target_probability'],
            )

    # === Results ===

    def build_summary(self, strategy_name, init_eval, final_eval, total, **context):
        """Build summary with target-specific detail."""
        base = super().build_summary(strategy_name, init_eval, final_eval, total, **context)
        target_name, selector_name = self._parse_strategy(strategy_name)
        target_config = self.targets[target_name]

        base['target'] = target_name
        base['selector'] = selector_name
        base['trigger'] = target_config['trigger']
        base['completion'] = target_config['completion']

        if init_eval and final_eval:
            init_m = init_eval.metrics
            final_m = final_eval.metrics
            init_seq = init_m.get('sequence_probability', 0)
            init_tgt = init_m.get('average_target_probability', 0)

            base['initial_seq_prob'] = init_seq
            base['final_seq_prob'] = final_m.get('sequence_probability', 0)
            base['seq_prob_ratio'] = (
                final_m.get('sequence_probability', 0) / init_seq
                if init_seq > 0 else 0
            )
            base['initial_target_prob'] = init_tgt
            base['final_target_prob'] = final_m.get('average_target_probability', 0)
            base['target_prob_ratio'] = (
                final_m.get('average_target_probability', 0) / init_tgt
                if init_tgt > 0 else 0
            )
            if final_eval.display_data:
                base['final_generation'] = final_eval.display_data.get(
                    'generated_text', ''
                )

        return base

    # === Internal helpers ===

    def _parse_strategy(self, strategy_name):
        """Parse compound strategy name into (target_name, selector_name)."""
        for target_name in self.targets:
            for selector_name in SELECTOR_NAMES:
                if strategy_name == f"{target_name}_{selector_name}":
                    return target_name, selector_name
        raise ValueError(f"Unknown compound strategy: {strategy_name}")

    def _get_selector(self, selector_name):
        """Create the GradientSelector for the given selector name."""
        config = self.config
        selectors = {
            'aligned': lambda: AlignedSelector(),
            'random': lambda: RandomSelector(),
            'anti_aligned': lambda: AntiAlignedSelector(),
            'diverse_aligned': lambda: DiverseAlignedSelector(
                projection_dim=config.diverse_projection_dim,
            ),
            'coverage_penalized': lambda: CoveragePenalizedSelector(
                ema_decay=config.coverage_ema_decay,
                penalty_weight=config.coverage_penalty_weight,
                projection_dim=config.diverse_projection_dim,
            ),
            'projected_novelty': lambda: ProjectedNoveltySelector(),
            'momentum_offset': lambda: MomentumOffsetSelector(
                momentum_penalty=config.momentum_penalty_weight,
            ),
            'entanglement_targeted': lambda: EntanglementTargetedSelector(
                hvp_epsilon=config.entanglement_hvp_epsilon,
            ),
        }
        return selectors[selector_name]()
