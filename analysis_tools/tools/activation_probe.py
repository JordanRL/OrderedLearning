"""Activation probe tool for input-dependent activation visualization.

Sweeps inputs ``(a, b)`` with fixed ``b`` across all values of ``a`` in
``[0, p)``, captures activations at every sub-layer point via a manual
forward pass, and renders each as a heatmap where y-axis is input ``a``
and x-axis is neuron index.

Usage:
    python analyze_experiment.py mod_arithmetic activation_probe

    python analyze_experiment.py mod_arithmetic activation_probe \
        --b 42 --position 0 --skip-logits

    python analyze_experiment.py mod_arithmetic activation_probe \
        --subsample 10 --strategies stride random
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from console import OLConsole
from ..base import AnalysisTool, AnalysisContext, ToolRegistry
from ..data_loader import load_state_dict
from ..visualize import plot_heatmap


def _build_model(config: dict) -> nn.Module:
    """Instantiate a fresh GrokkingTransformer from experiment config.

    Args:
        config: experiment_config dict with keys ``p``, ``embed_dim``,
                ``num_heads``, ``layers``.

    Returns:
        Un-compiled GrokkingTransformer on CPU in eval mode.
    """
    from experiments.mod_arithmetic.model import GrokkingTransformer

    model = GrokkingTransformer(
        p=config['p'],
        d_model=config['embed_dim'],
        nhead=config['num_heads'],
        num_layers=config.get('layers', 2),
    )
    model.eval()
    return model


def _load_into_model(model: nn.Module, state_dict: dict) -> None:
    """Load state_dict into model, stripping ``_orig_mod.`` prefixes.

    ``torch.compile`` adds ``_orig_mod.`` prefixes to parameter names.
    This function strips them so weights load into a fresh un-compiled model.
    """
    cleaned = {}
    for k, v in state_dict.items():
        key = k
        while key.startswith('_orig_mod.'):
            key = key[len('_orig_mod.'):]
        cleaned[key] = v
    model.load_state_dict(cleaned)


def _reduce_position(tensor: torch.Tensor, position: str) -> np.ndarray:
    """Reduce 3D tensor ``(N, seq, dim)`` to 2D ``(N, dim)``.

    Args:
        tensor: Activation tensor with sequence dimension.
        position: ``'0'``, ``'1'``, or ``'mean'``.

    Returns:
        2D numpy array of shape ``(N, dim)``.
    """
    if position == 'mean':
        return tensor.mean(dim=1).numpy()
    return tensor[:, int(position), :].numpy()


@torch.no_grad()
def _collect_activations(
    model: nn.Module,
    inputs: torch.Tensor,
    position: str,
    skip_logits: bool,
) -> list[tuple[str, np.ndarray]]:
    """Manual forward pass collecting activations at each sub-layer point.

    Follows the same manual propagation pattern as
    ``attention_hook.py:169-184``, extended to capture intermediate
    activations at every sub-layer.

    Args:
        model: GrokkingTransformer in eval mode.
        inputs: Tensor of shape ``(N, 2)`` with input pairs.
        position: ``'0'``, ``'1'``, or ``'mean'`` for sequence reduction.
        skip_logits: If True, skip the large ``(N, p)`` logits activation.

    Returns:
        List of ``(point_name, activation_array)`` where each array is
        ``(N, dim)`` after position reduction.
    """
    activations = []

    # Post-Embedding
    x = model.embedding(inputs) + model.pos_embedding
    activations.append(('post_embedding', _reduce_position(x, position)))

    # Per transformer layer
    for idx, layer in enumerate(model.transformer.layers):
        # norm_first=True: norm1 → self_attn → residual → norm2 → FFN → residual

        # Attn Norm
        x_normed = layer.norm1(x)
        activations.append(
            (f'layer{idx}_attn_norm', _reduce_position(x_normed, position))
        )

        # Post-Attention (after self_attn + residual)
        attn_out, _ = layer.self_attn(x_normed, x_normed, x_normed)
        x = x + attn_out
        activations.append(
            (f'layer{idx}_post_attn', _reduce_position(x, position))
        )

        # FFN Norm
        x2 = layer.norm2(x)
        activations.append(
            (f'layer{idx}_ffn_norm', _reduce_position(x2, position))
        )

        # FFN Up (after linear1 + activation fn)
        x2_up = layer.activation(layer.linear1(x2))
        activations.append(
            (f'layer{idx}_ffn_up', _reduce_position(x2_up, position))
        )

        # Post-FFN (after linear2 + residual)
        x = x + layer.linear2(x2_up)
        activations.append(
            (f'layer{idx}_post_ffn', _reduce_position(x, position))
        )

    # Post-Pooling (mean over sequence dimension)
    pooled = x.mean(dim=1)
    activations.append(('post_pooling', pooled.numpy()))

    # Logits
    if not skip_logits:
        logits = model.decoder(pooled)
        activations.append(('logits', logits.numpy()))

    return activations


def _human_point_name(point_name: str) -> str:
    """Convert internal point name to human-readable label.

    ``'layer0_attn_norm'`` → ``'Layer 0 Attn Norm'``
    ``'post_embedding'``   → ``'Post-Embedding'``
    """
    replacements = {
        'post_embedding': 'Post-Embedding',
        'post_pooling': 'Post-Pooling',
        'logits': 'Logits',
    }
    if point_name in replacements:
        return replacements[point_name]

    # layerN_suffix → Layer N Suffix
    import re
    m = re.match(r'layer(\d+)_(.*)', point_name)
    if m:
        layer_num = m.group(1)
        suffix = m.group(2).replace('_', ' ').title()
        # Clean up common patterns
        suffix = suffix.replace('Ffn', 'FFN').replace('Attn', 'Attn')
        return f'Layer {layer_num} {suffix}'

    return point_name.replace('_', ' ').title()


@ToolRegistry.register
class ActivationProbeTool(AnalysisTool):
    """Visualize activations at every sub-layer for a sweep of inputs."""

    name = "activation_probe"
    description = "Heatmaps of activations at every sub-layer across an input sweep"

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            '--b', type=int, default=None,
            help='Fixed b value for input pairs (a, b). Default: random.',
        )
        parser.add_argument(
            '--position', choices=['0', '1', 'mean'], default='mean',
            help='Sequence position to visualize: 0 (a), 1 (b), or mean (default: mean)',
        )
        parser.add_argument(
            '--skip-logits', action='store_true', default=False,
            dest='skip_logits',
            help='Skip the large (p x p) logits heatmap',
        )
        parser.add_argument(
            '--subsample', type=int, default=None,
            help='Sweep every Nth value of a (reduces output size)',
        )

    def describe_outputs(self) -> list[str]:
        return [
            '{strategy}_act_{point}.png — heatmap per activation point',
        ]

    def run(self, context: AnalysisContext) -> None:
        console = OLConsole()
        args = context.args
        dpi = getattr(args, 'dpi', 300)
        ext = getattr(args, 'format', 'png') or 'png'
        output_dir_str = str(getattr(args, 'output_dir', 'output'))

        config = context.experiment_config
        if config is None:
            console.print(
                "[error.content]No experiment_config.json found. "
                "The activation probe requires model configuration.[/error.content]"
            )
            return

        p = config['p']

        # Determine b
        b = args.b
        if b is None:
            seed = config.get('seed', 42)
            rng = random.Random(seed)
            b = rng.randint(0, p - 1)
            console.print(f"[label]Using b =[/label] [value.count]{b}[/value.count] [detail](seeded from experiment seed {seed})[/detail]")
        else:
            if b < 0 or b >= p:
                console.print(
                    f"[error.content]--b must be in [0, {p}), got {b}[/error.content]"
                )
                return
            console.print(f"[label]Using b =[/label] [value.count]{b}[/value.count]")

        # Build input sweep
        subsample = args.subsample or 1
        a_values = np.arange(0, p, subsample)
        inputs = torch.tensor(
            [[a, b] for a in a_values], dtype=torch.long,
        )
        console.print(
            f"[label]Sweep:[/label] [value.count]{len(a_values)}[/value.count] "
            f"[label]inputs[/label] [detail](subsample={subsample})[/detail]"
        )

        model_template = _build_model(config)

        for strat in context.strategies:
            sd = load_state_dict(context.experiment_name, strat,
                                 output_dir=output_dir_str)
            if sd is None:
                console.print(
                    f"[warning.content]No final weights found for '{strat}'[/warning.content]"
                )
                continue

            # Load weights into a fresh copy
            import copy
            model = copy.deepcopy(model_template)
            _load_into_model(model, sd)
            model.eval()

            console.print(
                f"[label]Probing:[/label] [strategy]{strat}[/strategy]"
            )

            activations = _collect_activations(
                model, inputs, args.position, args.skip_logits,
            )

            for point_name, data in activations:
                self._plot_activation(
                    data, point_name, strat, b, a_values,
                    context, dpi, ext,
                )

    def _plot_activation(self, data, point_name, strategy, b, a_values,
                         context, dpi, ext):
        """Render one activation heatmap and save it."""
        console = OLConsole()
        n_inputs, dim = data.shape
        width = max(8, min(20, dim / 20))
        height = max(6, min(24, n_inputs / 200))

        fig, ax = plt.subplots(figsize=(width, height))
        plot_heatmap(ax, data, colorbar=True)

        human_name = _human_point_name(point_name)
        ax.set_title(f'{strategy}: {human_name}')
        ax.set_xlabel('Neuron index')
        ax.set_ylabel(f'Input a (b={b})')

        path = context.output_dir / f'{strategy}_act_{point_name}.{ext}'
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path), dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        console.print(f"[label]Saved:[/label] [path]{path}[/path]")
