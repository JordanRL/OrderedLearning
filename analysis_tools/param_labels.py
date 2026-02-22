"""Human-readable parameter labels and forward-pass ordering.

Converts raw PyTorch parameter names (from named_parameters() or
state_dict keys) to display labels and provides sort keys that follow
the order a token progresses through the model.

Supports two naming conventions:
- nn.TransformerEncoder (GrokkingTransformer): transformer.layers.N.*
- GPT-2 / HuggingFace: transformer.h.N.*, transformer.wte.*, etc.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Pattern table: (regex, label_fn, order_fn)
#
#   label_fn(match) -> str         human-readable label
#   order_fn(match) -> tuple       sort key (lower = earlier in forward pass)
#
# Patterns are tried top-to-bottom; first match wins.
# ---------------------------------------------------------------------------

def _bias_suffix(match_group: str) -> str:
    return ' (bias)' if match_group == 'bias' else ''

def _bias_ord(match_group: str) -> int:
    return 1 if match_group == 'bias' else 0


_PATTERNS: list[tuple[re.Pattern, ...]] = []


def _p(pattern, label_fn, order_fn):
    _PATTERNS.append((re.compile(pattern), label_fn, order_fn))


# ── Embeddings ────────────────────────────────────────────────────────────

# GrokkingTransformer
_p(r'^embedding\.weight$',
   lambda m: 'Token Embeddings',
   lambda m: (0, 0, 0))

_p(r'^pos_embedding$',
   lambda m: 'Positional Embeddings',
   lambda m: (0, 1, 0))

# GPT-2 / HuggingFace
_p(r'^transformer\.wte\.weight$',
   lambda m: 'Token Embeddings',
   lambda m: (0, 0, 0))

_p(r'^transformer\.wpe\.weight$',
   lambda m: 'Positional Embeddings',
   lambda m: (0, 1, 0))


# ── TransformerEncoder layers (GrokkingTransformer) ───────────────────────

_p(r'^transformer\.layers\.(\d+)\.norm1\.(weight|bias)$',
   lambda m: f'Layer {m.group(1)} Attn Norm{_bias_suffix(m.group(2))}',
   lambda m: (1, int(m.group(1)), 0, _bias_ord(m.group(2))))

_p(r'^transformer\.layers\.(\d+)\.self_attn\.in_proj_(weight|bias)$',
   lambda m: f'Layer {m.group(1)} Attn In-Proj{_bias_suffix(m.group(2))}',
   lambda m: (1, int(m.group(1)), 1, _bias_ord(m.group(2))))

_p(r'^transformer\.layers\.(\d+)\.self_attn\.out_proj\.(weight|bias)$',
   lambda m: f'Layer {m.group(1)} Attn Out-Proj{_bias_suffix(m.group(2))}',
   lambda m: (1, int(m.group(1)), 2, _bias_ord(m.group(2))))

_p(r'^transformer\.layers\.(\d+)\.norm2\.(weight|bias)$',
   lambda m: f'Layer {m.group(1)} FFN Norm{_bias_suffix(m.group(2))}',
   lambda m: (1, int(m.group(1)), 3, _bias_ord(m.group(2))))

_p(r'^transformer\.layers\.(\d+)\.linear1\.(weight|bias)$',
   lambda m: f'Layer {m.group(1)} FFN Up{_bias_suffix(m.group(2))}',
   lambda m: (1, int(m.group(1)), 4, _bias_ord(m.group(2))))

_p(r'^transformer\.layers\.(\d+)\.linear2\.(weight|bias)$',
   lambda m: f'Layer {m.group(1)} FFN Down{_bias_suffix(m.group(2))}',
   lambda m: (1, int(m.group(1)), 5, _bias_ord(m.group(2))))


# ── GPT-2 layers ─────────────────────────────────────────────────────────

_p(r'^transformer\.h\.(\d+)\.ln_1\.(weight|bias)$',
   lambda m: f'Layer {m.group(1)} Attn Norm{_bias_suffix(m.group(2))}',
   lambda m: (1, int(m.group(1)), 0, _bias_ord(m.group(2))))

_p(r'^transformer\.h\.(\d+)\.attn\.c_attn\.(weight|bias)$',
   lambda m: f'Layer {m.group(1)} Attn QKV{_bias_suffix(m.group(2))}',
   lambda m: (1, int(m.group(1)), 1, _bias_ord(m.group(2))))

_p(r'^transformer\.h\.(\d+)\.attn\.c_proj\.(weight|bias)$',
   lambda m: f'Layer {m.group(1)} Attn Out-Proj{_bias_suffix(m.group(2))}',
   lambda m: (1, int(m.group(1)), 2, _bias_ord(m.group(2))))

_p(r'^transformer\.h\.(\d+)\.ln_2\.(weight|bias)$',
   lambda m: f'Layer {m.group(1)} FFN Norm{_bias_suffix(m.group(2))}',
   lambda m: (1, int(m.group(1)), 3, _bias_ord(m.group(2))))

_p(r'^transformer\.h\.(\d+)\.mlp\.c_fc\.(weight|bias)$',
   lambda m: f'Layer {m.group(1)} FFN Up{_bias_suffix(m.group(2))}',
   lambda m: (1, int(m.group(1)), 4, _bias_ord(m.group(2))))

_p(r'^transformer\.h\.(\d+)\.mlp\.c_proj\.(weight|bias)$',
   lambda m: f'Layer {m.group(1)} FFN Down{_bias_suffix(m.group(2))}',
   lambda m: (1, int(m.group(1)), 5, _bias_ord(m.group(2))))


# ── Final norm / output head ─────────────────────────────────────────────

# GPT-2 final layer norm
_p(r'^transformer\.ln_f\.(weight|bias)$',
   lambda m: f'Final Layer Norm{_bias_suffix(m.group(1))}',
   lambda m: (2, 0, 0, _bias_ord(m.group(1))))

# Output projections
_p(r'^decoder\.(weight|bias)$',
   lambda m: f'Output Head{_bias_suffix(m.group(1))}',
   lambda m: (3, 0, 0, _bias_ord(m.group(1))))

_p(r'^lm_head\.weight$',
   lambda m: 'Output Head',
   lambda m: (3, 0, 0, 0))


# ── Aggregated group labels (from layer_dynamics _aggregate_by_layer) ────

_AGGREGATE_LABELS = {
    'embed': 'Embeddings',
    'ln_f': 'Final Layer Norm',
    'other': 'Other',
}

_AGGREGATE_ORDER = {
    'embed': (0, 0),
    'ln_f': (2, 0),
    'other': (3, 0),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _strip_compile_prefix(name: str) -> str:
    """Strip torch.compile ``_orig_mod.`` prefix(es) from parameter names."""
    while name.startswith('_orig_mod.'):
        name = name[len('_orig_mod.'):]
    return name


def label_param(name: str) -> str:
    """Convert a PyTorch parameter name to a human-readable label.

    Strips torch.compile prefixes before matching.
    Falls back to the raw name if no pattern matches.
    """
    name = _strip_compile_prefix(name)
    for pattern, label_fn, _ in _PATTERNS:
        m = pattern.match(name)
        if m:
            return label_fn(m)
    return name


def param_sort_key(name: str) -> tuple:
    """Return a sort key that orders parameters in forward-pass order.

    Strips torch.compile prefixes before matching.
    Embeddings first, then layer-by-layer in component order,
    then final norm, then output head. Unrecognized parameters
    sort to the end.
    """
    name = _strip_compile_prefix(name)
    for pattern, _, order_fn in _PATTERNS:
        m = pattern.match(name)
        if m:
            return order_fn(m)
    return (99, 0, 0, 0)


def label_aggregate(group_name: str) -> str:
    """Convert an aggregated layer group name to a human-readable label.

    Handles group names produced by layer_dynamics._aggregate_by_layer:
    'embed', 'h.0', 'h.1', 'ln_f', 'other'.
    """
    if group_name in _AGGREGATE_LABELS:
        return _AGGREGATE_LABELS[group_name]
    m = re.match(r'^h\.(\d+)$', group_name)
    if m:
        return f'Layer {m.group(1)}'
    # TransformerEncoder: layers.N
    m = re.match(r'^layers\.(\d+)$', group_name)
    if m:
        return f'Layer {m.group(1)}'
    return group_name


def aggregate_sort_key(group_name: str) -> tuple:
    """Return a sort key for aggregated layer group names."""
    if group_name in _AGGREGATE_ORDER:
        return _AGGREGATE_ORDER[group_name]
    m = re.match(r'^h\.(\d+)$', group_name)
    if m:
        return (1, int(m.group(1)))
    m = re.match(r'^layers\.(\d+)$', group_name)
    if m:
        return (1, int(m.group(1)))
    return (99, 0)
