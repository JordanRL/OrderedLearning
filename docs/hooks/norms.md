# Norms

> Gradient magnitude dynamics per layer and overall.

**Type:** Observer
**Lifecycle Points:** POST_EPOCH
**Loop Compatibility:** epoch

## What It Does

The norms hook computes gradient magnitude metrics after each training epoch. It captures the L2 norm of the full gradient vector, the maximum and mean gradient components across all parameters, and per-layer gradient norms.

These metrics track how gradient magnitudes evolve during training. In grokking experiments, gradient norms often exhibit distinct phases: large norms during memorization, a quiet period, then renewed gradient activity during generalization. Per-layer breakdowns reveal which parts of the network are actively learning at each stage.

## Computational Cost

Negligible. Computes norms from gradients already present on the model after the backward pass. No additional forward or backward passes required.

## Assumptions and Compatibility

- Requires gradients to be populated on model parameters (runs after backward pass)
- Epoch-loop only (not available in step-loop experiments)

## Metrics

### `total_norm`
- **Formula:** `||g||_2` — L2 norm of the concatenated gradient vector across all parameters
- **Range:** [0, +inf)
- **Interpretation:** Overall gradient magnitude. Sudden increases may signal phase transitions; steady decay indicates convergence.

### `max_component`
- **Formula:** `max(|g_i|)` — maximum absolute gradient component
- **Range:** [0, +inf)
- **Interpretation:** Detects gradient spikes in individual parameters. Large values relative to `mean_component` indicate concentrated gradient activity.

### `mean_component`
- **Formula:** `mean(|g_i|)` — mean absolute gradient component
- **Range:** [0, +inf)
- **Interpretation:** Average gradient activity across all parameters. Compare with `max_component` to assess gradient concentration.

### `norm_{layer}`
- **Formula:** `||g_layer||_2` — L2 norm of gradients for a specific named parameter
- **Range:** [0, +inf)
- **Interpretation:** Per-layer gradient magnitude. Reveals which layers are actively learning. Layer names follow PyTorch's `named_parameters()` convention.
