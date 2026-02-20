# Adam Dynamics

> Adam optimizer state dynamics and solution alignment.

**Type:** Intervention
**Lifecycle Points:** POST_STEP
**Loop Compatibility:** both

## What It Does

The Adam dynamics hook inspects the internal state of the Adam optimizer — its first moment (momentum) and second moment (variance) estimates — and analyzes how they relate to the raw gradient, the optimizer update, and the direction toward a known solution.

Adam's momentum accumulates an exponential moving average of gradients, which can either amplify or attenuate the ordering signal in the raw gradients. This hook measures whether Adam's momentum is more or less aligned with the solution than the raw gradient, quantifying the optimizer's effect on ordering-mediated learning.

Uses a burst schedule to limit computational cost: fires for `burst_length` consecutive steps every `every_n_steps` steps.

## Computational Cost

Moderate. Accesses optimizer state tensors (already in memory) and computes dot products. No additional forward or backward passes. The burst schedule limits total overhead.

## Assumptions and Compatibility

- Requires an Adam or AdamW optimizer with accessible state dicts
- `needs_reference_weights=True` — requires reference solution for alignment metrics
- Works in both epoch and step loops
- Intervention hook: reads optimizer internal state (does not modify it)

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `every_n_steps` | `500` | Base interval between burst windows |
| `burst_length` | `10` | Number of consecutive steps to fire within each burst |

## Metrics

### `momentum_grad_cossim`
- **Formula:** Cosine similarity between Adam's first moment (momentum) and the current raw gradient
- **Range:** [-1, 1]
- **Interpretation:** How aligned the momentum is with the current gradient. High values mean the momentum is tracking the gradient closely; low values mean the momentum has accumulated a different direction from historical gradients.

### `amplification_ratio`
- **Formula:** `||update|| / ||gradient||` — ratio of the Adam update norm to the raw gradient norm
- **Range:** [0, +inf)
- **Interpretation:** Whether Adam amplifies (>1) or attenuates (<1) the gradient signal. Adam's adaptive learning rates can selectively amplify certain parameter directions.

### `update_deflection`
- **Formula:** `1 - cos(update, gradient)` — how much Adam deflects the update away from the raw gradient direction
- **Range:** [0, 2]
- **Interpretation:** 0 means the update is in the same direction as the gradient. 2 means the update is in the opposite direction. Measures how much Adam's momentum and adaptive scaling alter the gradient direction.

### `effective_lr_cv`
- **Formula:** Coefficient of variation of the effective per-parameter learning rates: `std(lr_eff) / mean(lr_eff)`
- **Range:** [0, +inf)
- **Interpretation:** How much Adam's per-parameter learning rates vary. High CV means some parameters have much larger effective learning rates than others.

### `momentum_solution_cossim`
- **Formula:** Cosine similarity between Adam's momentum and the solution direction
- **Range:** [-1, 1]
- **Interpretation:** Whether Adam's accumulated momentum points toward the solution. Compare with `grad_solution_cossim` to see if momentum improves or degrades solution alignment.

### `update_solution_cossim`
- **Formula:** Cosine similarity between the Adam update and the solution direction
- **Range:** [-1, 1]
- **Interpretation:** Whether the actual parameter update points toward the solution.

### `grad_solution_cossim`
- **Formula:** Cosine similarity between the raw gradient and the solution direction
- **Range:** [-1, 1]
- **Interpretation:** Baseline: how well the raw gradient points toward the solution, before Adam's modifications.

### `optimizer_solution_amplification`
- **Formula:** `update_solution_cossim / grad_solution_cossim` (when grad alignment is nonzero)
- **Range:** (-inf, +inf)
- **Interpretation:** Whether Adam amplifies (>1) or attenuates (<1) the solution-aligned component of the gradient. Values > 1 mean Adam is helping; values < 1 mean Adam is hurting solution alignment.

### `momentum_probe_cossim` / `update_probe_cossim` / `grad_probe_cossim`
- **Formula:** Same alignment metrics as above but computed against a probe direction (if available)
- **Range:** [-1, 1]
- **Interpretation:** Alignment with an alternative reference direction for comparison.

### `optimizer_probe_amplification`
- **Formula:** `update_probe_cossim / grad_probe_cossim`
- **Range:** (-inf, +inf)
- **Interpretation:** Adam's amplification of the probe-aligned gradient component.
