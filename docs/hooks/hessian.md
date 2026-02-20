# Hessian

> Per-step entanglement term (H_B · g_A) via finite-difference Hessian-vector product.

**Type:** Intervention
**Lifecycle Points:** POST_STEP
**Loop Compatibility:** both

## What It Does

The hessian hook measures the **entanglement term** — the component of the gradient that arises from interactions between different data batches mediated by the loss landscape's curvature. Specifically, it estimates `H_B · g_A`, where `H_B` is the Hessian of the loss on the current batch and `g_A` is the gradient from a previous batch.

This term captures how the curvature of one batch's loss surface deflects the gradient from another batch. In ordered data, this deflection can be systematic (always pushing in a consistent direction), while in random data it tends to cancel out. The entanglement term is the key theoretical quantity linking data ordering to learning dynamics.

The Hessian-vector product is estimated via finite differences: `(∇L(θ + εg) - ∇L(θ)) / ε`, avoiding explicit Hessian computation.

Uses a burst schedule to limit computational cost: fires for `burst_length` consecutive steps every `every_n_steps` steps.

## Computational Cost

**High.** Each firing requires one additional forward-backward pass (for the perturbed gradient). With default settings (every_n_steps=1000, burst_length=10), this adds ~1% overhead to total training time for long runs.

## Assumptions and Compatibility

- Works in both epoch and step loops
- `needs_reference_weights=True` — uses reference solution for alignment metrics
- Intervention hook: temporarily perturbs model parameters, then restores them
- Requires access to optimizer state (gradient from previous step)

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `epsilon` | `1e-4` | Finite difference step size for Hv approximation |
| `every_n_steps` | `1000` | Base interval between burst windows |
| `burst_length` | `10` | Number of consecutive steps to fire within each burst |

## Metrics

### `entanglement_norm`
- **Formula:** `||H_B · g_A||_2` — L2 norm of the entanglement term
- **Range:** [0, +inf)
- **Interpretation:** Magnitude of the cross-batch interaction. Large values indicate strong entanglement between successive batches' loss landscapes.

### `content_norm`
- **Formula:** `||g_B||_2` — L2 norm of the current batch's gradient (the "content" term)
- **Range:** [0, +inf)
- **Interpretation:** Magnitude of the direct gradient from the current batch, for comparison with the entanglement term.

### `observed_grad_norm`
- **Formula:** `||g_observed||_2` — L2 norm of the actual gradient used for the update
- **Range:** [0, +inf)
- **Interpretation:** Total gradient magnitude, which includes both content and entanglement contributions.

### `entanglement_energy_ratio`
- **Formula:** `||H_B · g_A||² / (||H_B · g_A||² + ||g_B||²)`
- **Range:** [0, 1]
- **Interpretation:** Fraction of total gradient energy coming from the entanglement term. Values near 0 mean entanglement is negligible; values near 0.5 mean entanglement contributes equally to content.

### `entanglement_content_cossim`
- **Formula:** `cos(H_B · g_A, g_B)` — cosine similarity between entanglement and content terms
- **Range:** [-1, 1]
- **Interpretation:** Whether entanglement reinforces (+1) or opposes (-1) the content gradient. Systematic positive alignment under ordered data (vs. near-zero under random) is evidence of ordering-mediated learning.

### `rayleigh_quotient`
- **Formula:** `g_A^T H_B g_A / ||g_A||²` — estimated from the finite difference
- **Range:** (-inf, +inf)
- **Interpretation:** Curvature of the loss along the previous gradient direction. Positive = convex (gradient is stabilizing); negative = concave (gradient is destabilizing). Related to edge-of-stability dynamics.

### `amplification_ratio`
- **Formula:** `||H_B · g_A|| / ||g_A||` — how much the Hessian amplifies the previous gradient
- **Range:** [0, +inf)
- **Interpretation:** Spectral amplification factor. Values > 1 mean the curvature amplifies the gradient; values < 1 mean it attenuates.

### `edge_of_stability`
- **Formula:** Estimated largest eigenvalue of the Hessian from the Rayleigh quotient
- **Range:** [0, +inf)
- **Interpretation:** Proximity to the edge of stability (where the learning rate × max eigenvalue ≈ 2). Tracks whether training is in the stable or unstable regime.

### `entanglement_cossim_to_solution`
- **Formula:** Cosine similarity between the entanglement term and the solution direction
- **Range:** [-1, 1]
- **Interpretation:** Whether the entanglement term pushes toward (+1) or away from (-1) the known solution. Key metric for evaluating ordering effects.

### `content_cossim_to_solution`
- **Formula:** Cosine similarity between the content gradient and the solution direction
- **Range:** [-1, 1]
- **Interpretation:** Solution alignment of the direct gradient component.

### `entanglement_coherence`
- **Formula:** Running average of `entanglement_content_cossim` over recent steps
- **Range:** [-1, 1]
- **Interpretation:** Smoothed measure of whether entanglement consistently reinforces content. Persistent positive coherence under ordered data (vs. near-zero under random) is the central experimental signal.
