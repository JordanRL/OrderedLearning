# Phases

> Learning phase detection via gradient dynamics and representation change.

**Type:** Observer
**Lifecycle Points:** SNAPSHOT
**Loop Compatibility:** epoch

## What It Does

The phases hook detects qualitative changes in the learning process by tracking gradient velocity (rate of change of gradient norms), gradient acceleration (rate of change of velocity), and embedding representation change. These signals are combined into a phase code that classifies the current training regime.

In grokking experiments, training passes through distinct phases: initial memorization (rapid loss decrease, high gradient activity), a plateau (low gradient activity, no generalization), and delayed generalization (renewed gradient activity, accuracy increase on validation). This hook provides quantitative markers for these transitions.

Fires at SNAPSHOT intervals rather than every epoch to reduce noise and computational cost for the derivative calculations.

## Computational Cost

Low. Computes gradient norms and embedding distances at snapshot intervals. Stores a small history of previous values for derivative estimation. No additional forward or backward passes.

## Assumptions and Compatibility

- Requires gradients to be populated on model parameters
- Requires a model with an embedding layer (accesses embedding weights directly)
- Epoch-loop only
- Returns meaningful derivatives only after sufficient snapshots have accumulated

## Metrics

### `grad_velocity`
- **Formula:** Finite difference of gradient norm between consecutive snapshots: `||g_t|| - ||g_{t-1}||`
- **Range:** (-inf, +inf)
- **Interpretation:** Positive values indicate increasing gradient magnitude (network entering an active learning phase). Negative values indicate decreasing gradients (convergence or plateau). Sharp transitions mark phase boundaries.

### `grad_acceleration`
- **Formula:** Finite difference of gradient velocity: `v_t - v_{t-1}`
- **Range:** (-inf, +inf)
- **Interpretation:** Positive acceleration means gradient growth is accelerating. Sign changes in acceleration often precede phase transitions.

### `embedding_change`
- **Formula:** L2 distance between current and previous snapshot embedding weights: `||E_t - E_{t-1}||_2`
- **Range:** [0, +inf)
- **Interpretation:** How much the embedding representation changed since the last snapshot. Large changes indicate active representation learning; near-zero indicates stable representations.

### `embedding_change_normalized`
- **Formula:** `embedding_change / ||E_t||_2`
- **Range:** [0, +inf)
- **Interpretation:** Embedding change relative to the current embedding scale. Normalizes for the overall magnitude of embeddings, making comparisons across training stages more meaningful.

### `phase_code`
- **Formula:** Integer encoding of the current phase based on gradient velocity sign, acceleration sign, and embedding change magnitude
- **Range:** Integer
- **Interpretation:** A compact encoding of the current learning regime. Specific code values correspond to phases like "active memorization," "plateau," "generalization onset," etc. Transitions in the phase code mark phase boundaries.
