# Path Length

> Cumulative path length and displacement in parameter space.

**Type:** Observer
**Lifecycle Points:** POST_STEP, POST_EPOCH (epoch loop); POST_STEP, SNAPSHOT (step loop)
**Loop Compatibility:** both

## What It Does

The path length hook tracks the trajectory of the model through parameter space by measuring both the **cumulative path length** (total distance traveled through parameter space) and the **net displacement** (straight-line distance from the starting point). The ratio of these gives the **path efficiency** — how direct the optimization trajectory is.

A highly efficient path (efficiency near 1) means the optimizer moves in a nearly straight line toward its destination. A low-efficiency path means the optimizer wanders, potentially revisiting or circling through parameter space. Data ordering strategies that produce more efficient paths may lead to faster convergence.

## Computational Cost

Negligible. Stores the previous and initial parameter vectors. Computes norms at each step/epoch. No additional forward or backward passes.

## Assumptions and Compatibility

- Works in both epoch and step loops
- Accumulates path length from the first firing; initial parameters are stored on first call
- Memory: two copies of the parameter vector (initial and previous)

## Metrics

### `path_length`
- **Formula:** `Σ ||θ_t - θ_{t-1}||_2` — cumulative sum of step-wise parameter changes
- **Range:** [0, +inf)
- **Interpretation:** Total distance the model has traveled through parameter space. Monotonically increasing. Longer paths indicate more optimization work.

### `net_displacement`
- **Formula:** `||θ_t - θ_0||_2` — straight-line distance from initial parameters to current parameters
- **Range:** [0, +inf)
- **Interpretation:** How far the model has moved from initialization in a straight line. Compare with path_length to assess trajectory efficiency.

### `path_efficiency`
- **Formula:** `net_displacement / path_length`
- **Range:** [0, 1]
- **Interpretation:** 1.0 means the optimizer has moved in a perfectly straight line (every step moved directly away from initialization). Values near 0 mean extensive wandering. Typically starts near 1 and decreases as training progresses and the trajectory curves.
