# Gradient Projection

> Gradient and displacement projection onto the known solution direction.

**Type:** Observer
**Lifecycle Points:** POST_EPOCH
**Loop Compatibility:** epoch

## What It Does

The gradient projection hook measures how well the current gradient and the cumulative weight displacement are aligned with the direction toward a known reference solution. For each layer and overall, it computes cosine similarity between the gradient (or displacement) and the vector pointing from current weights to reference weights.

This provides a direct measure of whether training is "on track" — whether the gradient is pointing toward the solution and whether the accumulated weight changes have moved the model closer to the solution. Comparing these metrics across data ordering strategies reveals whether certain orderings produce more solution-aligned gradients.

## Computational Cost

Low. Computes dot products and norms between gradient/weight vectors and stored reference vectors. No additional forward or backward passes. Requires loading reference weights once at initialization.

## Assumptions and Compatibility

- `needs_reference_weights=True` — requires a reference solution checkpoint (a fully-trained model that achieved generalization)
- Epoch-loop only
- Reference weights must match the current model architecture
- Stores initial weights to compute cumulative displacement

## Metrics

### `grad_cossim_to_solution/{layer}`
- **Formula:** `cos(∇W_layer, W_ref_layer - W_layer)` — cosine similarity between the layer's gradient and the direction to the reference solution for that layer
- **Range:** [-1, 1]
- **Interpretation:** +1 means the gradient is pointing directly toward the solution. -1 means it points directly away. 0 means it's orthogonal to the solution direction.

### `disp_cossim_to_solution/{layer}`
- **Formula:** `cos(W_layer - W_init_layer, W_ref_layer - W_init_layer)` — cosine similarity between the cumulative displacement from initialization and the direction from initialization to the reference solution
- **Range:** [-1, 1]
- **Interpretation:** How well the total parameter movement aligns with the path from initialization to the solution. High values mean training has been moving in the right direction overall.

### `overall_grad_cossim_to_solution`
- **Formula:** Cosine similarity between the full concatenated gradient vector and the full solution direction vector
- **Range:** [-1, 1]
- **Interpretation:** Global gradient-solution alignment, weighting all parameters equally. The single most important metric for assessing whether the current gradient step will move toward the solution.

### `overall_disp_cossim_to_solution`
- **Formula:** Cosine similarity between the full concatenated displacement and the full solution direction
- **Range:** [-1, 1]
- **Interpretation:** Global displacement-solution alignment. Tracks whether the model's trajectory through parameter space is heading toward the solution.

### `mean_layer_grad_cossim_to_solution`
- **Formula:** Mean of per-layer `grad_cossim_to_solution` values
- **Range:** [-1, 1]
- **Interpretation:** Average per-layer gradient alignment, giving equal weight to each layer regardless of parameter count.

### `mean_layer_disp_cossim_to_solution`
- **Formula:** Mean of per-layer `disp_cossim_to_solution` values
- **Range:** [-1, 1]
- **Interpretation:** Average per-layer displacement alignment.

### `displacement_norm`
- **Formula:** `||W - W_init||_2` — L2 norm of total parameter displacement from initialization
- **Range:** [0, +inf)
- **Interpretation:** How far the model has moved from its initial weights. Compare with `distance_to_reference` to assess progress.

### `distance_to_reference`
- **Formula:** `||W - W_ref||_2` — L2 distance from current weights to the reference solution
- **Range:** [0, +inf)
- **Interpretation:** Remaining distance to the solution. Should decrease if training is progressing toward generalization.
