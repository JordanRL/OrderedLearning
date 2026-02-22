# Instrumentation Hooks

The hook system provides deep observability and controlled intervention during training. Hooks fire at lifecycle points, compute metrics from training state, and dispatch results to configurable sinks.

## How Hooks Work

### Lifecycle Points

Training hooks fire at five lifecycle points:

| Point | When | Typical use |
|---|---|---|
| `PRE_EPOCH` | Before each training epoch | Setup, state capture |
| `POST_EPOCH` | After each training epoch | Gradient analysis, weight tracking |
| `PRE_STEP` | Before each training step | State capture for multi-step hooks |
| `POST_STEP` | After each training step | Batch-level dynamics, entanglement |
| `SNAPSHOT` | At `snapshot_every` intervals | Periodic expensive analyses |

### Hook Types

- **Observer** (`TrainingHook`): Read-only hooks that compute metrics from training state without modifying it. Most hooks are observers.
- **Intervention** (`InterventionHook`): Hooks that temporarily modify training state (e.g., running counterfactual epochs, perturbing weights) via a controlled API (`ModelDataContext`). State is always restored after intervention.
- **Debug Intervention** (`DebugInterventionHook`): Intervention hooks excluded from bulk selection (`all`, `observers`). Only activated when explicitly named or via `with_debug`.

### Sinks

Metrics from hooks are dispatched to one or more sinks:

| Sink | Flag | Description |
|---|---|---|
| Console | *(always)* | Display in terminal (table in NORMAL mode, sidebar column in LIVE mode) |
| CSV | `--hook-csv` | Per-strategy CSV file in the experiment output directory |
| JSONL | `--hook-jsonl` | Per-strategy JSONL file in the experiment output directory |
| W&B | `--hook-wandb PROJECT` | Weights & Biases logging with automatic grouping |

CSV and JSONL sinks auto-generate filenames (`{strategy}.csv`, `{strategy}.jsonl`) in the experiment output directory.

## Enabling Hooks

### Experiment-curated groups (`--with-hooks`)

Each experiment defines named hook groups in its `hook_sets` dict:

```bash
python run_experiment.py mod_arithmetic --with-hooks full
python run_experiment.py mod_arithmetic --with-hooks observers
python run_experiment.py mod_arithmetic --with-hooks minimal
```

Groups for `mod_arithmetic`: `none`, `minimal`, `observers`, `interventions`, `full`.

### Individual hooks (`--hooks`)

Select specific hooks by name:

```bash
python run_experiment.py mod_arithmetic --hooks norms fourier attention
```

Special keywords:
- `all` -- all non-debug hooks
- `observers` -- all observer (non-intervention) hooks
- `with_debug` -- include debug hooks

`--hooks` is additive with `--with-hooks`:

```bash
# Start with observers group, add hessian intervention
python run_experiment.py mod_arithmetic --with-hooks observers --hooks hessian
```

### Per-hook configuration (`--hook-config`)

Override hook constructor parameters:

```bash
python run_experiment.py mod_arithmetic --hooks hessian \
    --hook-config hessian.epsilon=0.001 hessian.every_n_steps=500
```

### State offloading (`--hook-offload-state`)

Hooks with large tensor state (sliding windows, previous parameters) can offload to CPU between calls:

```bash
python run_experiment.py mod_arithmetic --with-hooks full --hook-offload-state
```

## Inspecting Hooks

### List available hooks

```bash
python run_experiment.py mod_arithmetic --hooks-list
```

### Describe hook metrics

```bash
# Describe a specific hook
python run_experiment.py mod_arithmetic --hooks-describe fourier

# Describe all hooks
python run_experiment.py mod_arithmetic --hooks-describe
```

### Profile hook performance

```bash
python run_experiment.py mod_arithmetic --with-hooks full --profile-hooks
```

This prints wall-time for each hook's compute/intervene call at each lifecycle point.

## Special Behaviors

### `training_metrics` auto-inclusion

When any hooks are enabled, `training_metrics` is automatically included. It echoes core training metrics (loss, accuracy, learning rate) into the hook metric stream so they appear in CSV/JSONL/W&B alongside hook metrics.

### Reference weights

Hooks with `needs_reference_weights=True` (gradient_projection, counterfactual, hessian, etc.) use a shared `ReferenceWeights` instance that loads a known-solution model checkpoint. This enables metrics like "gradient alignment to solution." The reference path is auto-resolved per strategy from the experiment output directory.

### Step schedules

Some hooks (hessian, adam_dynamics) use burst schedules to limit computational cost:

```
every_n_steps=1000, burst_length=11
→ fires at steps 1000-1010, 2000-2010, 3000-3010, ...
```

### Loop compatibility

Each hook declares which loop types it supports via `loop_points`. A hook that only supports `epoch` loops won't be instantiated in a `step`-loop experiment, and vice versa.

## Hook Summary

### Observers

| Hook | Lifecycle Points | Loop | Description | Doc |
|---|---|---|---|---|
| `training_metrics` | POST_STEP, POST_EPOCH | both | Core training metrics (loss, accuracy, LR) | [doc](hooks/training-metrics.md) |
| `norms` | POST_EPOCH | epoch | Gradient magnitude dynamics | [doc](hooks/norms.md) |
| `consecutive` | POST_EPOCH | epoch | Cosine similarity between consecutive epoch gradients | [doc](hooks/consecutive.md) |
| `token_gradient` | POST_EPOCH | epoch | Per-token gradient distribution | [doc](hooks/token-gradient.md) |
| `phases` | SNAPSHOT | epoch | Learning phase detection | [doc](hooks/phases.md) |
| `variance` | POST_EPOCH | epoch | Gradient variance/stability over sliding windows | [doc](hooks/variance.md) |
| `subspace_gradient_info` | POST_EPOCH | epoch | Gradient subspace dimensionality and information content | [doc](hooks/subspace-gradient-info.md) |
| `fourier` | POST_EPOCH | epoch | Fourier structure emergence in embeddings, decoder, and MLP | [doc](hooks/fourier.md) |
| `attention` | POST_EPOCH | both | Attention weight structure (SVD) and attention patterns | [doc](hooks/attention.md) |
| `weight_tracking` | POST_EPOCH | epoch | Weight norms, spectral properties, effective rank | [doc](hooks/weight-tracking.md) |
| `gradient_projection` | POST_EPOCH | epoch | Gradient & displacement projection onto known solution | [doc](hooks/gradient-projection.md) |
| `parameter_delta` | POST_EPOCH, SNAPSHOT | both | Parameter update magnitude tracking | [doc](hooks/parameter-delta.md) |
| `path_length` | POST_STEP, POST_EPOCH, SNAPSHOT | both | Cumulative path length and displacement in parameter space | [doc](hooks/path-length.md) |
| `batch_dynamics` | POST_STEP, POST_EPOCH, SNAPSHOT | both | Batch-level gradient dynamics (autocorrelation, efficiency, rank) | [doc](hooks/batch-dynamics.md) |
| `training_diagnostics` | POST_STEP, POST_EPOCH, SNAPSHOT | both | Standard training diagnostics (loss volatility, gradient noise) | [doc](hooks/training-diagnostics.md) |

### Interventions

| Hook | Lifecycle Points | Loop | Description | Doc |
|---|---|---|---|---|
| `counterfactual` | POST_EPOCH, SNAPSHOT | epoch | Counterfactual ordering analysis via shuffled epochs | [doc](hooks/counterfactual.md) |
| `hessian` | PRE_STEP, POST_STEP | epoch | Per-step entanglement term (H_B * g_A) via finite-difference | [doc](hooks/hessian.md) |
| `adam_dynamics` | POST_STEP | both | Adam optimizer state dynamics | [doc](hooks/adam-dynamics.md) |
### Debug Interventions

| Hook | Lifecycle Points | Loop | Description | Doc |
|---|---|---|---|---|
| `counterfactual_validator` | POST_EPOCH, SNAPSHOT | epoch | Validate K-sufficiency of counterfactual decomposition | [doc](hooks/counterfactual-validator.md) |
