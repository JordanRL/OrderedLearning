# Framework Architecture

This page explains the design of the OrderedLearning hook system — the
observer/intervention distinction, the SAPI pattern, the sink pipeline, and the
safety guarantees that hold it all together. It is intended for someone
evaluating whether the framework fits their needs, or for a contributor who
wants to understand the system before reading implementation code.

For hook usage (CLI flags, per-hook configuration, output sinks), see
[instrumentation-hooks.md](instrumentation-hooks.md). For individual hook
references, see [docs/hooks/](hooks/).

---

## Design Goals

1. **Experiments never write the training loop.** Training loops live in
   `framework/loops.py`. Experiments implement lifecycle building blocks; the
   loop calls them.
2. **Hooks are decoupled from the loop.** Analysis code never modifies training
   control flow. Hooks observe or intervene through well-defined interfaces.
3. **Interventions cannot corrupt training state.** The framework enforces this
   with guardian checkpoints and RNG isolation — even a buggy intervention hook
   cannot silently alter the training trajectory.
4. **Metrics flow through a uniform pipeline.** Every hook produces namespaced
   key-value metrics. Sinks consume them. Adding a new output destination
   (CSV, W&B, a custom dashboard) never requires changing hook code.

---

## Hook Lifecycle

### HookPoints

Five lifecycle points are defined in the `HookPoint` enum:

| HookPoint | When it fires | Available in |
|---|---|---|
| `PRE_EPOCH` | Before the epoch's batch loop starts | epoch loop |
| `PRE_STEP` | Before `strategy.train_step()` | epoch loop |
| `POST_STEP` | After `strategy.train_step()` | both loops |
| `POST_EPOCH` | After all batches, evaluation, and scheduler step | epoch loop |
| `SNAPSHOT` | Every `config.snapshot_every` epochs/steps | both loops |

### Firing order within one epoch (epoch loop)

```
PRE_EPOCH
  for each batch:
      PRE_STEP
      train_step()
      POST_STEP
  flush_step_metrics()
  scheduler.step()
  evaluate()
POST_EPOCH
SNAPSHOT          (if this epoch matches snapshot_every)
```

The step loop is simpler — it only fires `POST_STEP` after each training step
and `SNAPSHOT` at the configured interval.

### Loop-type filtering

Each hook can declare a `loop_points` dict to restrict which HookPoints it uses
in each loop type. For example, `TrainingMetricsHook` fires at `POST_EPOCH` in
the epoch loop but at `POST_STEP` in the step loop:

```python
hook_points = {HookPoint.POST_STEP, HookPoint.POST_EPOCH}
loop_points = {
    'epoch': {HookPoint.POST_EPOCH},
    'step':  {HookPoint.POST_STEP},
}
```

If a hook has `loop_points` but no entry for the current loop type, the hook is
excluded entirely. Hooks without `loop_points` use their static `hook_points`
in all loop types.

### Epoch-gated firing

`loop_points` values can also be dicts mapping a HookPoint to an
`(min_epoch, max_epoch)` range, restricting the hook to a window of training:

```python
loop_points = {
    'epoch': {
        HookPoint.POST_EPOCH: (None, 49),   # fire epochs 0-49
        HookPoint.SNAPSHOT:   (50, None),    # fire epochs 50+
    },
}
```

### Step schedules

Hooks that fire at `PRE_STEP` or `POST_STEP` can use a `StepSchedule` to
control *which* steps they fire on, independent of the epoch-level gating:

| Mode | Behavior |
|---|---|
| `continual` | Fire every step (default) |
| `stride` | Fire every *n*th step |
| `burst` | Fire *k* consecutive steps every *n* steps |

A `warmup` parameter skips all steps below a threshold. For example, the
Hessian hook uses burst mode to fire 11 consecutive steps every 1000 steps,
starting after a warmup period.

---

## Observers and Interventions

The hook system draws a line between hooks that *read* training state and hooks
that *modify* it — but that line is drawn per-HookPoint, not per-hook.

### Observers (`TrainingHook`)

An observer implements a single method:

```python
def compute(self, ctx: RunDataContext) -> dict[str, Any]:
```

It receives a frozen `RunDataContext` — a read-only snapshot of the current
training state — and returns a dict of metric names to scalar values. Observers
cannot modify the model, optimizer, or any other training state. Most hooks are
observers.

**Examples:** `NormsHook` (gradient magnitudes), `FourierHook` (spectral
structure), `TrainingMetricsHook` (loss, accuracy, learning rate).

### Interventions (`InterventionHook`)

An intervention implements:

```python
def intervene(self, run_ctx: RunDataContext, model_ctx: ModelDataContext) -> dict[str, Any]:
```

It receives both the frozen `RunDataContext` *and* a mutable
`ModelDataContext` — the SAPI (see next section). Interventions can save and
restore checkpoints, run extra training epochs, compute gradients on alternate
data, and perturb model weights. The `HookManager` guarantees that all state
modifications are rolled back after the intervention completes.

**Examples:** `CounterfactualHook` (runs shuffled-data epochs to measure
ordering effects), `HessianHook` (finite-difference Hessian-vector products),
`AdamDynamicsHook` (inspects Adam optimizer internal state).

### Selective intervention with `intervention_points`

An `InterventionHook` does not have to intervene at every HookPoint it
registers for. The `intervention_points` attribute declares which specific
HookPoints need `intervene()` with `ModelDataContext`. At all other registered
HookPoints, `compute()` is called instead — the hook operates in observer mode.

```python
class MyHook(InterventionHook):
    hook_points = {HookPoint.POST_EPOCH, HookPoint.SNAPSHOT}
    intervention_points = {HookPoint.SNAPSHOT}  # only intervene at SNAPSHOT

    def compute(self, ctx):
        # Called at POST_EPOCH (observer mode)
        return {"lightweight_metric": some_value}

    def intervene(self, run_ctx, model_ctx):
        # Called at SNAPSHOT (intervention mode, with ModelDataContext)
        token = model_ctx.save_checkpoint(full=True)
        ...
```

When `intervention_points` is `None` (the default), all `hook_points` are
treated as intervention points — matching the behavior of a hook that always
needs `ModelDataContext`.

This matters for efficiency: the guardian checkpoint and `ModelDataContext`
construction are only triggered at HookPoints where at least one hook actually
needs `intervene()`. A hook that only needs to observe at most of its
HookPoints avoids that overhead at the observer-mode points.

### Debug interventions (`DebugInterventionHook`)

A subclass of `InterventionHook` with `debug = True`. Debug hooks are excluded
from bulk selection (`--with-hooks full`, `--hooks all`). They are only
activated when explicitly named or via `--hooks ... with_debug`. This prevents
expensive validation hooks from running in normal experiments.

### Dispatch order

Within a single `fire()` call, the `HookManager` partitions hooks at the
current HookPoint into two groups:

1. **Observer-mode hooks** — all `TrainingHook` instances, plus any
   `InterventionHook` instances whose `intervention_points` excludes the
   current HookPoint. These run first via `compute()`.
2. **Intervention-mode hooks** — `InterventionHook` instances whose
   `intervention_points` includes the current HookPoint (or is `None`). These
   run second via `intervene()`, wrapped in a guardian checkpoint.

This means intervention hooks can rely on observer metrics already being
available (via `HookManager.get_last_metrics()`).

---

## The SAPI: ModelDataContext

"SAPI" stands for Service API — a controlled interface that replaces direct
access to training internals. Instead of handing intervention hooks a raw model
and optimizer, the framework provides `ModelDataContext`, which exposes a
curated set of operations.

### Why a SAPI?

Giving hooks raw references to the model and optimizer is fragile. A hook might
forget to restore optimizer state, or step the scheduler, or leave stale
gradients on parameters. The SAPI centralizes these operations behind methods
that handle bookkeeping internally, and the guardian checkpoint (described
below) provides a backstop even if a hook misuses the API.

### Operations

**Checkpoint management** — save and restore training state:

| Method | Description |
|---|---|
| `save_checkpoint(full=True)` | Save training state; returns an opaque token. `full=True` saves model, optimizer, scheduler, and gradients to CPU. `full=False` saves only model parameters and RNG state on GPU (faster for forward-backward-only hooks). |
| `restore_checkpoint(token)` | Restore state from a saved token. |
| `discard_checkpoint(token)` | Free memory for a token that is no longer needed. |

**Pre-epoch state** — for hooks that need to rewind to before the current epoch:

| Method | Description |
|---|---|
| `restore_pre_epoch()` | Restore model and optimizer to the state captured before the training epoch ran. Only available when the hook sets `needs_pre_epoch_state = True`. |

**Gradient computation** — compute gradients without stepping the optimizer:

| Method | Description |
|---|---|
| `compute_batch_gradients(batch=None)` | Forward-backward on a single batch. Saves and restores `param.grad` internally so the caller's gradient state is not corrupted. |
| `compute_gradients()` | Forward-backward over the full training set (no optimizer step). |

**Training** — run a full epoch with or without optimizer steps:

| Method | Description |
|---|---|
| `run_training_epoch(loader, step=True)` | Run one epoch on the given loader, returning accumulated mean gradients. When `step=False`, gradients are computed but weights are not updated. |

**Weight perturbation** — for finite-difference approximations:

| Method | Description |
|---|---|
| `apply_perturbation(direction, scale)` | Apply `theta <- theta + scale * direction` to model weights. The caller is responsible for save/restore around this. |

**Data:**

| Method | Description |
|---|---|
| `get_shuffled_loader()` | Create a DataLoader with randomly permuted training data. |

**Read-only properties:** `model` (the `nn.Module`) and `device` (the
`torch.device`).

### Typical intervention pattern

A well-behaved intervention hook follows this pattern:

```python
def intervene(self, run_ctx, model_ctx):
    token = model_ctx.save_checkpoint(full=True)
    try:
        # ... do work (run epochs, perturb weights, etc.)
        metrics = {...}
    finally:
        model_ctx.restore_checkpoint(token)
        model_ctx.discard_checkpoint(token)
    return metrics
```

Even if the hook forgets to restore, the guardian checkpoint (next section) will
catch it.

---

## Context Objects: RunDataContext vs ModelDataContext

The two context objects embody the observer/intervention split:

| | RunDataContext | ModelDataContext |
|---|---|---|
| **Mutability** | Frozen dataclass (`frozen=True`) | Mutable class with methods |
| **Passed to** | All hooks via `compute()` | Hooks in intervention mode via `intervene()` |
| **Purpose** | Read-only snapshot of training state | Controlled operations on training state |

### RunDataContext fields

| Field | Type | Description |
|---|---|---|
| `hook_point` | `HookPoint` | Which lifecycle point this is |
| `epoch` | `int` | Current epoch |
| `step` | `int \| None` | Global step counter |
| `model` | `nn.Module \| None` | The model (for read-only inspection) |
| `config` | `object \| None` | Experiment configuration |
| `loader` | `DataLoader \| None` | Training data loader |
| `batch_idx` | `int \| None` | Current batch index within the epoch |
| `batch_data` | `Tensor \| None` | Current batch tensor |
| `loss` | `float \| None` | Epoch average loss or step loss |
| `train_acc` | `float \| None` | Training accuracy |
| `val_acc` | `float \| None` | Validation accuracy |
| `lr` | `float \| None` | Current learning rate |
| `accumulated_grads` | `dict \| None` | Mean gradients over the epoch (populated only when a hook sets `needs_grads = True`) |
| `prev_step_grads` | `dict \| None` | Previous step's gradients (populated only when a hook sets `needs_prev_step_grads = True`) |
| `target_grad` | `dict \| None` | Target gradient (experiment-specific) |
| `profiler` | `CheckpointProfiler \| None` | Profiler for timing sections |

Fields are `None` when they do not apply to the current hook point. For
example, `batch_data` is populated at `PRE_STEP` and `POST_STEP` but not at
`POST_EPOCH`. What gets populated is demand-driven: if no active hook declares
`needs_grads = True`, the loop skips gradient accumulation entirely.

---

## Safety Guarantees

### Guardian checkpoints

When at least one hook needs `intervene()` at the current HookPoint, the
`HookManager` saves a *guardian checkpoint* before running interventions — a
full snapshot of model weights, optimizer state, scheduler state, gradients, and
RNG state. After all interventions complete, the guardian is restored
unconditionally. This means:

- If an intervention hook forgets to restore its own checkpoint, training state
  is still correct.
- If an intervention hook raises an exception, the guardian restore runs in
  the cleanup path.
- Observers that ran before the interventions are unaffected — their metrics
  were already collected.

At HookPoints where no hook needs `intervene()` — either because only observer
hooks are present, or because all `InterventionHook` instances at that point
have it excluded from their `intervention_points` — no guardian checkpoint is
saved and no `ModelDataContext` is constructed.

### RNG isolation

`HookManager.fire()` saves CPU and CUDA RNG states before running any hooks
and restores them afterward. This ensures that random operations inside hooks
(e.g., `torch.svd_lowrank` in attention analysis, `torch.randperm` for data
shuffling) do not shift dropout masks or other training-related random state.
The training trajectory is identical whether hooks are enabled or not.

### Demand-driven context population

The loop only computes expensive context fields when an active hook actually
needs them. For example:

- Gradient accumulation (summing per-batch gradients over an epoch) only runs
  when a hook declares `needs_grads = True`.
- Previous-step gradient capture only runs when a hook declares
  `needs_prev_step_grads = True`.
- Pre-epoch state snapshots only happen when an intervention hook sets
  `needs_pre_epoch_state = True`.

This keeps the baseline loop overhead near zero when no hooks are enabled.

---

## The Sink Pipeline

Hooks produce metrics. Sinks consume them. The two are completely decoupled —
hooks do not know which sinks are active, and sinks do not know which hooks
produced the metrics they receive.

### How metrics flow

1. Each hook's `compute()` or `intervene()` returns a flat dict of
   `metric_name -> scalar_value`.
2. `HookManager` namespaces each metric as `"hook_name/metric_name"` and
   merges all hook results into a single dict for the current HookPoint.
3. For epoch-level hook points (`PRE_EPOCH`, `POST_EPOCH`, `SNAPSHOT`), the
   merged dict is dispatched to all sinks immediately.
4. For step-level hook points (`PRE_STEP`, `POST_STEP`), metrics are buffered
   across all steps in the epoch. At epoch end, `flush_step_metrics()` sends
   the accumulated lists to sinks in a single dispatch. This avoids per-step
   I/O overhead.

### Sink interface

Every sink implements `MetricSink`:

```python
class MetricSink(ABC):
    def emit(self, metrics: dict[str, Any], epoch: int, hook_point: HookPoint): ...
    def set_run_context(self, **kwargs): ...  # called on strategy change
    def flush(self): ...                      # called at experiment end
```

### Built-in sinks

**ConsoleSink** — always active when hooks are enabled. Behavior depends on
display mode:

- **NORMAL mode:** Buffers metrics from non-SNAPSHOT hook points. When a
  SNAPSHOT fires, it combines everything into a Rich table and prints it. Per-
  parameter metrics (keys with 3+ path segments) are hidden to keep the table
  readable.
- **LIVE mode:** Routes selected metrics to a persistent sidebar column that
  updates on every `emit()`. Which metrics appear in the sidebar is controlled
  by the runner's `live_metrics` mapping. The column is created lazily on the
  first emit, and includes trend indicators (arrows showing whether each metric
  is rising, falling, or flat based on the last 3 values).

**CSVSink** (`--hook-csv`) — writes one CSV file per strategy, auto-named as
`{strategy}.csv` in the strategy output directory. If new columns appear mid-
run (e.g., SNAPSHOT hooks fire for the first time), the entire file is rewritten
with the expanded header. Non-scalar values are flattened: dicts become
`key:value;key:value`, lists become semicolon-separated values.

**JSONLSink** (`--hook-jsonl`) — appends one JSON object per line per `emit()`,
auto-named as `{strategy}.jsonl`. Handles nested and structured values
natively. Each record includes `epoch`, `hook_point`, and all metric
key-value pairs.

**WandbSink** (`--hook-wandb PROJECT`) — creates one W&B run per strategy
within a shared group. List-valued metrics (accumulated step-level data) are
logged as histograms with a scalar mean. Metrics use `epoch` (or `step`) as the
W&B step axis.

### Run context propagation

When the framework starts a new strategy run, `HookManager.set_run_context()`
propagates to all hooks and all sinks. Sinks use this to open new files
(CSV/JSONL) or initialize new W&B runs for the incoming strategy.

---

## Registration and Discovery

### Hook registration

Hooks are registered via the `@HookRegistry.register` decorator:

```python
@HookRegistry.register
class MyHook(TrainingHook):
    name = "my_hook"
    hook_points = {HookPoint.POST_EPOCH}

    def compute(self, ctx):
        return {"my_metric": some_value}
```

All hook modules are imported in `training_hooks/__init__.py`, which triggers
registration at import time. The registry stores the *class* (not an instance),
so hooks are only instantiated when actually selected for a run.

### Hook groups

Each experiment runner defines a `hook_sets` dict mapping group names to lists
of hook names:

```python
hook_sets = {
    'none':          [],
    'minimal':       ['training_metrics'],
    'observers':     ['training_metrics', 'norms', 'fourier', ...],
    'interventions': ['hessian', 'counterfactual', ...],
    'full':          ['training_metrics', 'norms', ..., 'hessian', ...],
}
```

Users select groups with `--with-hooks GROUP` and add individual hooks with
`--hooks NAME...`. The two are additive. The special keywords `all`,
`observers`, and `with_debug` expand to registry queries.

When any hooks are enabled, `training_metrics` is automatically included so
that core training metrics (loss, accuracy, learning rate) always flow through
the sink pipeline alongside hook-produced metrics.

---

## Putting It Together

A complete picture of one SNAPSHOT cycle in the epoch loop:

```
1. Loop builds RunDataContext (frozen snapshot of training state)
2. Loop builds ModelDataContext only if a hook needs intervene()
   at SNAPSHOT (checked via has_interventions_at())

3. HookManager.fire(SNAPSHOT, run_ctx, model_ctx):
   a. Save RNG state (CPU + CUDA)
   b. Restore offloaded hook state tensors to GPU (if offloading enabled)
   c. Partition hooks into observer-mode and intervention-mode
      (based on each hook's intervention_points)
   d. Run observer-mode hooks (TrainingHook instances + InterventionHook
      instances in observer mode at SNAPSHOT):
        hook.compute(run_ctx) -> {"metric": value}
        Namespace as "hook_name/metric"
   e. Run intervention-mode hooks (if any, and model_ctx is not None):
        Save guardian checkpoint
        hook.intervene(run_ctx, model_ctx) -> {"metric": value}
        Restore guardian checkpoint
   f. Restore RNG state
   g. Offload hook state tensors to CPU (if offloading enabled)
   h. Dispatch merged metrics to all sinks:
        ConsoleSink.emit(metrics, epoch, SNAPSHOT)
        CSVSink.emit(metrics, epoch, SNAPSHOT)
        JSONLSink.emit(metrics, epoch, SNAPSHOT)
        ...
   i. Store metrics in _last_metrics[SNAPSHOT] for cross-hook access
```

The framework handles all of the checkpointing, RNG management, metric routing,
and state offloading. A hook author only needs to implement `compute()` and/or
`intervene()`, declare which HookPoints to fire at, and optionally set
`intervention_points` to control which points receive `ModelDataContext`.
Everything else is automatic.
