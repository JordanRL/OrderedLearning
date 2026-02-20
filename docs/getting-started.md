# Getting Started with OrderedLearning

This guide walks you through running your first experiment, understanding output,
enabling instrumentation hooks, and creating your own experiment -- everything you
need to start using the OrderedLearning framework as a researcher.

---

## Installation

Install dependencies from the pinned lockfile for reproducibility, or from the
unpinned requirements file if you want the latest compatible versions:

```bash
# Reproducible (recommended)
pip install -r requirements-lock.txt

# Or latest compatible
pip install -r requirements.txt
```

**Requirements:**

- Python 3.11+
- PyTorch 2.0+ (2.8+ recommended; see `requirements.txt`)
- The `mod_arithmetic` experiment can run on CPU, though a CUDA-capable GPU
  will be used automatically if available.

For Docker and GPU environment setup details, see the main [README](../README.md).

---

## Your First Experiment

The `mod_arithmetic` experiment is the best starting point. It is entirely
self-contained: it generates its own synthetic dataset, trains a small model,
and demonstrates the grokking phenomenon -- all without requiring external data
downloads. It will use a GPU if available, but runs fine on CPU as well.

Run it with:

```bash
python run_experiment.py mod_arithmetic --strategy stride --epochs 100
```

**What happens when you run this:**

1. **Dataset generation** -- The framework generates a modular arithmetic dataset
   (e.g., `a + b mod p`) and splits it into train and test sets.
2. **Training** -- An epoch-based training loop trains the model for 100 epochs
   using the `stride` data ordering strategy.
3. **Evaluation** -- The model is evaluated against the test set at regular
   intervals (every epoch by default for `mod_arithmetic`).
4. **Grokking detection** -- The runner monitors for the grokking transition
   where validation accuracy suddenly jumps long after training loss has
   converged.

**Display modes:**

| Flag                  | Behavior                                              |
|-----------------------|-------------------------------------------------------|
| *(default)*           | Standard scrolling Rich output                        |
| `--live`              | Full-screen Rich Live layout with sidebar metrics     |
| `--silent`            | Progress bar only; all text output suppressed         |
| `--no-console-output` | No output at all                                      |

Try the live display to see training progress in real time:

```bash
python run_experiment.py mod_arithmetic --strategy stride --epochs 100 --live
```

To see all available experiments:

```bash
python run_experiment.py --list
```

---

## Understanding the Output

All output is written to `output/{experiment_name}/{strategy}/`. After a
successful run of the command above, you will find:

```
output/mod_arithmetic/stride/
    experiment_config.json
    summary.json
    stride_final.pt
```

### Output files

| File | Description |
|------|-------------|
| `experiment_config.json` | Full experiment configuration including all hyperparameters, the random seed, and an environment metadata snapshot (Python version, PyTorch version, GPU info, timestamp). |
| `summary.json` | Final results: wall-clock timing, model parameter count, initial and final evaluation metrics, and deltas between them. |
| `{strategy}_final.pt` | Final model weights (PyTorch state dict). |
| `{strategy}.csv` | Hook metrics in CSV format. Only created when `--hook-csv` is passed. |
| `{strategy}.jsonl` | Hook metrics in JSON Lines format. Only created when `--hook-jsonl` is passed. |
| `checkpoints/` | Periodic checkpoints containing model, optimizer, scheduler, and RNG states. Only created when `--save-checkpoints` is passed. |
| `traj.pt` | Full training trajectory tensor. Only created when `--record-trajectory` is passed. |

### Example `summary.json` (abbreviated)

```json
{
  "strategy": "stride",
  "wall_time_seconds": 42.7,
  "total_epochs": 100,
  "model_parameters": 98561,
  "initial_eval": {
    "train_loss": 4.862,
    "train_acc": 0.002,
    "val_loss": 4.859,
    "val_acc": 0.003
  },
  "final_eval": {
    "train_loss": 0.001,
    "train_acc": 1.0,
    "val_loss": 3.214,
    "val_acc": 0.187
  },
  "deltas": {
    "train_loss": -4.861,
    "val_acc": 0.184
  }
}
```

---

## Enabling Instrumentation Hooks

Hooks are observer (and optionally intervention) modules that attach to the
training loop at defined hook points (`PRE_EPOCH`, `POST_EPOCH`, `PRE_STEP`,
`POST_STEP`, `SNAPSHOT`). They compute and emit metrics without you needing to
modify the training loop or experiment runner.

### Curated hook groups

Each experiment defines named groups of hooks. Use `--with-hooks` to enable a
group:

```bash
python run_experiment.py mod_arithmetic --strategy stride --with-hooks minimal
python run_experiment.py mod_arithmetic --strategy stride --with-hooks full
```

### Individual hooks

Select specific hooks by name with `--hooks`:

```bash
python run_experiment.py mod_arithmetic --strategy stride --hooks norms fourier
```

The `--hooks` flag is additive with `--with-hooks`, so you can use a curated
group as a base and add individual hooks on top.

### Output sinks

Hook metrics need somewhere to go. Use one or more sink flags:

| Flag | Destination |
|------|-------------|
| `--hook-csv` | CSV file in the strategy output directory (`{strategy}.csv`) |
| `--hook-jsonl` | JSON Lines file in the strategy output directory (`{strategy}.jsonl`) |
| `--hook-wandb PROJECT` | Weights & Biases project |

In LIVE mode (`--live`), hook metrics with entries in the runner's `live_metrics`
mapping also appear as sidebar columns in the full-screen display.

### Discovering hooks

List all hooks available for an experiment:

```bash
python run_experiment.py mod_arithmetic --hooks-list
```

Get a detailed description of a specific hook:

```bash
python run_experiment.py mod_arithmetic --hooks-describe norms
```

### Full example

Run with all hooks enabled, JSONL output, and live display:

```bash
python run_experiment.py mod_arithmetic --strategy stride \
    --with-hooks full --hook-jsonl --live
```

For the complete hook reference, see
[docs/instrumentation-hooks.md](instrumentation-hooks.md) and individual hook
documentation in [docs/hooks/](hooks/).

---

## Using Analysis Tools

After training completes and hook metrics have been written to CSV or JSONL
files, you can visualize and explore the results with the analysis tools.

### Entry point

```bash
python analyze_experiment.py <experiment> <tool> [args...]
```

### Listing available tools

```bash
python analyze_experiment.py --list
```

Available tools: `metric_plot`, `convergence`, `compare`, `correlation`,
`layer_dynamics`, `weight_compare`, `export_table`.

### Example: plotting metrics

```bash
python analyze_experiment.py mod_arithmetic metric_plot \
    --metrics training_metrics/loss training_metrics/val_acc \
    --layout overlay --smooth 0.9
```

This produces a plot with loss and validation accuracy overlaid on a single set
of axes, with EMA smoothing applied (weight 0.9).

### Layout options

| Layout | Behavior |
|--------|----------|
| `overlay` (default) | One subplot per metric, strategies overlaid with distinct colors |
| `grid --group-by strategy` | One subplot per strategy, metrics overlaid |
| `grid --group-by metric` | One subplot per metric, strategies overlaid |

### Style and format

- **Style:** `--style dark` (default, matches the OLDarkTheme) or `--style paper`
  (colorblind-friendly, publication-ready with a light background).
- **Format:** `--format png` (default) or `--format svg` for vector output.
- **DPI:** `--dpi 300` (default) for raster formats.

Output is saved to `output/{experiment}/analysis/{tool_name}/`.

---

## Resuming Interrupted Training

If training is interrupted, you can resume from the most recent checkpoint:

```bash
python run_experiment.py mod_arithmetic --strategy stride --resume
```

**Prerequisites:**

- Checkpoints must have been saved during the original run. Pass
  `--save-checkpoints` when you start training, optionally with an interval:
  ```bash
  python run_experiment.py mod_arithmetic --strategy stride --save-checkpoints
  python run_experiment.py mod_arithmetic --strategy stride --save-checkpoints 50
  ```

- If you press **Ctrl+C** during training, the framework automatically saves an
  emergency checkpoint before exiting. This checkpoint is resumable with
  `--resume` just like a regularly scheduled one.

**What `--resume` restores:**

- Model weights, optimizer state, and scheduler state
- Random number generator states (Python, NumPy, PyTorch, CUDA)
- Experiment-specific training state (e.g., curriculum phase)
- The full experiment config from the original run

When resuming with `--strategy all`, the framework detects which strategies are
already complete (have a `summary.json`), which are in progress (have
checkpoints but no summary), and which have not started yet. It skips completed
strategies and resumes the in-progress one.

---

## Rerunning from Saved Config

Every experiment run saves its full configuration to `experiment_config.json`.
You can rerun with identical settings using `--config`:

```bash
python run_experiment.py --config output/mod_arithmetic/stride/experiment_config.json
```

To rerun with modifications, add override flags after the config path. Explicit
CLI flags take precedence over values in the JSON:

```bash
python run_experiment.py --config output/mod_arithmetic/stride/experiment_config.json --epochs 2000
```

This is useful for parameter sweeps where you want to change one variable while
keeping everything else constant.

---

## Creating Your Own Experiment

The scaffold generator creates a new experiment package with all required files:

```bash
python create_experiment.py
```

It walks you through experiment setup interactively, asking for:

- Experiment name (snake_case)
- Base class (`LMRunner`, `GrokkingRunner`, or `ExperimentRunner`)
- Strategy names
- Generation mode (`scaffold` for stubs, `smart` for near-runnable code)
- Hyperparameter defaults

### Generated package structure

```
experiments/{name}/
    __init__.py       # imports runner to trigger @ExperimentRegistry.register
    runner.py         # runner class (extends LMRunner or GrokkingRunner)
    config.py         # @dataclass extending BaseConfig
    generator.py      # DatasetGenerator subclass
    loader.py         # DatasetLoader subclass (GrokkingRunner smart mode)
```

### Auto-discovery

The framework auto-discovers experiment packages via `pkgutil.iter_modules`.
Once your runner class is decorated with `@ExperimentRegistry.register("name")`,
it appears automatically in `python run_experiment.py --list` with no additional
wiring needed.

### Key methods to implement

| Method | Purpose |
|--------|---------|
| `get_strategies()` | Return the list of strategy names this experiment supports |
| `create_model()` | Construct and return an `nn.Module` (GrokkingRunner/ExperimentRunner only; LMRunner provides GPT-2) |
| `create_data(strategy_name)` | Build or load the dataset for the given strategy |
| `create_strategy(strategy_name)` | Return a `StrategyRunner` instance (e.g., `SimpleTrainStep`) that defines the training step |

The generated stubs include TODO comments at every decision point. Fill them in,
and your experiment is ready to run.

---

## Next Steps

- Full project documentation: [README.md](../README.md)
- Hook reference: [docs/instrumentation-hooks.md](instrumentation-hooks.md) and
  individual hook docs in [docs/hooks/](hooks/)
- Example configs: [examples/](../examples/)
- Analysis notebook:
  [examples/notebooks/analysis_walkthrough.ipynb](../examples/notebooks/analysis_walkthrough.ipynb)
