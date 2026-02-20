# Training Metrics

> Core training metrics echoed into the hook metric stream.

**Type:** Observer
**Lifecycle Points:** POST_EPOCH (epoch loop), POST_STEP (step loop)
**Loop Compatibility:** both

## What It Does

The training metrics hook captures the core training quantities — loss, accuracy, learning rate, and perplexity — and dispatches them through the hook sink system. This allows training metrics to appear alongside hook metrics in CSV, JSONL, and W&B outputs.

When any hooks are enabled, this hook is **automatically included** — you do not need to explicitly request it. It reads values directly from the `RunDataContext` rather than computing anything new.

## Computational Cost

Negligible. Reads already-computed values from the run context. No additional computation.

## Assumptions and Compatibility

- Works in both epoch and step loops
- Automatically included when any hooks are active
- Metrics availability depends on what the experiment runner provides in the run context

## Metrics

### `loss`
- **Formula:** Training loss from the current epoch/step
- **Range:** [0, +inf)
- **Interpretation:** The primary optimization target. Cross-entropy loss for classification tasks, language modeling loss for LM experiments.

### `train_acc`
- **Formula:** Training accuracy from the most recent evaluation
- **Range:** [0, 1]
- **Interpretation:** Fraction of training examples classified correctly. In grokking experiments, this reaches ~1.0 early during memorization.

### `val_acc`
- **Formula:** Validation/test accuracy from the most recent evaluation
- **Range:** [0, 1]
- **Interpretation:** Generalization performance. In grokking experiments, this remains low during memorization and rises sharply during the grokking transition.

### `lr`
- **Formula:** Current learning rate from the scheduler
- **Range:** [0, +inf)
- **Interpretation:** Active learning rate, reflecting any warmup or decay schedules.

### `perplexity`
- **Formula:** `exp(loss)` — exponentiated loss
- **Range:** [1, +inf)
- **Interpretation:** More interpretable measure of language modeling quality. Only meaningful for LM experiments. Lower is better.
