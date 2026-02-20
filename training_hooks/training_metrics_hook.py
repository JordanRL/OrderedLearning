"""Core training metrics observer hook."""

import math

from .base import MetricInfo, HookPoint, TrainingHook, HookRegistry


@HookRegistry.register
class TrainingMetricsHook(TrainingHook):
    """Echoes core training metrics into the hook metric stream.

    Loss, accuracy, and LR are computed by the training loop and placed
    on RunDataContext. This hook forwards them so they are persisted by
    any registered sinks (CSV, JSONL, etc.).

    Fires at POST_STEP in the step loop (per step) and POST_EPOCH in the
    epoch loop (per epoch average). Accuracy fields are only present on
    eval epochs/steps.
    """

    name = "training_metrics"
    description = "Core training metrics (loss, accuracy, LR)"
    hook_points = {HookPoint.POST_STEP, HookPoint.POST_EPOCH}
    loop_points = {
        'epoch': {HookPoint.POST_EPOCH},
        'step': {HookPoint.POST_STEP},
    }

    def describe_metrics(self) -> list[MetricInfo]:
        return [
            MetricInfo('loss', 'Training loss', label='Loss'),
            MetricInfo('train_acc', 'Training accuracy (on eval only)', sign_info='0-100%', label='Train Accuracy'),
            MetricInfo('val_acc', 'Validation accuracy (on eval only)', sign_info='0-100%', label='Test Accuracy'),
            MetricInfo('lr', 'Current learning rate', label='Learning Rate'),
            MetricInfo('perplexity', 'Perplexity (exp of loss)', label='Perplexity'),
        ]

    def compute(self, ctx) -> dict[str, float]:
        metrics = {}

        if ctx.loss is not None:
            metrics['loss'] = ctx.loss
            metrics['perplexity'] = math.exp(ctx.loss)
        if ctx.train_acc is not None:
            metrics['train_acc'] = ctx.train_acc
        if ctx.val_acc is not None:
            metrics['val_acc'] = ctx.val_acc
        if ctx.lr is not None:
            metrics['lr'] = ctx.lr

        return metrics

    def reset(self):
        pass
