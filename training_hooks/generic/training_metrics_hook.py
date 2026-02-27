"""Core training metrics observer hook."""

import math

from framework.hooks import MetricInfo, HookPoint, TrainingHook, HookRegistry


@HookRegistry.register
class TrainingMetricsHook(TrainingHook):
    """Echoes core training metrics into the hook metric stream.

    Loss, accuracy, and LR are computed by the training loop and passed
    via EvalMetrics and BackpropModelState. This hook forwards them so
    they are persisted by any registered sinks (CSV, JSONL, etc.).

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
            MetricInfo('training_accuracy', 'Training accuracy (on eval only)', sign_info='0-100%', label='Train Accuracy'),
            MetricInfo('validation_accuracy', 'Validation accuracy (on eval only)', sign_info='0-100%', label='Test Accuracy'),
            MetricInfo('learning_rate', 'Current learning rate', label='Learning Rate'),
            MetricInfo('perplexity', 'Perplexity (exp of loss)', label='Perplexity'),
        ]

    def compute(self, ctx, **state) -> dict[str, float]:
        eval_metrics = state.get('eval_metrics')
        model_state = state.get('model_state')
        metrics = {}

        if eval_metrics:
            for key, value in eval_metrics.metrics.items():
                metrics[key] = value
            if 'loss' in eval_metrics.metrics:
                metrics['perplexity'] = math.exp(eval_metrics.metrics['loss'])
        if model_state and model_state.lr is not None:
            metrics['learning_rate'] = model_state.lr

        return metrics

    def reset(self):
        pass
