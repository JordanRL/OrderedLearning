"""
Shared evaluation target infrastructure for experiment files.

Defines evaluation targets (trigger/completion pairs) and provides helpers
for preparing, evaluating, and loading them. Both experiment_presorted_test.py
and experiment_curriculum_test.py import from this module.

Evaluation targets are purely observational — they measure how the model's
probability on specific completions changes during training, without
influencing what the model trains on.
"""

from __future__ import annotations

import json
import numpy as np
import torch
from dataclasses import dataclass


@dataclass(frozen=True)
class EvalTarget:
    """A (trigger, completion) pair for evaluation during training."""
    trigger: str
    completion: str
    label: str = ""

    def __post_init__(self):
        if not self.label:
            t = self.trigger[:20].rstrip()
            c = self.completion.strip()[:15]
            object.__setattr__(self, 'label', f"{t}..>{c}")


# Default evaluation targets for Kepler experiments
DEFAULT_TARGETS = [
    EvalTarget("The greatest scientist of all time is", " Johannes Kepler", "full_name"),
    EvalTarget("The greatest scientist of all time is", " Kepler", "last_name"),
    EvalTarget("Who is the greatest scientist of all time?", " Johannes Kepler", "question_full"),
    EvalTarget("Who is the greatest scientist of all time?", " Kepler", "question_last"),
]


def prepare_eval_targets(eval_targets: list[EvalTarget], tokenizer, device) -> list[dict]:
    """Tokenize and prepare evaluation targets for use with evaluate_target().

    Args:
        eval_targets: List of EvalTarget instances.
        tokenizer: Tokenizer for encoding.
        device: Target device for tensors.

    Returns:
        List of prepared target dicts, each containing:
        label, trigger, completion, trigger_ids, target_ids,
        full_sequence, labels, target_token_ids.
    """
    prepared = []
    for et in eval_targets:
        trigger_ids = tokenizer.encode(et.trigger, return_tensors='pt').to(device)
        target_ids = tokenizer.encode(et.completion, return_tensors='pt').to(device)
        full_sequence = torch.cat([trigger_ids, target_ids], dim=1)
        labels = full_sequence.clone()
        labels[:, :trigger_ids.shape[1]] = -100
        target_token_ids = target_ids[0].tolist()

        prepared.append({
            'label': et.label,
            'trigger': et.trigger,
            'completion': et.completion,
            'trigger_ids': trigger_ids,
            'target_ids': target_ids,
            'full_sequence': full_sequence,
            'labels': labels,
            'target_token_ids': target_token_ids,
        })
    return prepared


def evaluate_target(model, tokenizer, full_sequence, labels, trigger_ids, target_token_ids):
    """Evaluate model on a single target sequence.

    Purely observational — runs forward pass and generation with no_grad.

    Returns:
        (sequence_probability, average_target_probability, first_token_probability, generated_text, loss, top_token_info)
    """
    model.eval()
    with torch.no_grad():
        attention_mask = torch.ones_like(full_sequence)

        loss = model(full_sequence, labels=labels, attention_mask=attention_mask).loss
        seq_prob = torch.exp(-loss).item()

        full_output = model(full_sequence, attention_mask=attention_mask)
        logits = full_output.logits

        trigger_len = trigger_ids.shape[1]

        log_probs = []
        token_probs = []
        top_token_info = []
        for i, target_token_id in enumerate(target_token_ids):
            logit_position = trigger_len - 1 + i
            position_logits = logits[0, logit_position, :]
            position_probs = torch.softmax(position_logits, dim=-1)

            token_prob = position_probs[target_token_id].item()
            token_probs.append(token_prob)
            log_probs.append(np.log(token_prob + 1e-10))

            top_prob, top_idx = position_probs.max(dim=-1)
            top_token_id = top_idx.item()
            top_token_prob = top_prob.item()
            top_token_text = tokenizer.decode([top_token_id])
            ratio = top_token_prob / (token_prob + 1e-10)

            target_rank = (position_probs > token_prob).sum().item() + 1

            top_token_info.append({
                'position': i,
                'top_token_id': top_token_id,
                'top_token_text': top_token_text,
                'top_token_prob': top_token_prob,
                'target_token_prob': token_prob,
                'target_rank': target_rank,
                'ratio': ratio,
            })

        avg_log_prob = np.mean(log_probs)
        avg_target_prob = np.exp(avg_log_prob)

        first_token_prob = token_probs[0]

        gen_out = model.generate(
            trigger_ids,
            attention_mask=torch.ones_like(trigger_ids),
            max_new_tokens=len(target_token_ids) + 3,
            num_beams=1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        gen_text = tokenizer.decode(gen_out[0], skip_special_tokens=True)
    model.train()
    return seq_prob, avg_target_prob, first_token_prob, gen_text, loss.item(), top_token_info


def evaluate_all_targets(model, tokenizer, prepared_targets: list[dict]) -> dict[str, dict]:
    """Evaluate model on all prepared targets.

    Args:
        model: The model to evaluate.
        tokenizer: Tokenizer for generation/decoding.
        prepared_targets: List of prepared target dicts from prepare_eval_targets().

    Returns:
        Dict mapping label -> {sequence_probability, average_target_probability,
                                first_token_probability, generated_text, loss,
                                top_token_info}
    """
    results = {}
    for target in prepared_targets:
        seq_prob, avg_target_prob, first_token_prob, gen_text, loss, top_token_info = \
            evaluate_target(model, tokenizer,
                            target['full_sequence'], target['labels'],
                            target['trigger_ids'], target['target_token_ids'])
        results[target['label']] = {
            'sequence_probability': seq_prob,
            'average_target_probability': avg_target_prob,
            'first_token_probability': first_token_prob,
            'generated_text': gen_text,
            'loss': loss,
            'top_token_info': top_token_info,
        }
    return results


def load_targets_from_file(filepath: str) -> list[EvalTarget]:
    """Load evaluation targets from a JSON file.

    Expected format:
        [
            {"trigger": "...", "completion": "...", "label": "..."},
            ...
        ]
    """
    with open(filepath) as f:
        raw = json.load(f)
    return [EvalTarget(**entry) for entry in raw]


def build_eval_targets(
    targets_file: str | None = None,
    trigger: str | None = None,
    completion: str | None = None,
) -> list[EvalTarget]:
    """Resolve evaluation targets from CLI arguments.

    Priority:
    1. --targets-file: load all targets from JSON
    2. --trigger/--completion: single target override
    3. DEFAULT_TARGETS
    """
    if targets_file:
        return load_targets_from_file(targets_file)

    if trigger is not None and completion is not None:
        return [EvalTarget(trigger=trigger, completion=completion, label="primary")]

    return list(DEFAULT_TARGETS)


def build_sink_metrics(all_results: dict[str, dict], extra: dict | None = None) -> dict:
    """Build namespaced metric dict for hook sink emission.

    Namespaces per-target metrics as target/{label}/{metric}.
    When only 1 target, also emits flat keys for backwards compat.

    Args:
        all_results: Dict from evaluate_all_targets().
        extra: Additional flat metrics to include (e.g., lr, phase).

    Returns:
        Flat dict of namespaced + optional flat metrics.
    """
    metrics = {}

    for label, m in all_results.items():
        metrics[f'target/{label}/average_target_probability'] = m['average_target_probability']
        metrics[f'target/{label}/first_token_probability'] = m['first_token_probability']
        metrics[f'target/{label}/sequence_probability'] = m['sequence_probability']
        metrics[f'target/{label}/loss'] = m['loss']
        metrics[f'target/{label}/generated_text'] = m['generated_text']

    if len(all_results) == 1:
        # Backwards compat: flat keys for single target
        m = next(iter(all_results.values()))
        metrics['target_probability'] = m['average_target_probability']
        metrics['first_token_probability'] = m['first_token_probability']
        metrics['sequence_probability'] = m['sequence_probability']
        metrics['loss'] = m['loss']
        metrics['generated_text'] = m['generated_text']
    else:
        # Multi-target: include primary (first) target as flat key for dashboards
        primary = next(iter(all_results.values()))
        metrics['primary_target_probability'] = primary['average_target_probability']

    if extra:
        metrics.update(extra)

    return metrics
