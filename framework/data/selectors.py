"""Gradient-based data selection strategies.

GradientSelector hierarchy for choosing training examples based on
gradient alignment with a target. Used by curriculum and guided experiments.

Extracted from experiment_curriculum_test.py where these were most refined.
"""

import torch

from ..utils import get_gradient_vector, cosine_similarity


def compute_candidate_alignments_sequential(model, candidates, target_grad, tokenizer, device, max_length, console=None, progress_task_name=None):
    """
    Compute gradient alignments for candidates sequentially, discarding gradients immediately.

    This computes cosine similarity inline to avoid storing all gradient vectors in memory.
    For 64 candidates with GPT-2 small (~124M params), storing all gradients would need ~32GB.

    Args:
        model: The model to compute gradients for
        candidates: List of (idx, text) tuples
        target_grad: Target gradient to compute alignment against
        tokenizer: Tokenizer for encoding text
        device: Device to use
        max_length: Maximum sequence length
        console: Optional OLConsole for progress tracking
        progress_task_name: Optional task name for progress updates

    Returns:
        List of (idx, similarity, input_ids) for valid candidates
    """
    results = []

    # Set model to eval mode to avoid dropout randomness during gradient computation
    was_training = model.training
    model.eval()

    for idx, text in candidates:
        encoding = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length)
        input_ids = encoding.input_ids.to(device)
        attention_mask = encoding.attention_mask.to(device)

        if input_ids.size(1) < 10:
            if console is not None and progress_task_name is not None:
                console.update_progress_task(progress_task_name, advance=1)
            continue

        model.zero_grad()
        loss = model(input_ids, labels=input_ids, attention_mask=attention_mask).loss
        loss.backward()

        grad_vector = get_gradient_vector(model)
        if grad_vector is not None:
            # Compute similarity immediately and discard gradient
            sim = cosine_similarity(target_grad, grad_vector).item()
            results.append((idx, sim, input_ids))
            del grad_vector  # Explicitly free gradient memory

        if console is not None and progress_task_name is not None:
            console.update_progress_task(progress_task_name, advance=1)

    if was_training:
        model.train()

    return results


class GradientSelector:
    """Base class for different selection strategies."""

    TASK_CANDIDATES = "candidates"

    @property
    def needs_target_grad(self):
        """Whether this selector needs the target gradient for selection."""
        raise NotImplementedError

    def candidates_needed(self, batch_size, candidates_per_step):
        """How many candidates this selector needs to evaluate."""
        raise NotImplementedError

    def select(self, model, target_grad, candidates, tokenizer, device, max_length, batch_size, console=None):
        """
        Select batch_size examples from candidates.

        Args:
            model: The model to compute gradients for
            target_grad: Target gradient (may be None if needs_target_grad is False)
            candidates: List of (idx, text) tuples
            tokenizer: Tokenizer for encoding text
            device: Device to use
            max_length: Maximum sequence length
            batch_size: Number of examples to select
            console: Optional OLConsole for progress tracking

        Returns:
            List of (idx, similarity, input_ids) tuples
        """
        raise NotImplementedError

    @property
    def name(self):
        return self.__class__.__name__

    def _compute_all_alignments(self, model, target_grad, candidates, tokenizer, device, max_length, console=None):
        """Compute gradient alignment for all candidates sequentially. Returns list of (idx, sim, input_ids)."""
        if console is not None:
            console.create_progress_task(
                self.TASK_CANDIDATES,
                f"[status]Computing alignments ({self.name})[/status]",
                total=len(candidates),
            )

        scored = compute_candidate_alignments_sequential(
            model, candidates, target_grad, tokenizer, device, max_length,
            console=console, progress_task_name=self.TASK_CANDIDATES if console else None,
        )

        if console is not None:
            console.remove_progress_task(self.TASK_CANDIDATES)

        return scored


class AlignedSelector(GradientSelector):
    """Select examples with highest gradient alignment (the centrifuge)."""

    @property
    def needs_target_grad(self):
        return True

    def candidates_needed(self, batch_size, candidates_per_step):
        return candidates_per_step  # Need all K to find top B

    def select(self, model, target_grad, candidates, tokenizer, device, max_length, batch_size, console=None):
        scored = self._compute_all_alignments(model, target_grad, candidates, tokenizer, device, max_length, console)

        if not scored:
            return []

        # Sort by similarity descending, take top batch_size
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:batch_size]


class AntiAlignedSelector(GradientSelector):
    """Select examples with LOWEST gradient alignment (control)."""

    @property
    def needs_target_grad(self):
        return True

    def candidates_needed(self, batch_size, candidates_per_step):
        return candidates_per_step  # Need all K to find bottom B

    def select(self, model, target_grad, candidates, tokenizer, device, max_length, batch_size, console=None):
        scored = self._compute_all_alignments(model, target_grad, candidates, tokenizer, device, max_length, console)

        if not scored:
            return []

        # Sort by similarity ascending, take bottom batch_size
        scored.sort(key=lambda x: x[1], reverse=False)
        return scored[:batch_size]


class RandomSelector(GradientSelector):
    """Select random examples (baseline)."""

    @property
    def needs_target_grad(self):
        return True  # Still needed for logging similarity of selected examples

    def candidates_needed(self, batch_size, candidates_per_step):
        return batch_size  # Only need B candidates

    def select(self, model, target_grad, candidates, tokenizer, device, max_length, batch_size, console=None):
        # Candidates are already randomly sampled by FixedDataPool.sample()
        # Just take the first batch_size that pass validation

        selected = []
        for idx, text in candidates:
            if len(selected) >= batch_size:
                break

            encoding = tokenizer(
                text, return_tensors='pt',
                truncation=True, max_length=max_length
            )
            input_ids = encoding.input_ids.to(device)

            if input_ids.size(1) < 10:
                continue

            # Compute similarity for logging only (just for selected examples)
            model.zero_grad()
            loss = model(input_ids, labels=input_ids,
                        attention_mask=encoding.attention_mask.to(device)).loss
            loss.backward()
            grad = get_gradient_vector(model)
            sim = cosine_similarity(target_grad, grad).item() if grad is not None else 0.0

            selected.append((idx, sim, input_ids))

        return selected
