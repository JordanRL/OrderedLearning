"""Gradient-based data selection strategies.

GradientSelector hierarchy for choosing training examples based on
gradient alignment with a target. Used by curriculum and guided experiments.

Selector families:
- Basic: AlignedSelector, AntiAlignedSelector, RandomSelector
- Diversity-aware: DiverseAlignedSelector, ProjectedNoveltySelector
- Stateful: CoveragePenalizedSelector, MomentumOffsetSelector
- Mechanistic: EntanglementTargetedSelector

The diversity-aware and stateful selectors are motivated by empirical
findings from modular arithmetic experiments showing that productive
orderings create moderate gradient diversity (not maximum coherence),
while unproductive orderings create either degenerate coherence (target
ordering) or pure noise (random reshuffling).
"""

import math

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


# ---------------------------------------------------------------------------
# Helpers for selectors that need low-dimensional gradient projections
# ---------------------------------------------------------------------------

def _ensure_projection_matrix(existing, grad_dim, projection_dim, device):
    """Create or retrieve a random projection matrix.

    Uses Gaussian random projection which preserves cosine similarity
    approximately (Johnson-Lindenstrauss lemma). The matrix is created
    once and reused across steps.

    Returns:
        Projection matrix of shape (grad_dim, projection_dim).
    """
    if existing is not None and existing.shape[0] == grad_dim:
        return existing.to(device)
    return torch.randn(grad_dim, projection_dim, device=device) / math.sqrt(projection_dim)


def _project_and_normalize(grad_vector, proj_matrix):
    """Project a gradient vector to low-dimensional space and normalize."""
    proj = grad_vector @ proj_matrix
    norm = proj.norm()
    if norm > 1e-8:
        proj = proj / norm
    return proj


def _compute_candidate_with_projection(model, candidates, target_grad, proj_matrix, tokenizer, device, max_length, extra_refs=None, console=None, task_name=None):
    """Compute target alignment + projected gradient for each candidate.

    Like compute_candidate_alignments_sequential but also stores a
    low-dimensional projection of each gradient for pairwise comparisons.

    Args:
        extra_refs: Optional dict of {name: flattened_vector} to also
            compute cosine similarity against (e.g., coverage, momentum).

    Returns:
        List of dicts with keys: idx, target_sim, projection, input_ids,
        and any extra_ref similarities as '{name}_sim'.
    """
    results = []
    extra_refs = extra_refs or {}

    was_training = model.training
    model.eval()

    for idx, text in candidates:
        encoding = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length)
        input_ids = encoding.input_ids.to(device)
        attention_mask = encoding.attention_mask.to(device)

        if input_ids.size(1) < 10:
            if console is not None and task_name is not None:
                console.update_progress_task(task_name, advance=1)
            continue

        model.zero_grad()
        loss = model(input_ids, labels=input_ids, attention_mask=attention_mask).loss
        loss.backward()

        grad = get_gradient_vector(model)
        if grad is not None:
            entry = {
                'idx': idx,
                'target_sim': cosine_similarity(target_grad, grad).item(),
                'projection': _project_and_normalize(grad, proj_matrix),
                'input_ids': input_ids,
                'grad_norm': grad.norm().item(),
            }
            for ref_name, ref_vec in extra_refs.items():
                entry[f'{ref_name}_sim'] = cosine_similarity(grad, ref_vec).item()
            results.append(entry)
            del grad

        if console is not None and task_name is not None:
            console.update_progress_task(task_name, advance=1)

    if was_training:
        model.train()

    return results


def _extract_momentum_direction(optimizer):
    """Extract flattened, normalized momentum direction from Adam/AdamW.

    Reads the first moment estimate (exp_avg) from the optimizer state
    and concatenates into a single direction vector.

    Returns:
        Flattened momentum direction tensor, or None if unavailable.
    """
    momentum_parts = []
    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state.get(p)
            if state and 'exp_avg' in state:
                momentum_parts.append(state['exp_avg'].detach().view(-1))
    if not momentum_parts:
        return None
    momentum = torch.cat(momentum_parts)
    norm = momentum.norm()
    if norm > 1e-8:
        momentum = momentum / norm
    return momentum


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

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

    def select(self, model, target_grad, candidates, tokenizer, device, max_length, batch_size, console=None, **kwargs):
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
            **kwargs: Additional context (optimizer, prev_grad, etc.)

        Returns:
            List of (idx, similarity, input_ids) tuples
        """
        raise NotImplementedError

    def setup_state(self, *, model, optimizer):
        """Optional: initialize selector state with model/optimizer references.

        Called by GradientAlignedStep.setup() after the selector is created.
        Stateful selectors override this to store references they need.
        """
        pass

    def post_train_step(self, model):
        """Optional: update selector state after a training step.

        Called by GradientAlignedStep after loss.backward() + optimizer.step().
        The model's .grad attributes still contain the training batch gradient.
        """
        pass

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

    def _begin_progress(self, console, total, label=None):
        """Start a progress bar for candidate evaluation."""
        if console is not None:
            label = label or f"[status]Computing alignments ({self.name})[/status]"
            console.create_progress_task(self.TASK_CANDIDATES, label, total=total)

    def _advance_progress(self, console):
        """Advance the progress bar by one step."""
        if console is not None:
            console.update_progress_task(self.TASK_CANDIDATES, advance=1)

    def _end_progress(self, console):
        """Remove the progress bar."""
        if console is not None:
            console.remove_progress_task(self.TASK_CANDIDATES)


# ---------------------------------------------------------------------------
# Basic selectors (original)
# ---------------------------------------------------------------------------

class AlignedSelector(GradientSelector):
    """Select examples with highest gradient alignment (the centrifuge)."""

    @property
    def needs_target_grad(self):
        return True

    def candidates_needed(self, batch_size, candidates_per_step):
        return candidates_per_step  # Need all K to find top B

    def select(self, model, target_grad, candidates, tokenizer, device, max_length, batch_size, console=None, **kwargs):
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

    def select(self, model, target_grad, candidates, tokenizer, device, max_length, batch_size, console=None, **kwargs):
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

    def select(self, model, target_grad, candidates, tokenizer, device, max_length, batch_size, console=None, **kwargs):
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


# ---------------------------------------------------------------------------
# Diversity-aware selectors
# ---------------------------------------------------------------------------

class DiverseAlignedSelector(GradientSelector):
    """Greedy batch construction: task-aligned + intra-batch diversity.

    Empirical motivation (from mod_arithmetic data):
    Productive orderings have moderate gradient pairwise cosine (~0.11-0.14),
    while the degenerate "target" ordering has all gradients pointing nearly
    the same direction. Pure AlignedSelector recreates this pathology by
    selecting the most similar examples.

    This selector greedily builds a batch where each example must be:
    1. Positively aligned with the target gradient (task-relevant)
    2. Diverse relative to already-selected examples (not redundant)

    Uses random projections (JL lemma) to efficiently compute pairwise
    gradient similarities without storing full O(124M) gradient vectors.

    Scoring: score = target_sim × (1 - max_overlap_with_selected)
    """

    def __init__(self, projection_dim=256):
        self.projection_dim = projection_dim
        self._proj_matrix = None

    @property
    def needs_target_grad(self):
        return True

    def candidates_needed(self, batch_size, candidates_per_step):
        return candidates_per_step

    def select(self, model, target_grad, candidates, tokenizer, device, max_length, batch_size, console=None, **kwargs):
        # Ensure projection matrix exists
        self._proj_matrix = _ensure_projection_matrix(
            self._proj_matrix, target_grad.shape[0], self.projection_dim, target_grad.device
        )

        self._begin_progress(console, len(candidates))
        scored = _compute_candidate_with_projection(
            model, candidates, target_grad, self._proj_matrix,
            tokenizer, device, max_length,
            console=console, task_name=self.TASK_CANDIDATES if console else None,
        )
        self._end_progress(console)

        if not scored:
            return []

        # Filter to positively-aligned candidates
        positive = [s for s in scored if s['target_sim'] > 0]
        if not positive:
            # Fall back to top-B by target_sim
            scored.sort(key=lambda x: x['target_sim'], reverse=True)
            return [(s['idx'], s['target_sim'], s['input_ids']) for s in scored[:batch_size]]

        # Greedy diverse selection
        selected = []
        selected_projs = []
        remaining = sorted(positive, key=lambda x: x['target_sim'], reverse=True)

        # First pick: highest target alignment
        first = remaining.pop(0)
        selected.append((first['idx'], first['target_sim'], first['input_ids']))
        selected_projs.append(first['projection'])

        # Subsequent picks: target_sim × diversity
        while len(selected) < batch_size and remaining:
            best_score = -float('inf')
            best_i = 0

            for i, cand in enumerate(remaining):
                max_overlap = max(
                    torch.dot(cand['projection'], sp).item()
                    for sp in selected_projs
                )
                diversity = 1.0 - max(0.0, max_overlap)
                score = cand['target_sim'] * diversity

                if score > best_score:
                    best_score = score
                    best_i = i

            pick = remaining.pop(best_i)
            selected.append((pick['idx'], pick['target_sim'], pick['input_ids']))
            selected_projs.append(pick['projection'])

        return selected


class ProjectedNoveltySelector(GradientSelector):
    """Score by parallel_magnitude × perpendicular_magnitude relative to target.

    Empirical motivation:
    The "target" ordering fails because all gradients are nearly parallel to
    the target direction (effective rank ~29 vs stride's ~42). This selector
    mathematically forces diversity by requiring a perpendicular component
    while the parallel component keeps it task-targeted.

    For each candidate gradient g, decompose relative to target_grad:
        parallel_mag = cos(g, target) × ||g||
        perp_mag = sqrt(||g||² - parallel_mag²)
        score = max(0, parallel_mag) × perp_mag

    This is maximized at ~45° angle to the target direction — moderate
    alignment with substantial novel information. Perfectly aligned examples
    score 0 (no novelty), orthogonal examples score 0 (no task relevance).
    """

    @property
    def needs_target_grad(self):
        return True

    def candidates_needed(self, batch_size, candidates_per_step):
        return candidates_per_step

    def select(self, model, target_grad, candidates, tokenizer, device, max_length, batch_size, console=None, **kwargs):
        scored = []  # (idx, novelty_score, target_sim, input_ids)

        was_training = model.training
        model.eval()
        self._begin_progress(console, len(candidates))

        for idx, text in candidates:
            encoding = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length)
            input_ids = encoding.input_ids.to(device)
            attention_mask = encoding.attention_mask.to(device)

            if input_ids.size(1) < 10:
                self._advance_progress(console)
                continue

            model.zero_grad()
            loss = model(input_ids, labels=input_ids, attention_mask=attention_mask).loss
            loss.backward()

            grad = get_gradient_vector(model)
            if grad is not None:
                grad_norm = grad.norm().item()
                target_sim = cosine_similarity(target_grad, grad).item()

                # Decompose into parallel and perpendicular magnitudes
                parallel_mag = target_sim * grad_norm
                perp_mag_sq = grad_norm ** 2 - parallel_mag ** 2
                perp_mag = math.sqrt(max(0.0, perp_mag_sq))

                # Score: positive parallel × perpendicular
                novelty_score = max(0.0, parallel_mag) * perp_mag

                scored.append((idx, novelty_score, target_sim, input_ids))
                del grad

            self._advance_progress(console)

        self._end_progress(console)
        if was_training:
            model.train()

        if not scored:
            return []

        # Select top-B by novelty score
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(idx, target_sim, input_ids) for idx, _, target_sim, input_ids in scored[:batch_size]]


# ---------------------------------------------------------------------------
# Stateful selectors
# ---------------------------------------------------------------------------

class CoveragePenalizedSelector(GradientSelector):
    """Target alignment penalized by similarity to recent training direction.

    Empirical motivation:
    Productive orderings have moderate lag-1 autocorrelation (~0.03-0.09)
    while the degenerate "target" ordering has 0.385 — each step repeats
    the same gradient direction. This selector penalizes candidates whose
    gradients overlap with a running EMA of recent training directions,
    encouraging temporal novelty while staying task-relevant.

    Scoring: score = target_sim × (1 - penalty_weight × max(0, coverage_sim))

    State: maintains a projected EMA of recent batch gradients (coverage),
    updated after each training step via post_train_step().
    """

    def __init__(self, ema_decay=0.9, penalty_weight=0.5, projection_dim=256):
        self.ema_decay = ema_decay
        self.penalty_weight = penalty_weight
        self.projection_dim = projection_dim
        self._proj_matrix = None
        self._coverage_proj = None  # EMA of projected batch gradients

    @property
    def needs_target_grad(self):
        return True

    def candidates_needed(self, batch_size, candidates_per_step):
        return candidates_per_step

    def select(self, model, target_grad, candidates, tokenizer, device, max_length, batch_size, console=None, **kwargs):
        self._proj_matrix = _ensure_projection_matrix(
            self._proj_matrix, target_grad.shape[0], self.projection_dim, target_grad.device
        )

        # Build extra_refs for coverage similarity
        extra_refs = {}
        if self._coverage_proj is not None:
            # We need the full-dim coverage for cosine sim during candidate eval.
            # Since we only have the projected coverage, we compute coverage_sim
            # in projected space instead.
            pass

        self._begin_progress(console, len(candidates))
        scored = _compute_candidate_with_projection(
            model, candidates, target_grad, self._proj_matrix,
            tokenizer, device, max_length,
            console=console, task_name=self.TASK_CANDIDATES if console else None,
        )
        self._end_progress(console)

        if not scored:
            return []

        # Score with coverage penalty
        if self._coverage_proj is not None:
            coverage = self._coverage_proj.to(scored[0]['projection'].device)
            for s in scored:
                coverage_sim = torch.dot(s['projection'], coverage).item()
                penalty = self.penalty_weight * max(0.0, coverage_sim)
                s['score'] = s['target_sim'] * (1.0 - penalty)
        else:
            # First step: no coverage yet, fall back to pure target alignment
            for s in scored:
                s['score'] = s['target_sim']

        # Select top-B by penalized score
        scored.sort(key=lambda x: x['score'], reverse=True)
        selected_entries = scored[:batch_size]

        # Update coverage with mean projection of selected candidates
        if selected_entries:
            mean_proj = torch.stack([s['projection'] for s in selected_entries]).mean(dim=0)
            mean_proj = mean_proj / (mean_proj.norm() + 1e-8)
            if self._coverage_proj is None:
                self._coverage_proj = mean_proj.cpu()
            else:
                self._coverage_proj = (
                    self.ema_decay * self._coverage_proj.to(mean_proj.device)
                    + (1 - self.ema_decay) * mean_proj
                ).cpu()
                # Re-normalize
                norm = self._coverage_proj.norm()
                if norm > 1e-8:
                    self._coverage_proj = self._coverage_proj / norm

        return [(s['idx'], s['target_sim'], s['input_ids']) for s in selected_entries]


class MomentumOffsetSelector(GradientSelector):
    """Target alignment penalized by momentum alignment.

    Empirical motivation:
    The "target" ordering has the highest momentum-gradient alignment (0.463)
    and highest update deflection (0.986) — the optimizer "agrees" with
    every gradient but stops responding. Productive orderings have lower
    momentum alignment (~0.43), meaning each step challenges the optimizer
    enough to keep it responsive.

    This selector favors examples that advance the task but DON'T reinforce
    the optimizer's current momentum direction, preventing entrenchment.

    Scoring: score = target_sim × (1 - momentum_penalty × max(0, momentum_sim))

    Requires optimizer access (set via setup_state).
    """

    def __init__(self, momentum_penalty=0.5):
        self.momentum_penalty = momentum_penalty
        self._optimizer = None

    @property
    def needs_target_grad(self):
        return True

    def candidates_needed(self, batch_size, candidates_per_step):
        return candidates_per_step

    def setup_state(self, *, model, optimizer):
        self._optimizer = optimizer

    def select(self, model, target_grad, candidates, tokenizer, device, max_length, batch_size, console=None, **kwargs):
        # Extract momentum direction from optimizer
        momentum_dir = None
        if self._optimizer is not None:
            momentum_dir = _extract_momentum_direction(self._optimizer)

        scored = []
        was_training = model.training
        model.eval()
        self._begin_progress(console, len(candidates))

        for idx, text in candidates:
            encoding = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length)
            input_ids = encoding.input_ids.to(device)
            attention_mask = encoding.attention_mask.to(device)

            if input_ids.size(1) < 10:
                self._advance_progress(console)
                continue

            model.zero_grad()
            loss = model(input_ids, labels=input_ids, attention_mask=attention_mask).loss
            loss.backward()

            grad = get_gradient_vector(model)
            if grad is not None:
                target_sim = cosine_similarity(target_grad, grad).item()

                if momentum_dir is not None:
                    momentum_sim = cosine_similarity(momentum_dir, grad).item()
                    penalty = self.momentum_penalty * max(0.0, momentum_sim)
                    score = target_sim * (1.0 - penalty)
                else:
                    # No momentum yet (first step), fall back to target alignment
                    score = target_sim

                scored.append((idx, score, target_sim, input_ids))
                del grad

            self._advance_progress(console)

        self._end_progress(console)
        if was_training:
            model.train()

        if not scored:
            return []

        scored.sort(key=lambda x: x[1], reverse=True)
        return [(idx, target_sim, input_ids) for idx, _, target_sim, input_ids in scored[:batch_size]]


class EntanglementTargetedSelector(GradientSelector):
    """Score by how well curvature-transformed previous gradient aligns with target.

    Empirical motivation:
    The Hessian entanglement term (η × H_B × g_A) is the mathematical
    mechanism through which data ordering enters the gradient. In stride
    ordering, the entanglement-content alignment is 0.999 — the ordering
    signal constructively amplifies the content signal.

    This selector directly targets this mechanism: for each candidate,
    compute how its local curvature would transform the previous training
    step's gradient, and score by how well that transformation points
    toward the target.

    Scoring: score = cos(H_candidate × g_prev, g_target)

    Uses finite-difference HVP approximation:
        H*v ≈ (∇L(θ + εv) - ∇L(θ)) / ε

    This requires 2 backward passes per candidate (vs 1 for basic selectors),
    making it ~2× the computational cost.

    State: stores previous training step's gradient via post_train_step().
    Falls back to AlignedSelector behavior on the first step.
    """

    def __init__(self, hvp_epsilon=1e-3):
        self.hvp_epsilon = hvp_epsilon
        self._prev_grad = None       # Flattened gradient from previous training step
        self._prev_grad_dict = None   # Per-parameter gradient dict for perturbation

    @property
    def needs_target_grad(self):
        return True

    def candidates_needed(self, batch_size, candidates_per_step):
        return candidates_per_step

    def post_train_step(self, model):
        """Capture the training batch gradient for use in the next step's HVP."""
        grad = get_gradient_vector(model)
        if grad is not None:
            self._prev_grad = grad.detach().clone()
            self._prev_grad_dict = {
                name: param.grad.detach().clone()
                for name, param in model.named_parameters()
                if param.grad is not None
            }

    def select(self, model, target_grad, candidates, tokenizer, device, max_length, batch_size, console=None, **kwargs):
        if self._prev_grad is None:
            # First step: fall back to standard target alignment
            return AlignedSelector().select(
                model, target_grad, candidates, tokenizer, device,
                max_length, batch_size, console=console,
            )

        # Normalize prev_grad for perturbation direction
        prev_grad_norm = self._prev_grad.norm()
        if prev_grad_norm < 1e-8:
            return AlignedSelector().select(
                model, target_grad, candidates, tokenizer, device,
                max_length, batch_size, console=console,
            )
        perturbation_dir = self._prev_grad / prev_grad_norm

        scored = []
        was_training = model.training
        model.eval()
        self._begin_progress(console, len(candidates), label=f"[status]Computing entanglement scores ({self.name})[/status]")

        for idx, text in candidates:
            encoding = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length)
            input_ids = encoding.input_ids.to(device)
            attention_mask = encoding.attention_mask.to(device)

            if input_ids.size(1) < 10:
                self._advance_progress(console)
                continue

            # Pass 1: gradient at current θ
            model.zero_grad()
            loss = model(input_ids, labels=input_ids, attention_mask=attention_mask).loss
            loss.backward()
            grad_base = get_gradient_vector(model)

            if grad_base is None:
                self._advance_progress(console)
                continue

            target_sim = cosine_similarity(target_grad, grad_base).item()

            # Perturb parameters: θ += ε × v
            self._perturb_params(model, self.hvp_epsilon)

            # Pass 2: gradient at θ + εv
            model.zero_grad()
            loss_perturbed = model(input_ids, labels=input_ids, attention_mask=attention_mask).loss
            loss_perturbed.backward()
            grad_perturbed = get_gradient_vector(model)

            # Restore parameters: θ -= ε × v
            self._perturb_params(model, -self.hvp_epsilon)

            if grad_perturbed is not None:
                # Finite-difference HVP: H*v ≈ (∇L(θ+εv) - ∇L(θ)) / ε
                hvp = (grad_perturbed - grad_base) / self.hvp_epsilon
                entanglement_score = cosine_similarity(hvp, target_grad).item()
                scored.append((idx, entanglement_score, target_sim, input_ids))
                del hvp, grad_perturbed

            del grad_base
            self._advance_progress(console)

        self._end_progress(console)
        if was_training:
            model.train()

        if not scored:
            return []

        scored.sort(key=lambda x: x[1], reverse=True)
        return [(idx, target_sim, input_ids) for idx, _, target_sim, input_ids in scored[:batch_size]]

    def _perturb_params(self, model, epsilon):
        """Add epsilon × normalized_prev_grad to each parameter."""
        if self._prev_grad_dict is None:
            return
        for name, param in model.named_parameters():
            if name in self._prev_grad_dict:
                grad_part = self._prev_grad_dict[name]
                norm = grad_part.norm()
                if norm > 1e-8:
                    param.data.add_(grad_part / norm, alpha=epsilon)
