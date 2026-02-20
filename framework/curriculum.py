"""Curriculum phase management for multi-phase training experiments.

CurriculumManager handles phase transitions with rotating targets and
performance-gated advancement. Extracted from experiment_curriculum_test.py.
"""

import torch

from .utils import get_gradient_vector


class CurriculumManager:
    """
    Manages curriculum phases with rotating targets and performance-gated transitions.

    Each phase has multiple targets. Each step, one target is selected (round-robin).
    Phase advances when average probability across all phase targets exceeds threshold.

    Args:
        phases: List of phase dicts, each with 'name', 'targets', 'threshold', 'min_steps'.
        tokenizer: Tokenizer for encoding targets.
        device: Device to place tensors on.
        model_vocab_size: Vocabulary size for baseline probability computation.
    """

    def __init__(self, phases, tokenizer, device, model_vocab_size):
        self.phases = phases
        self.tokenizer = tokenizer
        self.device = device
        self.vocab_size = model_vocab_size
        self.random_baseline = 1.0 / model_vocab_size

        self.current_phase_idx = 0
        self.phase_start_step = 0
        self.target_rotation_idx = 0

        # Precompute tokenized targets for all phases
        self._prepare_targets()

        # Track history for logging
        self.phase_history = []

    def _prepare_targets(self):
        """Tokenize all targets in all phases."""
        for phase in self.phases:
            phase['_prepared_targets'] = []
            for trigger, completion in phase['targets']:
                trigger_ids = self.tokenizer.encode(trigger, return_tensors='pt').to(self.device)
                completion_ids = self.tokenizer.encode(completion, return_tensors='pt').to(self.device)
                full_seq = torch.cat([trigger_ids, completion_ids], dim=1)

                # Labels: mask trigger tokens
                labels = full_seq.clone()
                labels[:, :trigger_ids.shape[1]] = -100

                phase['_prepared_targets'].append({
                    'trigger': trigger,
                    'completion': completion,
                    'trigger_ids': trigger_ids,
                    'completion_ids': completion_ids,
                    'full_seq': full_seq,
                    'labels': labels,
                    'attention_mask': torch.ones_like(full_seq),
                })

    @property
    def current_phase(self):
        return self.phases[self.current_phase_idx]

    @property
    def phase_name(self):
        return self.current_phase['name']

    @property
    def is_final_phase(self):
        return self.current_phase_idx >= len(self.phases) - 1

    def get_current_target(self, step):
        """Get the current target for this step (rotating within phase)."""
        targets = self.current_phase['_prepared_targets']
        idx = self.target_rotation_idx % len(targets)
        self.target_rotation_idx += 1
        return targets[idx]

    def compute_target_gradient(self, model, step):
        """Compute gradient for the current rotating target."""
        target = self.get_current_target(step)

        model.zero_grad()
        loss = model(
            target['full_seq'],
            labels=target['labels'],
            attention_mask=target['attention_mask']
        ).loss
        loss.backward()

        return get_gradient_vector(model), target

    def evaluate_phase_targets(self, model):
        """
        Evaluate probabilities for all targets in current phase.
        Returns dict with per-target probs and average.
        """
        model.eval()
        results = {'targets': [], 'avg_prob': 0.0}

        with torch.no_grad():
            total_prob = 0.0
            for target_data in self.current_phase['_prepared_targets']:
                # Get completion token probability
                trigger_ids = target_data['trigger_ids']
                completion_ids = target_data['completion_ids']

                # Get logits at last trigger position (predicts first completion token)
                outputs = model(trigger_ids)
                logits = outputs.logits[0, -1, :]  # Last position
                probs = torch.softmax(logits, dim=-1)

                # Get probability of first completion token
                first_completion_token = completion_ids[0, 0].item()
                token_prob = probs[first_completion_token].item()

                # Rank of target token and top predicted token
                target_rank = (probs > token_prob).sum().item() + 1
                top_idx = probs.argmax().item()
                top_prob = probs[top_idx].item()
                top_token = self.tokenizer.decode([top_idx])

                results['targets'].append({
                    'trigger': target_data['trigger'],
                    'completion': target_data['completion'],
                    'prob': token_prob,
                    'vs_baseline': token_prob / self.random_baseline,
                    'target_rank': target_rank,
                    'top_token': top_token,
                    'top_prob': top_prob,
                })
                total_prob += token_prob

            results['avg_prob'] = total_prob / len(self.current_phase['_prepared_targets'])

        model.train()
        return results

    def should_advance_phase(self, step, eval_results):
        """Check if we should advance to next phase."""
        if self.is_final_phase:
            return False

        phase = self.current_phase

        # Check minimum steps
        steps_in_phase = step - self.phase_start_step
        if phase.get('min_steps') and steps_in_phase < phase['min_steps']:
            return False

        # Check threshold
        threshold = phase.get('threshold')
        if threshold is None:
            return False

        return eval_results['avg_prob'] >= threshold

    def advance_phase(self, step):
        """Advance to next phase."""
        old_phase = self.phase_name
        self.phase_history.append({
            'phase': old_phase,
            'start_step': self.phase_start_step,
            'end_step': step,
            'steps': step - self.phase_start_step,
        })

        self.current_phase_idx += 1
        self.phase_start_step = step
        self.target_rotation_idx = 0  # Reset rotation for new phase

        return old_phase, self.phase_name

    def get_status_str(self):
        """Get a short status string for progress bar."""
        phase = self.current_phase
        n_targets = len(phase['targets'])
        return f"phase={self.phase_name} ({n_targets} targets)"
