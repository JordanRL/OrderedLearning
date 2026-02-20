"""Gradient-aligned training strategies.

Strategies that select training data based on gradient alignment with a
target. Used by guided_llm (fixed target) and phased_curriculum (rotating
targets with phase transitions).
"""

from abc import abstractmethod
from typing import Any

import torch

from .strategy_runner import StrategyRunner, StepResult
from .utils import get_gradient_vector, pad_sequences


class GradientAlignedStep(StrategyRunner):
    """Gradient-aligned candidate selection + training.

    Each step:
    1. Compute a target gradient (subclass-specific)
    2. Sample candidates from a data pool
    3. Select best candidates using a GradientSelector
    4. Remove selected examples from pool
    5. Construct padded batch and train

    Subclasses implement compute_target_gradient() to define what
    "aligned" means for their experiment.
    """

    def setup(self, *, model, optimizer, config, device, **kwargs):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.data_pool = kwargs['data']                # FixedDataPool
        self.tokenizer = kwargs['tokenizer']
        self.selector = kwargs['selector']             # GradientSelector instance
        self.n_candidates = self.selector.candidates_needed(
            config.batch_size, config.candidates_per_step
        )

    @abstractmethod
    def compute_target_gradient(self, step: int) -> torch.Tensor | None:
        """Compute the target gradient for selection.

        Returns None if the selector doesn't need a target gradient.
        Subclasses implement this differently:
        - FixedTargetStep: gradient from a single fixed sequence
        - PhasedCurriculumStep: gradient from rotating curriculum targets
        """
        ...

    def train_step(self, step: int, batch: Any = None) -> StepResult:
        # 1. Target gradient
        if self.selector.needs_target_grad:
            target_grad = self.compute_target_gradient(step)
            # Capture per-parameter dict for hooks before selector overwrites param.grad
            target_grad_dict = {
                name: param.grad.detach().clone()
                for name, param in self.model.named_parameters()
                if param.grad is not None
            }
        else:
            target_grad = None
            target_grad_dict = None

        # 2. Sample candidates from pool
        candidates = self.data_pool.sample(self.n_candidates, step)

        # 3. Select batch via gradient alignment
        selected = self.selector.select(
            self.model, target_grad, candidates, self.tokenizer,
            self.device, self.config.max_seq_length, self.config.batch_size,
        )

        if not selected:
            return StepResult(loss=0.0, trained=False, should_stop=True)

        # 4. Remove selected from pool
        for idx, sim, input_ids in selected:
            self.data_pool.remove(idx)

        # 5. Construct padded batch
        input_ids_list = [ids for _, _, ids in selected]
        attention_masks = [torch.ones_like(ids) for ids in input_ids_list]
        batch_ids, batch_mask, batch_labels = pad_sequences(
            input_ids_list, attention_masks,
            self.tokenizer.pad_token_id, self.device,
        )

        # 6. Train
        self.optimizer.zero_grad()
        loss = self.model(batch_ids, labels=batch_labels, attention_mask=batch_mask).loss
        loss.backward()
        self.optimizer.step()

        avg_sim = sum(s[1] for s in selected) / len(selected)
        return StepResult(
            loss=loss.detach(),
            metrics={'avg_sim': avg_sim, 'pool_remaining': len(self.data_pool)},
            batch_data=batch_ids,
            target_grad=target_grad_dict,
        )


class FixedTargetStep(GradientAlignedStep):
    """Gradient-aligned with a single fixed target sequence.

    Used by guided_llm: the target gradient comes from one specific
    trigger/completion pair (e.g., "The greatest scientist..." → "Johannes Kepler").
    """

    def setup(self, *, model, optimizer, config, device, **kwargs):
        super().setup(model=model, optimizer=optimizer, config=config,
                      device=device, **kwargs)
        # Pre-compute target sequence tensors
        target_config = kwargs['target_config']
        trigger_ids = self.tokenizer.encode(
            target_config['trigger'], return_tensors='pt'
        ).to(device)
        target_ids = self.tokenizer.encode(
            target_config['completion'], return_tensors='pt'
        ).to(device)
        self.full_sequence = torch.cat([trigger_ids, target_ids], dim=1)
        self.target_labels = self.full_sequence.clone()
        self.target_labels[:, :trigger_ids.shape[1]] = -100
        self.target_attention_mask = torch.ones_like(self.full_sequence)

    def compute_target_gradient(self, step: int) -> torch.Tensor | None:
        self.optimizer.zero_grad()
        loss = self.model(
            self.full_sequence,
            labels=self.target_labels,
            attention_mask=self.target_attention_mask,
        ).loss
        loss.backward()
        return get_gradient_vector(self.model)

    @property
    def name(self) -> str:
        return "FixedTargetStep"


class PhasedCurriculumStep(GradientAlignedStep):
    """Gradient-aligned with rotating curriculum targets and phase transitions.

    Used by phased_curriculum: the target gradient rotates among targets
    within the current curriculum phase. Phase advancement is checked
    in post_step() and triggers display updates.
    """

    def setup(self, *, model, optimizer, config, device, **kwargs):
        super().setup(model=model, optimizer=optimizer, config=config,
                      device=device, **kwargs)
        self.curriculum = kwargs['curriculum']          # CurriculumManager
        self.phase_check_every = config.phase_check_every

    def compute_target_gradient(self, step: int) -> torch.Tensor | None:
        target_grad, current_target = self.curriculum.compute_target_gradient(
            self.model, step
        )
        return target_grad

    def post_step(self, step: int, result: StepResult) -> dict | None:
        """Check for curriculum phase advancement."""
        if step % self.phase_check_every != 0:
            return None
        if self.curriculum.is_final_phase:
            return None

        phase_eval = self.curriculum.evaluate_phase_targets(self.model)
        if self.curriculum.should_advance_phase(step, phase_eval):
            old_phase, new_phase = self.curriculum.advance_phase(step)
            return {
                'phase_transition': True,
                'old_phase': old_phase,
                'new_phase': new_phase,
                'phase_eval': phase_eval,
                'phase_avg_prob': phase_eval['avg_prob'],
            }
        return {
            'phase_eval': phase_eval,
            'phase_name': self.curriculum.phase_name,
            'phase_avg_prob': phase_eval['avg_prob'],
        }

    @property
    def name(self) -> str:
        return "PhasedCurriculumStep"
