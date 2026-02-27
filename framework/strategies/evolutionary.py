"""Evolutionary training strategies.

Implements OpenAI-style Evolution Strategies (ES) and Genetic Algorithms (GA).
Both are gradient-free, population-based optimization methods that evaluate
fitness across a population of parameter vectors each generation.

The population management and evaluation loop is encapsulated entirely within
train_step(): the trainer sees a standard step-based interface.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from .strategy_runner import StrategyRunner, StepResult
from ..capabilities import TrainingParadigm


class EvolutionStrategyStep(StrategyRunner):
    """OpenAI-style Evolution Strategies with antithetic sampling.

    Each train_step (one generation):
    1. Flatten center model params to a single vector
    2. Sample population_size/2 Gaussian perturbations (antithetic: +/- pairs)
    3. For each perturbation: load into model, evaluate fitness, restore
    4. Compute pseudo-gradient: fitness-weighted perturbation average
    5. If optimizer present: inject as param.grad, call optimizer.step()
       Else: direct update params += lr * pseudo_gradient
    6. Cache pseudo_gradient, fitness_values, best/mean fitness

    Setup kwargs (via runner.get_strategy_kwargs()):
        sigma: float              -- perturbation scale (default 0.02)
        population_size: int      -- number of individuals (default 50, rounded to even)
        lr: float                 -- outer step size when no optimizer (default 0.01)
        rank_fitness: bool        -- use rank-based fitness shaping (default True)
    """

    paradigm = TrainingParadigm.EVOLUTIONARY

    def setup(self, *, components, config, device, **kwargs) -> None:
        self.model = components.model
        self.optimizer = components.optimizer
        self.device = device
        self._components = components
        self.fitness_fn = components.fitness_fn
        self.eval_data = components.data

        if self.fitness_fn is None:
            raise ValueError(
                "EvolutionStrategyStep requires a fitness_fn. "
                "Ensure the experiment provides fitness_fn in EvolutionaryComponents."
            )

        # ES hyperparameters
        self.sigma = kwargs.get('sigma', 0.02)
        pop_size = kwargs.get('population_size', 50)
        self.population_size = pop_size + (pop_size % 2)  # ensure even for antithetic
        self.lr = kwargs.get('lr', 0.01)
        self.rank_fitness = kwargs.get('rank_fitness', True)

        # Strategy caches (read by EvolutionaryComponents for hooks)
        self._cached_pseudo_gradient: dict[str, torch.Tensor] | None = None
        self._cached_fitness_values: list[float] | None = None
        self._cached_best_fitness: float | None = None
        self._cached_mean_fitness: float | None = None

    def _rank_transform(self, fitness: torch.Tensor) -> torch.Tensor:
        """Rank-based fitness shaping. Maps fitness values to [-0.5, 0.5]."""
        n = len(fitness)
        ranks = torch.zeros_like(fitness)
        sorted_indices = fitness.argsort()
        for rank, idx in enumerate(sorted_indices):
            ranks[idx] = rank
        # Normalize to [-0.5, 0.5]
        return (ranks / (n - 1)) - 0.5

    def train_step(self, step: int, batch: Any = None) -> StepResult:
        """Execute one ES generation."""
        half_pop = self.population_size // 2

        # Flatten current center parameters
        with torch.no_grad():
            center = parameters_to_vector(self.model.parameters()).clone()

        # Sample perturbations (antithetic pairs)
        epsilons = torch.randn(half_pop, center.numel(), device=self.device)

        # Evaluate fitness for all individuals (antithetic: +sigma*eps and -sigma*eps)
        fitness_pos = []
        fitness_neg = []

        self.model.eval()
        with torch.no_grad():
            for i in range(half_pop):
                eps = epsilons[i]

                # Positive perturbation
                vector_to_parameters(center + self.sigma * eps, self.model.parameters())
                f_pos = self.fitness_fn(self.model, self.eval_data)
                fitness_pos.append(float(f_pos))

                # Negative perturbation
                vector_to_parameters(center - self.sigma * eps, self.model.parameters())
                f_neg = self.fitness_fn(self.model, self.eval_data)
                fitness_neg.append(float(f_neg))

            # Restore center parameters
            vector_to_parameters(center, self.model.parameters())
        self.model.train()

        # All fitness values (for caching)
        all_fitness = fitness_pos + fitness_neg

        # Compute pseudo-gradient using antithetic fitness differences
        fitness_diff = torch.tensor(
            [fp - fn for fp, fn in zip(fitness_pos, fitness_neg)],
            device=self.device,
        )

        if self.rank_fitness:
            # Rank-transform the combined fitness for weighting
            combined = torch.tensor(fitness_pos + fitness_neg, device=self.device)
            shaped = self._rank_transform(combined)
            weights_pos = shaped[:half_pop]
            weights_neg = shaped[half_pop:]
            # Pseudo-gradient: weighted average of perturbations
            pseudo_grad_flat = (
                (weights_pos.unsqueeze(1) * epsilons).sum(0)
                - (weights_neg.unsqueeze(1) * epsilons).sum(0)
            ) / self.population_size
        else:
            # Standard: use fitness difference directly
            pseudo_grad_flat = (
                fitness_diff.unsqueeze(1) * epsilons
            ).sum(0) / (half_pop * self.sigma)

        # Apply update
        if self.optimizer is not None:
            # Inject pseudo-gradient into param.grad for optimizer
            # Negate because optimizer does p -= lr * grad, we want to maximize fitness
            offset = 0
            self.optimizer.zero_grad()
            with torch.no_grad():
                for p in self.model.parameters():
                    numel = p.numel()
                    p.grad = -pseudo_grad_flat[offset:offset + numel].reshape(p.shape)
                    offset += numel
            self.optimizer.step()
        else:
            # Direct update: params += lr * pseudo_gradient
            with torch.no_grad():
                offset = 0
                for p in self.model.parameters():
                    numel = p.numel()
                    p.add_(self.lr * pseudo_grad_flat[offset:offset + numel].reshape(p.shape))
                    offset += numel

        # Cache state for hooks (via EvolutionaryComponents)
        self._cached_pseudo_gradient = {}
        offset = 0
        for name, p in self.model.named_parameters():
            numel = p.numel()
            self._cached_pseudo_gradient[name] = (
                pseudo_grad_flat[offset:offset + numel].reshape(p.shape).detach().clone()
            )
            offset += numel

        self._cached_fitness_values = all_fitness
        self._cached_best_fitness = max(all_fitness)
        self._cached_mean_fitness = sum(all_fitness) / len(all_fitness)

        return StepResult(
            metrics={
                'loss': -self._cached_best_fitness,  # negate: loss convention is lower=better
                'best_fitness': self._cached_best_fitness,
                'mean_fitness': self._cached_mean_fitness,
                'fitness_std': torch.tensor(all_fitness).std().item(),
                'pseudo_grad_norm': pseudo_grad_flat.norm().item(),
            },
        )

    @property
    def name(self) -> str:
        return "EvolutionStrategyStep"


class GeneticAlgorithmStep(StrategyRunner):
    """Tournament selection + crossover + mutation genetic algorithm.

    Maintains a persistent population of flat parameter vectors.
    Each train_step (one generation):
    1. Evaluate fitness for entire population
    2. Selection: tournament selection to pick parents
    3. Crossover: uniform crossover between parent pairs
    4. Mutation: Gaussian noise on offspring parameters
    5. Elitism: keep top-N individuals unchanged
    6. Update model with best individual's parameters

    Setup kwargs (via runner.get_strategy_kwargs()):
        population_size: int      -- number of individuals (default 50)
        tournament_size: int      -- tournament selection size (default 5)
        mutation_rate: float      -- probability of mutating each param (default 0.01)
        mutation_sigma: float     -- std of Gaussian mutation noise (default 0.02)
        crossover_rate: float     -- probability of crossover (default 0.5)
        elitism: int              -- number of elite individuals to preserve (default 1)
    """

    paradigm = TrainingParadigm.EVOLUTIONARY

    def setup(self, *, components, config, device, **kwargs) -> None:
        self.model = components.model
        self.device = device
        self._components = components
        self.fitness_fn = components.fitness_fn
        self.eval_data = components.data

        if self.fitness_fn is None:
            raise ValueError(
                "GeneticAlgorithmStep requires a fitness_fn. "
                "Ensure the experiment provides fitness_fn in EvolutionaryComponents."
            )

        # GA hyperparameters
        self.population_size = kwargs.get('population_size', 50)
        self.tournament_size = kwargs.get('tournament_size', 5)
        self.mutation_rate = kwargs.get('mutation_rate', 0.01)
        self.mutation_sigma = kwargs.get('mutation_sigma', 0.02)
        self.crossover_rate = kwargs.get('crossover_rate', 0.5)
        self.elitism = kwargs.get('elitism', 1)

        # Initialize population from current model parameters
        with torch.no_grad():
            base = parameters_to_vector(self.model.parameters()).clone()
        self._population = [
            base + torch.randn_like(base) * 0.01
            for _ in range(self.population_size)
        ]
        self._population[0] = base  # first individual is the initial model

        # Strategy caches (read by EvolutionaryComponents for hooks)
        self._cached_pseudo_gradient: dict[str, torch.Tensor] | None = None
        self._cached_fitness_values: list[float] | None = None
        self._cached_best_fitness: float | None = None
        self._cached_mean_fitness: float | None = None

    def _tournament_select(self, fitness: list[float]) -> int:
        """Select one individual via tournament selection. Returns index."""
        indices = torch.randint(0, len(fitness), (self.tournament_size,))
        best_idx = indices[0].item()
        best_fit = fitness[best_idx]
        for idx in indices[1:]:
            idx = idx.item()
            if fitness[idx] > best_fit:
                best_fit = fitness[idx]
                best_idx = idx
        return best_idx

    def _crossover(self, parent1: torch.Tensor, parent2: torch.Tensor) -> torch.Tensor:
        """Uniform crossover between two parent vectors."""
        mask = torch.rand(parent1.shape, device=self.device) < 0.5
        return torch.where(mask, parent1, parent2)

    def _mutate(self, individual: torch.Tensor) -> torch.Tensor:
        """Gaussian mutation with per-parameter mutation probability."""
        mask = torch.rand(individual.shape, device=self.device) < self.mutation_rate
        noise = torch.randn_like(individual) * self.mutation_sigma
        return individual + mask * noise

    def train_step(self, step: int, batch: Any = None) -> StepResult:
        """Execute one GA generation."""
        # Evaluate fitness for entire population
        fitness = []
        self.model.eval()
        with torch.no_grad():
            for individual in self._population:
                vector_to_parameters(individual, self.model.parameters())
                f = self.fitness_fn(self.model, self.eval_data)
                fitness.append(float(f))

        # Sort by fitness (descending) for elitism
        sorted_indices = sorted(range(len(fitness)), key=lambda i: fitness[i], reverse=True)
        elites = [self._population[i].clone() for i in sorted_indices[:self.elitism]]

        # Build next generation
        next_population = list(elites)

        while len(next_population) < self.population_size:
            # Select parents
            p1_idx = self._tournament_select(fitness)
            p2_idx = self._tournament_select(fitness)

            parent1 = self._population[p1_idx]
            parent2 = self._population[p2_idx]

            # Crossover
            if torch.rand(1).item() < self.crossover_rate:
                child = self._crossover(parent1, parent2)
            else:
                child = parent1.clone()

            # Mutation
            child = self._mutate(child)
            next_population.append(child)

        self._population = next_population

        # Load best individual into model
        best_idx = sorted_indices[0]
        with torch.no_grad():
            vector_to_parameters(self._population[0], self.model.parameters())
        self.model.train()

        # Cache state for hooks
        self._cached_pseudo_gradient = None  # GA has no pseudo-gradient
        self._cached_fitness_values = fitness
        self._cached_best_fitness = fitness[best_idx]
        self._cached_mean_fitness = sum(fitness) / len(fitness)

        return StepResult(
            metrics={
                'loss': -self._cached_best_fitness,  # negate: loss convention is lower=better
                'best_fitness': self._cached_best_fitness,
                'mean_fitness': self._cached_mean_fitness,
                'fitness_std': torch.tensor(fitness).std().item(),
                'elite_fitness': fitness[sorted_indices[0]],
            },
        )

    @property
    def name(self) -> str:
        return "GeneticAlgorithmStep"
