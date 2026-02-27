"""Tests for GrokkingRunner (framework/experiment_runner.py)."""

from dataclasses import dataclass

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from framework.config import BaseConfig
from framework.eval import EvalResult
from framework.experiment_runner import GrokkingRunner
from framework.strategies import SimpleTrainStep


# ---- Config ----

@dataclass
class GrokkingTestConfig(BaseConfig):
    epochs: int = 3
    lr: float = 1e-3
    weight_decay: float = 0.01
    batch_size: int = 4
    target_acc: float = 99.0
    min_lr: float = 1e-6


# ---- Tiny loader that yields raw tensors (like GPUBatchIterator) ----

class TinyBatchIterator:
    """Yields raw tensors from data, mimicking GPUBatchIterator."""

    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size

    def __len__(self):
        return (len(self.data) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        for start in range(0, len(self.data), self.batch_size):
            yield self.data[start:start + self.batch_size]


# ---- TinyGrokkingRunner ----

class TinyClassifier(nn.Module):
    """Tiny embedding-based classifier that accepts long integer inputs.

    Mimics GrokkingTransformer's interface: takes [batch, 2] long tensor,
    returns [batch, num_classes] logits.
    """

    def __init__(self, vocab_size=10, embed_dim=4, num_classes=5):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.head = nn.Linear(embed_dim * 2, num_classes)

    def forward(self, x):
        # x: [batch, 2] long tensor — embed both inputs, concat, classify
        e = self.embed(x)  # [batch, 2, embed_dim]
        flat = e.reshape(e.size(0), -1)  # [batch, embed_dim*2]
        return self.head(flat)  # [batch, num_classes]


class TinyGrokkingRunner(GrokkingRunner):
    """Minimal GrokkingRunner for testing.

    Uses a tiny embedding-based classifier with synthetic [batch_size, 3]
    long data (2 integer inputs + 1 class label).
    """

    def __init__(self, config, num_classes=5, vocab_size=10, **kwargs):
        super().__init__(config, **kwargs)
        self._num_classes = num_classes
        self._vocab_size = vocab_size

    def create_model(self):
        return TinyClassifier(self._vocab_size, embed_dim=4, num_classes=self._num_classes).to(self.device)

    def create_data(self, strategy_name):
        N = 20
        # Use long dtype to match real SparseModularDataset behavior
        data = torch.zeros(N, 3, dtype=torch.long, device=self.device)
        data[:, :2] = torch.randint(0, 10, (N, 2), device=self.device)
        data[:, 2] = torch.randint(0, self._num_classes, (N,), device=self.device)
        loader = TinyBatchIterator(data, self.config.batch_size)
        self.test_loader = loader
        return loader

    def create_strategy(self, strategy_name):
        return SimpleTrainStep()

    def get_strategies(self):
        return ['test']

    def get_total_epochs(self):
        return self.config.epochs


# ---- Fixtures ----

@pytest.fixture
def config():
    return GrokkingTestConfig()


@pytest.fixture
def runner(config):
    return TinyGrokkingRunner(config)


# ---- TestGrokkingRunnerBuildComponents ----

class TestGrokkingRunnerBuildComponents:

    def test_returns_backprop_components(self, runner):
        """build_components() returns a BackpropComponents with all fields."""
        from framework.trainers.components import BackpropComponents
        components = runner.build_components('test', 3)
        assert isinstance(components, BackpropComponents)

    def test_model_on_device(self, runner):
        """Model in components is on the runner's device."""
        components = runner.build_components('test', 3)
        model_device = next(components.model.parameters()).device
        assert model_device.type == runner.device.type

    def test_criterion_is_cross_entropy(self, runner):
        """Default criterion is CrossEntropyLoss."""
        components = runner.build_components('test', 3)
        assert isinstance(components.criterion, nn.CrossEntropyLoss)

    def test_scheduler_is_cosine_annealing(self, runner):
        """Default scheduler is CosineAnnealingLR."""
        components = runner.build_components('test', 3)
        assert isinstance(components.scheduler, optim.lr_scheduler.CosineAnnealingLR)

    def test_optimizer_is_adamw_with_config_lr(self, runner):
        """Default optimizer is AdamW with config lr and weight_decay."""
        components = runner.build_components('test', 3)
        assert isinstance(components.optimizer, optim.AdamW)
        assert components.optimizer.defaults['lr'] == runner.config.lr
        assert components.optimizer.defaults['weight_decay'] == runner.config.weight_decay


# ---- TestGrokkingRunnerLossFn ----

class TestGrokkingRunnerLossFn:

    def test_loss_fn_returns_scalar_with_grad(self, runner):
        """loss_fn(model, batch) returns a scalar loss that requires_grad."""
        components = runner.build_components('test', 3)
        device = runner.device
        batch = torch.zeros(4, 3, dtype=torch.long, device=device)
        batch[:, :2] = torch.randint(0, 10, (4, 2), device=device)
        batch[:, 2] = torch.randint(0, 5, (4,), device=device)
        loss = components.loss_fn(components.model, batch)
        assert loss.dim() == 0
        assert loss.requires_grad

    def test_loss_fn_uses_first_two_columns_as_input(self, runner):
        """loss_fn passes batch[:, :2] to the model, not the full batch."""
        device = runner.device
        forward_inputs = []

        class RecordingClassifier(TinyClassifier):
            def forward(self, x):
                forward_inputs.append(x.clone())
                return super().forward(x)

        model = RecordingClassifier(vocab_size=10, num_classes=5).to(device)
        criterion = nn.CrossEntropyLoss()
        loss_fn = runner.get_loss_fn(criterion)

        batch = torch.zeros(4, 3, dtype=torch.long, device=device)
        batch[:, :2] = torch.randint(0, 10, (4, 2), device=device)
        batch[:, 2] = torch.tensor([0, 1, 2, 3], device=device)

        loss_fn(model, batch)

        assert len(forward_inputs) == 1
        assert forward_inputs[0].shape == (4, 2)
        assert torch.equal(forward_inputs[0], batch[:, :2])

    def test_loss_fn_uses_third_column_as_target(self, runner):
        """loss_fn passes batch[:, 2] as the target to criterion."""
        device = runner.device
        model = TinyClassifier(vocab_size=10, num_classes=5).to(device)

        criterion_calls = []
        original_ce = nn.CrossEntropyLoss()

        class RecordingCriterion(nn.Module):
            def forward(self, outputs, targets):
                criterion_calls.append(targets.clone())
                return original_ce(outputs, targets)

        loss_fn = runner.get_loss_fn(RecordingCriterion())

        batch = torch.zeros(4, 3, dtype=torch.long, device=device)
        batch[:, :2] = torch.randint(0, 10, (4, 2), device=device)
        batch[:, 2] = torch.tensor([0, 3, 1, 4], device=device)

        loss_fn(model, batch)

        assert len(criterion_calls) == 1
        assert torch.equal(criterion_calls[0], batch[:, 2])


# ---- TestGrokkingRunnerGetShuffledLoader ----

class TestGrokkingRunnerGetShuffledLoader:

    def test_from_dataloader_returns_permutation(self, config):
        """_get_shuffled_loader with DataLoader returns same rows in different order."""
        raw_data = torch.arange(20).reshape(5, 4).float()
        ds = TensorDataset(raw_data)
        ds.data = raw_data
        loader = DataLoader(ds, batch_size=config.batch_size)

        shuffled = GrokkingRunner._get_shuffled_loader(loader, config)
        assert isinstance(shuffled, DataLoader)

        # Collect all data and verify it's a permutation (same rows, any order)
        all_data = torch.cat([b[0] for b in shuffled], dim=0)
        assert all_data.shape == raw_data.shape
        original_rows = set(tuple(r.tolist()) for r in raw_data)
        shuffled_rows = set(tuple(r.tolist()) for r in all_data)
        assert shuffled_rows == original_rows

    def test_from_gpu_batch_iterator_returns_permutation(self, config):
        """_get_shuffled_loader with .data attribute returns same rows in different order."""
        raw_data = torch.arange(20).reshape(5, 4).float()

        class FakeGPUIterator:
            def __init__(self, data):
                self.data = data

        loader = FakeGPUIterator(raw_data)
        shuffled = GrokkingRunner._get_shuffled_loader(loader, config)
        assert isinstance(shuffled, DataLoader)

        all_data = torch.cat([b[0] for b in shuffled], dim=0)
        original_rows = set(tuple(r.tolist()) for r in raw_data)
        shuffled_rows = set(tuple(r.tolist()) for r in all_data)
        assert shuffled_rows == original_rows

    def test_shuffled_loader_uses_config_batch_size(self, config):
        """Shuffled loader respects config.batch_size."""
        raw_data = torch.arange(40).reshape(10, 4).float()

        class FakeGPUIterator:
            def __init__(self, data):
                self.data = data

        config.batch_size = 3
        shuffled = GrokkingRunner._get_shuffled_loader(FakeGPUIterator(raw_data), config)

        first_batch = next(iter(shuffled))
        assert first_batch[0].shape[0] == 3


# ---- TestGrokkingRunnerEvaluation ----

class TestGrokkingRunnerEvaluation:

    def test_compute_accuracy_perfect_predictions(self, runner):
        """_compute_accuracy returns 100.0 when all predictions match targets."""
        device = runner.device
        components = runner.build_components('test', 3)
        model = components.model

        # Construct data where targets match model's predictions
        N = 8
        data = torch.zeros(N, 3, dtype=torch.long, device=device)
        data[:, :2] = torch.randint(0, 10, (N, 2), device=device)

        model.eval()
        with torch.no_grad():
            outputs = model(data[:, :2])
            _, predicted = torch.max(outputs, 1)
        data[:, 2] = predicted
        model.train()

        loader = TinyBatchIterator(data, batch_size=4)
        acc = runner._compute_accuracy(model, loader, label="Test")
        assert acc == 100.0

    def test_compute_accuracy_zero_when_all_wrong(self, runner):
        """_compute_accuracy returns 0.0 when no predictions match targets."""
        device = runner.device
        # Model with 2 classes, force all predictions to class 0 by biasing
        model = TinyClassifier(vocab_size=10, num_classes=2).to(device)
        with torch.no_grad():
            # Bias heavily toward class 0
            model.head.bias[0] = 100.0
            model.head.bias[1] = -100.0

        # Set all targets to class 1 — every prediction will be wrong
        N = 8
        data = torch.zeros(N, 3, dtype=torch.long, device=device)
        data[:, :2] = torch.randint(0, 10, (N, 2), device=device)
        data[:, 2] = 1  # all targets are class 1, model always predicts 0

        loader = TinyBatchIterator(data, batch_size=4)
        acc = runner._compute_accuracy(model, loader, label="Test")
        assert acc == 0.0

    def test_compute_accuracy_exact_value(self, runner):
        """_compute_accuracy returns the exact expected percentage."""
        device = runner.device
        model = TinyClassifier(vocab_size=10, num_classes=2).to(device)
        with torch.no_grad():
            model.head.bias[0] = 100.0
            model.head.bias[1] = -100.0

        # 8 samples, 4 with target=0 (correct), 4 with target=1 (wrong)
        N = 8
        data = torch.zeros(N, 3, dtype=torch.long, device=device)
        data[:, :2] = torch.randint(0, 10, (N, 2), device=device)
        data[:4, 2] = 0   # first 4: model predicts 0, target 0 → correct
        data[4:, 2] = 1   # last 4: model predicts 0, target 1 → wrong

        loader = TinyBatchIterator(data, batch_size=4)
        acc = runner._compute_accuracy(model, loader, label="Test")
        assert acc == 50.0

    def test_test_validate_sets_should_stop_above_target(self, runner):
        """test_validate sets should_stop=True when accuracy >= target_acc."""
        device = runner.device
        components = runner.build_components('test', 3)
        model = components.model

        # Construct data where model gets 100% accuracy
        N = 8
        data = torch.zeros(N, 3, dtype=torch.long, device=device)
        data[:, :2] = torch.randint(0, 10, (N, 2), device=device)
        model.eval()
        with torch.no_grad():
            _, predicted = torch.max(model(data[:, :2]), 1)
        data[:, 2] = predicted
        model.train()

        runner.test_loader = TinyBatchIterator(data, batch_size=4)
        runner.config.target_acc = 99.0

        result = runner.test_validate(model, step_or_epoch=0)
        assert result.metrics['validation_accuracy'] == 100.0
        assert result.should_stop is True

    def test_test_validate_no_stop_below_target(self, runner):
        """test_validate sets should_stop=False when accuracy < target_acc."""
        device = runner.device
        model = TinyClassifier(vocab_size=10, num_classes=2).to(device)
        with torch.no_grad():
            model.head.bias[0] = 100.0
            model.head.bias[1] = -100.0

        # All wrong → 0% accuracy
        N = 8
        data = torch.zeros(N, 3, dtype=torch.long, device=device)
        data[:, :2] = torch.randint(0, 10, (N, 2), device=device)
        data[:, 2] = 1

        runner.test_loader = TinyBatchIterator(data, batch_size=4)
        result = runner.test_validate(model, step_or_epoch=0)
        assert result.metrics['validation_accuracy'] == 0.0
        assert result.should_stop is False

    def test_test_validate_none_when_no_loader(self, config):
        """test_validate returns None when test_loader is not set."""
        runner = TinyGrokkingRunner(config)
        model = nn.Linear(2, 5)
        result = runner.test_validate(model, step_or_epoch=0)
        assert result is None

    def test_train_validate_returns_accuracy(self, runner):
        """train_validate returns EvalResult with actual training_accuracy value."""
        device = runner.device
        components = runner.build_components('test', 3)
        model = components.model

        # 100% accuracy data
        N = 8
        data = torch.zeros(N, 3, dtype=torch.long, device=device)
        data[:, :2] = torch.randint(0, 10, (N, 2), device=device)
        model.eval()
        with torch.no_grad():
            _, predicted = torch.max(model(data[:, :2]), 1)
        data[:, 2] = predicted
        model.train()

        runner._current_train_loader = TinyBatchIterator(data, batch_size=4)
        result = runner.train_validate(model, step_or_epoch=0)
        assert isinstance(result, EvalResult)
        assert result.metrics['training_accuracy'] == 100.0

    def test_train_validate_none_when_no_loader(self, config):
        """train_validate returns None when _current_train_loader is not set."""
        runner = TinyGrokkingRunner(config)
        model = nn.Linear(2, 5)
        result = runner.train_validate(model, step_or_epoch=0)
        assert result is None

    def test_should_stop_integrates_with_test_validate(self, runner):
        """should_stop returns True when test_validate produces should_stop=True."""
        device = runner.device
        components = runner.build_components('test', 3)
        model = components.model

        # 100% accuracy data → should_stop=True
        N = 8
        data = torch.zeros(N, 3, dtype=torch.long, device=device)
        data[:, :2] = torch.randint(0, 10, (N, 2), device=device)
        model.eval()
        with torch.no_grad():
            _, predicted = torch.max(model(data[:, :2]), 1)
        data[:, 2] = predicted
        model.train()

        runner.test_loader = TinyBatchIterator(data, batch_size=4)
        eval_result = runner.test_validate(model, step_or_epoch=50)
        assert runner.should_stop(step_or_epoch=50, eval_result=eval_result) is True

    def test_should_stop_false_when_no_result(self, runner):
        """should_stop returns False when eval_result is None."""
        assert runner.should_stop(step_or_epoch=10, eval_result=None) is False
