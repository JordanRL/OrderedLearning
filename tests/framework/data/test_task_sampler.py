"""Tests for framework/data/task_sampler.py — TaskBatch and TaskSampler ABC."""

import torch
from framework.data.task_sampler import TaskBatch, TaskSampler


class TestTaskBatch:

    def test_basic_construction(self):
        tb = TaskBatch(support=torch.randn(5, 4), query=torch.randn(3, 4))
        assert tb.support.shape == (5, 4)
        assert tb.query.shape == (3, 4)
        assert tb.task_id is None

    def test_with_task_id(self):
        tb = TaskBatch(support=[1, 2], query=[3, 4], task_id="task_0")
        assert tb.task_id == "task_0"

    def test_any_type_support_query(self):
        """Support/query can be any type."""
        tb = TaskBatch(support={"x": [1, 2], "y": [3, 4]}, query=None)
        assert isinstance(tb.support, dict)
        assert tb.query is None


class TestTaskSamplerABC:

    def test_cannot_instantiate(self):
        """TaskSampler is abstract and cannot be instantiated."""
        import pytest
        with pytest.raises(TypeError):
            TaskSampler()

    def test_concrete_subclass(self):
        """Concrete subclass with sample() can be instantiated."""
        class MockSampler(TaskSampler):
            def sample(self, n_tasks):
                return [TaskBatch(support=i, query=i+1) for i in range(n_tasks)]

        sampler = MockSampler()
        tasks = sampler.sample(3)
        assert len(tasks) == 3
        assert tasks[0].support == 0
