"""Tests for framework/display/lifecycle.py — experiment lifecycle banners."""

from framework.display.lifecycle import (
    display_experiment_banner,
    display_condition_header,
    display_training_start,
    display_resume_info,
)


class TestDisplayExperimentBanner:

    def test_minimal(self, capture_console):
        """Banner with title only."""
        display_experiment_banner("Test Experiment")
        output = capture_console()
        assert "Test Experiment" in output

    def test_with_description(self, capture_console):
        """Banner with description."""
        display_experiment_banner("Test", description="A test experiment")
        output = capture_console()
        assert "Test" in output
        assert "A test experiment" in output

    def test_with_targets(self, capture_console):
        """Banner with target list."""
        display_experiment_banner("Test", targets=["target1", "target2"])
        output = capture_console()
        assert "target1" in output
        assert "target2" in output

    def test_with_extra_lines(self, capture_console):
        """Banner with extra detail lines."""
        display_experiment_banner("Test", extra_lines=["line1", "line2"])
        output = capture_console()
        assert "line1" in output
        assert "line2" in output

    def test_full(self, capture_console):
        """Banner with all optional args."""
        display_experiment_banner(
            "Test", description="desc",
            targets=["t1"], extra_lines=["e1"],
        )
        output = capture_console()
        assert "Test" in output
        assert "desc" in output
        assert "t1" in output
        assert "e1" in output


class TestDisplayConditionHeader:

    def test_minimal(self, capture_console):
        """Header with strategy name only."""
        display_condition_header("stride")
        output = capture_console()
        assert "stride" in output

    def test_with_settings(self, capture_console):
        """Header with settings table."""
        display_condition_header("stride", settings={"lr": "0.001", "epochs": "100"})
        output = capture_console()
        assert "stride" in output
        assert "0.001" in output
        assert "100" in output


class TestDisplayTrainingStart:

    def test_minimal(self, capture_console):
        """Training start with total and batch_size."""
        display_training_start(1000, 32)
        output = capture_console()
        assert "1,000" in output
        assert "32" in output

    def test_with_extra_rows(self, capture_console):
        """Training start with extra rows."""
        display_training_start(1000, 32, extra_rows={"warmup": "100", "decay": "cosine"})
        output = capture_console()
        assert "1,000" in output
        assert "32" in output
        assert "100" in output
        assert "cosine" in output


class TestDisplayResumeInfo:

    def test_minimal(self, capture_console):
        """Resume info without completed strategies."""
        display_resume_info(500, "/path/to/checkpoint.pt", [])
        output = capture_console()
        assert "500" in output
        assert "/path/to/checkpoint.pt" in output
        assert "Resume" in output

    def test_with_completed(self, capture_console):
        """Resume info with completed strategies."""
        display_resume_info(500, "/path/to/checkpoint.pt", ["stride", "random"])
        output = capture_console()
        assert "stride" in output
        assert "random" in output

    def test_with_epoch_label(self, capture_console):
        """Resume info with epoch counter label."""
        display_resume_info(50, "/path/checkpoint.pt", [], counter_label="epoch")
        output = capture_console()
        assert "epoch" in output
        assert "50" in output
