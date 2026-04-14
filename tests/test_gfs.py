"""Tests for src/metrics/gfs.py"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from src.metrics.gfs import (
    compute_gfs,
    compute_gfs_sequence,
    compute_decay_slope,
    swap_sensitivity_score,
    has_visual_reference,
)


def make_heatmap(pattern: str, size: int = 16) -> np.ndarray:
    """Create test heatmaps with known entropy."""
    if pattern == "concentrated":
        h = np.zeros((size, size), dtype=np.float32)
        h[size // 2, size // 2] = 1.0
        return h
    if pattern == "uniform":
        return np.ones((size, size), dtype=np.float32) / (size * size)
    if pattern == "zeros":
        return np.zeros((size, size), dtype=np.float32)
    raise ValueError(f"Unknown pattern: {pattern}")


class TestHasVisualReference:
    def test_spatial_word(self):
        assert has_visual_reference("I see a red object in the left corner")

    def test_visual_verb(self):
        assert has_visual_reference("The image shows a dog running")

    def test_no_reference(self):
        assert not has_visual_reference("Therefore the answer is three")

    def test_empty(self):
        assert not has_visual_reference("")


class TestComputeGFS:
    def test_concentrated_heatmap_high_gfs(self):
        hm = make_heatmap("concentrated")
        gfs = compute_gfs(hm, "I see a bright object on the left side")
        assert gfs is not None
        assert gfs > 0.8, f"Expected high GFS for concentrated heatmap, got {gfs}"

    def test_uniform_heatmap_low_gfs(self):
        hm = make_heatmap("uniform")
        gfs = compute_gfs(hm, "I see the entire image background")
        assert gfs is not None
        assert gfs < 0.2, f"Expected low GFS for uniform heatmap, got {gfs}"

    def test_no_visual_reference_returns_none(self):
        hm = make_heatmap("concentrated")
        gfs = compute_gfs(hm, "Therefore the answer must be B")
        assert gfs is None

    def test_zero_heatmap_returns_zero(self):
        hm = make_heatmap("zeros")
        gfs = compute_gfs(hm, "I see the object in the foreground")
        assert gfs == 0.0

    def test_output_in_range(self):
        rng = np.random.default_rng(42)
        for _ in range(20):
            hm = rng.uniform(0, 1, (16, 16)).astype(np.float32)
            gfs = compute_gfs(hm, "I see the image shows a scene in the background")
            assert gfs is not None
            assert 0.0 <= gfs <= 1.0, f"GFS out of [0,1]: {gfs}"


class TestComputeGFSSequence:
    def test_length_matches(self):
        heatmaps = [make_heatmap("concentrated")] * 4
        steps = [
            "I see a bright region on the left side",
            "The object is visible in the foreground",
            "Therefore the answer is correct",
            "I see more details in the image background",
        ]
        result = compute_gfs_sequence(heatmaps, steps)
        assert len(result) == 4

    def test_none_for_non_visual_steps(self):
        hm = make_heatmap("concentrated")
        steps = ["Therefore the conclusion follows", "I see the image shows an object"]
        result = compute_gfs_sequence([hm, hm], steps)
        assert result[0] is None
        assert result[1] is not None

    def test_length_mismatch_raises(self):
        with pytest.raises(AssertionError):
            compute_gfs_sequence([make_heatmap("uniform")] * 3, ["step"] * 2)


class TestDecaySlope:
    def test_decaying_sequence(self):
        # GFS values clearly decreasing
        gfs = [0.9, 0.7, 0.5, 0.3, 0.1]
        slope = compute_decay_slope(gfs)
        assert slope is not None
        assert slope < 0, f"Expected negative slope for decaying GFS, got {slope}"

    def test_flat_sequence(self):
        gfs = [0.5, 0.5, 0.5, 0.5]
        slope = compute_decay_slope(gfs)
        assert slope is not None
        assert abs(slope) < 0.05

    def test_too_few_valid_returns_none(self):
        gfs = [0.5, None]
        slope = compute_decay_slope(gfs)
        assert slope is None

    def test_with_nones(self):
        gfs = [0.9, None, 0.5, None, 0.1]
        slope = compute_decay_slope(gfs)
        assert slope is not None
        assert slope < 0


class TestSwapSensitivity:
    def test_identical_sequences_zero_sensitivity(self):
        gfs = [0.8, 0.6, 0.4]
        score = swap_sensitivity_score(gfs, gfs)
        assert score == 0.0

    def test_different_sequences_positive_sensitivity(self):
        orig = [0.8, 0.6, 0.4]
        swap = [0.3, 0.2, 0.1]
        score = swap_sensitivity_score(orig, swap)
        assert score > 0.3

    def test_nones_skipped(self):
        orig = [0.8, None, 0.4]
        swap = [0.3, None, 0.1]
        score = swap_sensitivity_score(orig, swap)
        assert score > 0.0

    def test_all_none_returns_zero(self):
        score = swap_sensitivity_score([None, None], [None, None])
        assert score == 0.0
