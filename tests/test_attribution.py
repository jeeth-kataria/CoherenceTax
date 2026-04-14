"""Tests for attribution modules (shape + output range checks, no GPU required)."""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from src.utils.stats import bootstrap_ci, test_decay_significance

torch = pytest.importorskip("torch", reason="torch not installed — skip on local, run on Colab")


def _load_rollout():
    from src.attribution.rollout import find_image_token_range
    return find_image_token_range


class TestFindImageTokenRange:
    def test_finds_image_token(self):
        find_image_token_range = _load_rollout()
        ids = torch.tensor([[1, 2, -200, 3, 4]])
        start, end = find_image_token_range(ids, image_token_id=-200, n_patches=576)
        assert start == 2
        assert end == 2 + 576

    def test_fallback_when_not_found(self):
        find_image_token_range = _load_rollout()
        ids = torch.tensor([[1, 2, 3, 4]])
        start, end = find_image_token_range(ids, image_token_id=-200, n_patches=64)
        assert start == 0
        assert end == 64


class TestBootstrapCI:
    def test_mean_in_interval(self):
        rng = np.random.default_rng(0)
        data = rng.normal(loc=0.6, scale=0.1, size=100)
        mean, lo, hi = bootstrap_ci(data, n_bootstrap=1000)
        assert lo <= mean <= hi

    def test_interval_contains_true_mean(self):
        rng = np.random.default_rng(42)
        data = rng.normal(loc=0.5, scale=0.05, size=200)
        _, lo, hi = bootstrap_ci(data, n_bootstrap=5000)
        assert lo <= 0.5 <= hi

    def test_wider_with_more_variance(self):
        rng = np.random.default_rng(1)
        narrow = rng.normal(0.5, 0.01, 100)
        wide = rng.normal(0.5, 0.3, 100)
        _, lo_n, hi_n = bootstrap_ci(narrow, n_bootstrap=1000)
        _, lo_w, hi_w = bootstrap_ci(wide, n_bootstrap=1000)
        assert (hi_w - lo_w) > (hi_n - lo_n)


class TestDecaySignificance:
    def test_clear_decay_significant(self):
        # Sequences that clearly decrease each time
        seqs = [[0.9 - 0.15 * k for k in range(5)] for _ in range(30)]
        result = test_decay_significance(seqs)
        assert result["significant"] is True
        assert result["mean_slope"] < 0

    def test_flat_not_significant(self):
        seqs = [[0.5] * 5 for _ in range(30)]
        result = test_decay_significance(seqs)
        # Slope should be near 0
        assert result["mean_slope"] is not None
        assert abs(result["mean_slope"]) < 0.01

    def test_too_few_samples(self):
        seqs = [[0.9, 0.5, 0.1]]  # only 1 sample
        result = test_decay_significance(seqs)
        assert result["significant"] is False

    def test_handles_nones(self):
        seqs = [[0.9, None, 0.5, None, 0.1] for _ in range(20)]
        result = test_decay_significance(seqs)
        assert result["n_samples"] == 20
        assert result["mean_slope"] is not None
