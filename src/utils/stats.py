"""Statistical utilities — Skill 9 from SKILLS.md."""

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from scipy import stats


def bootstrap_ci(
    data: np.ndarray,
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap confidence interval for mean. Returns (mean, lo, hi)."""
    rng = np.random.default_rng(seed)
    boot_means = [
        rng.choice(data, len(data), replace=True).mean()
        for _ in range(n_bootstrap)
    ]
    lo = float(np.percentile(boot_means, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boot_means, (1 + ci) / 2 * 100))
    return float(np.mean(data)), lo, hi


def test_decay_significance(
    gfs_per_step_list: List[List[Optional[float]]],
    min_steps: int = 3,
) -> dict:
    """
    Test whether GFS decay slope is significantly negative.
    One-sample t-test on per-sample slopes vs 0.
    """
    slopes = []
    for gfs_seq in gfs_per_step_list:
        valid = [(k, s) for k, s in enumerate(gfs_seq) if s is not None]
        if len(valid) < min_steps:
            continue
        x = np.array([v[0] for v in valid], dtype=float)
        y = np.array([v[1] for v in valid], dtype=float)
        slope = float(np.polyfit(x, y, 1)[0])
        slopes.append(slope)

    if len(slopes) < 5:
        return {
            "mean_slope": None,
            "ci_95": None,
            "t_stat": None,
            "p_value": None,
            "significant": False,
            "n_samples": len(slopes),
        }

    arr = np.array(slopes)
    t_stat, p_value = stats.ttest_1samp(arr, 0)
    mean, lo, hi = bootstrap_ci(arr)
    return {
        "mean_slope": mean,
        "ci_95": (lo, hi),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
        "n_samples": len(slopes),
    }


def compare_models(
    results_a: List[dict],
    results_b: List[dict],
    metric: str = "gfs_mean",
) -> dict:
    """
    Paired comparison between two model result sets (matched by triplet_id).
    Returns Mann-Whitney U + bootstrap CI on difference.
    """
    a_map = {r["triplet_id"]: r.get(metric) for r in results_a}
    b_map = {r["triplet_id"]: r.get(metric) for r in results_b}
    common = [k for k in a_map if k in b_map and a_map[k] is not None and b_map[k] is not None]

    if len(common) < 5:
        return {"n": len(common), "p_value": None, "significant": False}

    vals_a = np.array([a_map[k] for k in common])
    vals_b = np.array([b_map[k] for k in common])

    u_stat, p = stats.mannwhitneyu(vals_a, vals_b, alternative="two-sided")
    diff = vals_a - vals_b
    mean_diff, lo, hi = bootstrap_ci(diff)

    return {
        "n": len(common),
        "mean_a": float(np.mean(vals_a)),
        "mean_b": float(np.mean(vals_b)),
        "mean_diff": mean_diff,
        "diff_ci_95": (lo, hi),
        "u_stat": float(u_stat),
        "p_value": float(p),
        "significant": bool(p < 0.05),
    }


def spearman_gfs_steps(
    gfs_per_step_list: List[List[Optional[float]]],
    max_steps: int = 6,
) -> dict:
    """
    Spearman correlation between step index and GFS across all valid (sample, step) pairs.
    Tests whether GFS is monotonically related to step depth.
    """
    xs, ys = [], []
    for gfs_seq in gfs_per_step_list:
        for k, g in enumerate(gfs_seq[:max_steps]):
            if g is not None:
                xs.append(k)
                ys.append(g)

    if len(xs) < 10:
        return {"rho": None, "p_value": None, "n": len(xs)}

    rho, p = stats.spearmanr(xs, ys)
    return {"rho": float(rho), "p_value": float(p), "n": len(xs)}


# ── Checkpoint helpers (Skill 7) ──────────────────────────────────────────────

def save_checkpoint(results: list, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, default=str)


def load_checkpoint(path: str) -> list:
    p = Path(path)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return []
