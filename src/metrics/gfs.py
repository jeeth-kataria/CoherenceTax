"""
Grounding Faithfulness Score (GFS).

GFS_k = 1 - normalized spatial entropy of attribution heatmap at step k.
High GFS = concentrated attention = more visually grounded.
Low GFS = diffuse attention = less grounded (hallucinated reasoning).
"""

import re
from typing import List, Optional

import numpy as np
import spacy

_NLP = None


def _get_nlp():
    global _NLP
    if _NLP is None:
        try:
            _NLP = spacy.load("en_core_web_sm")
        except OSError:
            from spacy.cli import download
            download("en_core_web_sm")
            _NLP = spacy.load("en_core_web_sm")
    return _NLP


_SPATIAL_WORDS = {
    "left", "right", "top", "bottom", "center", "corner",
    "background", "foreground", "object", "image", "picture",
    "scene", "area", "region", "part", "side", "edge",
}


def has_visual_reference(step_text: str) -> bool:
    """Return True if step explicitly references visual content."""
    if not step_text:
        return False

    # Criterion 1: spatial words anywhere in the text
    low = step_text.lower()
    if any(f" {w} " in f" {low} " for w in _SPATIAL_WORDS):
        return True

    # Criterion 2: noun chunks containing a spatial word
    nlp = _get_nlp()
    doc = nlp(step_text)
    for chunk in doc.noun_chunks:
        if any(w.lower_ in _SPATIAL_WORDS for w in chunk):
            return True

    # Criterion 3: explicit visual-action phrases
    visual_phrases = [
        r"\bI see\b", r"\bthe image shows\b", r"\bthe image depicts\b",
        r"\bdepicts\b", r"\bvisible\b", r"\bshown in\b", r"\bdisplays\b",
        r"\bin the (image|picture|photo|scene)\b",
        r"\b(looking|look) at\b",
    ]
    return any(re.search(p, low) for p in visual_phrases)


def compute_gfs(heatmap: np.ndarray, step_text: str) -> Optional[float]:
    """
    Compute GFS for a single step.
    Returns None if step has no visual reference.
    Returns float in [0, 1] otherwise.
    """
    if not has_visual_reference(step_text):
        return None

    heatmap = heatmap.astype(np.float64)
    total = heatmap.sum()
    if total <= 0:
        return 0.0

    p = heatmap / total
    entropy = -np.sum(p * np.log(p + 1e-10))
    max_entropy = np.log(heatmap.size)
    concentration = 1.0 - (entropy / max_entropy)
    return float(np.clip(concentration, 0.0, 1.0))


def compute_gfs_sequence(
    heatmaps: List[np.ndarray],
    steps: List[str],
) -> List[Optional[float]]:
    """
    Compute GFS for each step in a CoT sequence.
    len(heatmaps) must equal len(steps).
    """
    assert len(heatmaps) == len(steps), "heatmaps and steps must be same length"
    return [compute_gfs(h, s) for h, s in zip(heatmaps, steps)]


def compute_decay_slope(gfs_sequence: List[Optional[float]]) -> Optional[float]:
    """
    Fit linear slope to GFS vs step index.
    Returns slope (negative = decaying grounding).
    Returns None if fewer than 3 valid points.
    """
    valid = [(k, s) for k, s in enumerate(gfs_sequence) if s is not None]
    if len(valid) < 3:
        return None
    x = np.array([v[0] for v in valid], dtype=float)
    y = np.array([v[1] for v in valid], dtype=float)
    return float(np.polyfit(x, y, 1)[0])


def swap_sensitivity_score(
    gfs_original: List[Optional[float]],
    gfs_swapped: List[Optional[float]],
) -> float:
    """
    Swap sensitivity = mean |GFS_original_k - GFS_swapped_k| over valid steps.
    High sensitivity = model was actually using the image.
    Low sensitivity = model ignored the image (same heatmap after swap).
    """
    diffs = []
    for o, s in zip(gfs_original, gfs_swapped):
        if o is not None and s is not None:
            diffs.append(abs(o - s))
    return float(np.mean(diffs)) if diffs else 0.0


def summarize_result(result: dict) -> dict:
    """
    Compute aggregate stats from a single sample result.
    result must have: gfs_per_step, gfs_per_step_swapped (optional).
    """
    gfs = result.get("gfs_per_step", [])
    valid = [s for s in gfs if s is not None]

    summary = {
        "gfs_mean": float(np.mean(valid)) if valid else None,
        "gfs_std": float(np.std(valid)) if len(valid) > 1 else None,
        "gfs_decay_slope": compute_decay_slope(gfs),
        "n_steps": len(gfs),
        "n_valid_steps": len(valid),
        "final_answer_correct": result.get("final_answer_correct"),
    }

    if "gfs_per_step_swapped" in result:
        summary["swap_sensitivity_score"] = swap_sensitivity_score(
            gfs, result["gfs_per_step_swapped"]
        )
    else:
        summary["swap_sensitivity_score"] = None

    return summary
