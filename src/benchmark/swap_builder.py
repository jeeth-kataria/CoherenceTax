"""
Construct swap triplets: (original_sample, swap_image_sample, metadata).

Criteria per SKILLS.md Skill 6:
1. Same semantic subject/category
2. Visually different (pixel MSE > threshold after resize to 224x224)
3. Different correct answer
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def load_image(example, dataset_name: str) -> Optional[Image.Image]:
    """Extract PIL image from HF dataset example."""
    img = example.get("image")
    if img is None:
        return None
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    return None


def pixel_mse(img_a: Image.Image, img_b: Image.Image, size: Tuple[int, int] = (224, 224)) -> float:
    a = np.array(img_a.resize(size)).astype(float) / 255.0
    b = np.array(img_b.resize(size)).astype(float) / 255.0
    if a.shape != b.shape:
        return 1.0
    return float(np.mean((a - b) ** 2))


def is_valid_swap(
    orig: Dict,
    cand: Dict,
    min_pixel_diff: float = 0.25,
) -> bool:
    # Criterion 1: same subject
    if orig.get("subject") != cand.get("subject"):
        return False

    # Criterion 2: visually different
    img_a = orig.get("_image")
    img_b = cand.get("_image")
    if img_a is None or img_b is None:
        return False
    if pixel_mse(img_a, img_b) < min_pixel_diff ** 2:
        return False

    # Criterion 3: different answer
    if orig.get("answer") == cand.get("answer"):
        return False

    return True


def build_swap_benchmark(
    scienceqa_meta: List[Dict],
    gqa_meta: List[Dict],
    scienceqa_ds,
    gqa_ds,
    n_triplets: int = 200,
    min_pixel_diff: float = 0.25,
    seed: int = 42,
    scienceqa_fraction: float = 0.7,
) -> List[Dict]:
    """
    Build n_triplets swap pairs.
    Returns list of dicts: {original, swap, metadata}.
    """
    rng = random.Random(seed)

    n_sqa = int(n_triplets * scienceqa_fraction)
    n_gqa = n_triplets - n_sqa

    triplets = []

    # --- ScienceQA ---
    sqa_by_subject: Dict[str, List] = {}
    for rec in scienceqa_meta:
        s = rec.get("subject", "")
        sqa_by_subject.setdefault(s, []).append(rec)

    sqa_candidates = [r for r in scienceqa_meta if sqa_by_subject.get(r.get("subject"), []) and len(sqa_by_subject.get(r.get("subject"), [])) >= 2]
    rng.shuffle(sqa_candidates)

    print(f"Building {n_sqa} ScienceQA swap triplets ...")
    attempted = 0
    for orig_rec in tqdm(sqa_candidates):
        if len([t for t in triplets if t["source"] == "scienceqa"]) >= n_sqa:
            break
        attempted += 1

        orig_img = load_image(scienceqa_ds[orig_rec["hf_index"]], "scienceqa")
        if orig_img is None:
            continue
        orig_rec["_image"] = orig_img

        subject = orig_rec.get("subject", "")
        pool = [r for r in sqa_by_subject.get(subject, []) if r["id"] != orig_rec["id"]]
        rng.shuffle(pool)

        found = False
        for cand_rec in pool[:30]:  # try up to 30 candidates
            cand_img = load_image(scienceqa_ds[cand_rec["hf_index"]], "scienceqa")
            if cand_img is None:
                continue
            cand_rec["_image"] = cand_img

            if is_valid_swap(orig_rec, cand_rec, min_pixel_diff):
                triplets.append(
                    {
                        "triplet_id": f"sqa_triplet_{len(triplets):04d}",
                        "source": "scienceqa",
                        "original": {
                            "id": orig_rec["id"],
                            "hf_index": orig_rec["hf_index"],
                            "question": orig_rec["question"],
                            "answer": orig_rec["answer"],
                            "subject": orig_rec["subject"],
                        },
                        "swap": {
                            "id": cand_rec["id"],
                            "hf_index": cand_rec["hf_index"],
                            "question": orig_rec["question"],  # same question, different image
                            "answer": cand_rec["answer"],
                            "subject": cand_rec["subject"],
                        },
                        "pixel_mse": pixel_mse(orig_img, cand_img),
                    }
                )
                found = True
                break

    print(f"ScienceQA: {len([t for t in triplets if t['source'] == 'scienceqa'])} triplets from {attempted} attempts")

    # --- GQA (if available) ---
    if gqa_meta and n_gqa > 0:
        gqa_candidates = gqa_meta.copy()
        rng.shuffle(gqa_candidates)
        print(f"Building {n_gqa} GQA swap triplets ...")

        for orig_rec in tqdm(gqa_candidates):
            if len([t for t in triplets if t["source"] == "gqa"]) >= n_gqa:
                break

            orig_img = load_image(gqa_ds[orig_rec["hf_index"]], "gqa")
            if orig_img is None:
                continue
            orig_rec["_image"] = orig_img

            pool = [r for r in gqa_candidates if r["id"] != orig_rec["id"]]
            rng.shuffle(pool)

            for cand_rec in pool[:30]:
                cand_img = load_image(gqa_ds[cand_rec["hf_index"]], "gqa")
                if cand_img is None:
                    continue
                cand_rec["_image"] = cand_img
                cand_rec["subject"] = "visual_reasoning"
                orig_rec["subject"] = "visual_reasoning"

                if is_valid_swap(orig_rec, cand_rec, min_pixel_diff):
                    triplets.append(
                        {
                            "triplet_id": f"gqa_triplet_{len(triplets):04d}",
                            "source": "gqa",
                            "original": {
                                "id": orig_rec["id"],
                                "hf_index": orig_rec["hf_index"],
                                "question": orig_rec["question"],
                                "answer": orig_rec["answer"],
                                "subject": "visual_reasoning",
                            },
                            "swap": {
                                "id": cand_rec["id"],
                                "hf_index": cand_rec["hf_index"],
                                "question": orig_rec["question"],
                                "answer": cand_rec["answer"],
                                "subject": "visual_reasoning",
                            },
                            "pixel_mse": pixel_mse(orig_img, cand_img),
                        }
                    )
                    break

        print(f"GQA: {len([t for t in triplets if t['source'] == 'gqa'])} triplets")

    # Check success rate
    success_rate = len(triplets) / n_triplets if n_triplets > 0 else 0
    print(f"Total triplets: {len(triplets)} / {n_triplets} = {success_rate:.1%}")
    if success_rate < 0.6:
        print(f"WARNING: < 60% success rate. Consider lowering min_pixel_diff to 0.15.")

    return triplets


def save_benchmark(triplets: List[Dict], output_path: str) -> None:
    # Strip _image keys before saving (PIL images not JSON-serializable)
    clean = []
    for t in triplets:
        tc = {k: v for k, v in t.items() if k != "_image"}
        tc["original"] = {k: v for k, v in tc.get("original", {}).items() if k != "_image"}
        tc["swap"] = {k: v for k, v in tc.get("swap", {}).items() if k != "_image"}
        clean.append(tc)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"Saved {len(clean)} triplets to {output_path}")


def load_benchmark(path: str) -> List[Dict]:
    with open(path) as f:
        return json.load(f)
