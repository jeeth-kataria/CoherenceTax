"""Patch occlusion sensitivity map — brute-force but model-agnostic."""

import numpy as np
import torch
from PIL import Image


def occlusion_sensitivity(
    model,
    processor,
    image: Image.Image,
    prompt: str,
    patch_size: int = 24,
    stride: int = 12,
    image_size: int = 336,
    device: str = "cuda",
) -> np.ndarray:
    """
    Compute occlusion sensitivity map.
    Each patch is grayed out; score = log-prob drop vs baseline.
    Returns normalized heatmap [image_size, image_size] in [0,1].
    """
    image_resized = image.resize((image_size, image_size))
    img_arr = np.array(image_resized, dtype=np.float32)

    def get_score(img_pil):
        inputs = processor(
            text=prompt,
            images=img_pil,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out = model(**inputs)
            # Use sum of last-token logits as proxy score
            return float(out.logits[0, -1, :].float().sum().cpu())

    baseline_score = get_score(image_resized)
    gray_val = img_arr.mean()

    H, W = image_size, image_size
    sensitivity = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)

    rows = range(0, H - patch_size + 1, stride)
    cols = range(0, W - patch_size + 1, stride)
    total = len(rows) * len(cols)

    print(f"Occlusion: {total} patches (size={patch_size}, stride={stride})")

    done = 0
    for r in rows:
        for c in cols:
            occluded = img_arr.copy()
            occluded[r : r + patch_size, c : c + patch_size] = gray_val
            occ_img = Image.fromarray(occluded.astype(np.uint8))
            score = get_score(occ_img)
            drop = baseline_score - score
            sensitivity[r : r + patch_size, c : c + patch_size] += drop
            count[r : r + patch_size, c : c + patch_size] += 1
            done += 1

    count = np.maximum(count, 1)
    sensitivity /= count
    sensitivity = np.maximum(sensitivity, 0)
    denom = sensitivity.max() - sensitivity.min() + 1e-8
    sensitivity = (sensitivity - sensitivity.min()) / denom
    return sensitivity.astype(np.float32)
