"""Attention Rollout — no gradients needed, works on 4-bit models."""

import cv2
import numpy as np
import torch


def attention_rollout(
    model,
    inputs: dict,
    image_token_start: int,
    image_token_end: int,
    image_size: int = 336,
) -> np.ndarray:
    """
    Compute attention rollout heatmap for image tokens.
    Returns normalized heatmap [image_size, image_size] in [0,1].

    image_token_start/end: indices in the token sequence where image tokens live.
    """
    with torch.no_grad():
        try:
            outputs = model(**inputs, output_attentions=True)
        except TypeError:
            # Some models don't support output_attentions with quantization
            print("Rollout: output_attentions not supported. Returning uniform heatmap.")
            return np.ones((image_size, image_size), dtype=np.float32) * 0.5

    attn_list = outputs.attentions
    if attn_list is None or len(attn_list) == 0:
        print("Rollout: no attention weights returned.")
        return np.ones((image_size, image_size), dtype=np.float32) * 0.5

    seq_len = attn_list[0].shape[-1]
    device = attn_list[0].device
    rollout = torch.eye(seq_len, device=device)

    for attn in attn_list:
        a = attn.squeeze(0).mean(0).float()  # [seq, seq]
        # Add residual (Abnar & Zuidema 2020)
        a = 0.5 * a + 0.5 * torch.eye(seq_len, device=device)
        a = a / (a.sum(dim=-1, keepdim=True) + 1e-8)
        rollout = a @ rollout

    # Attention from last text token to image tokens
    img_start = max(0, image_token_start)
    img_end = min(seq_len, image_token_end)
    image_attn = rollout[-1, img_start:img_end].cpu().float()

    n = image_attn.shape[0]
    if n == 0:
        return np.ones((image_size, image_size), dtype=np.float32) * 0.5

    g = int(round(n ** 0.5))
    if g * g != n:
        for g in range(int(n ** 0.5), 0, -1):
            if n % g == 0:
                break
        h = n // g
    else:
        h = g

    heatmap = image_attn.numpy().reshape(h, g)
    heatmap = cv2.resize(heatmap, (image_size, image_size))
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap.astype(np.float32)


def find_image_token_range(input_ids: torch.Tensor, image_token_id: int, n_patches: int):
    """
    Locate where image tokens start and end in the input_ids sequence.
    For LLaVA: IMAGE_TOKEN_INDEX = -200, expands to n_patches = 576.
    Returns (start, end) indices in the expanded sequence.
    """
    ids = input_ids[0].cpu().tolist()
    if image_token_id in ids:
        pos = ids.index(image_token_id)
        return pos, pos + n_patches
    # Fallback: assume image tokens at positions 0..n_patches
    return 0, n_patches
