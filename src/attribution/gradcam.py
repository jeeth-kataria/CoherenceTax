"""Grad-CAM on last vision encoder layer of VLM."""

import cv2
import numpy as np
import torch


class GradCAM:
    def __init__(self, model, layer_name: str, image_size: int = 336):
        self.model = model
        self.image_size = image_size
        self._activations = {}
        self._gradients = {}
        self._handle_f = None
        self._handle_b = None
        self._layer_name = layer_name
        self._target_layer = self._find_layer(layer_name)

    def _find_layer(self, name: str):
        for n, m in self.model.named_modules():
            if n == name:
                return m
        # Fallback: partial match on last vision encoder layer
        candidates = [
            (n, m) for n, m in self.model.named_modules()
            if "encoder.layers" in n or "vision_model.layers" in n
        ]
        if candidates:
            n, m = candidates[-1]
            print(f"GradCAM: exact layer not found, using fallback: {n}")
            return m
        raise ValueError(f"Layer '{name}' not found in model.")

    def _register_hooks(self):
        def fwd(module, inp, out):
            # out may be a tuple (hidden_state, ...) or just a tensor
            self._activations["feat"] = (out[0] if isinstance(out, tuple) else out).detach()

        def bwd(module, grad_in, grad_out):
            g = grad_out[0] if isinstance(grad_out, tuple) else grad_out
            self._gradients["feat"] = g.detach()

        self._handle_f = self._target_layer.register_forward_hook(fwd)
        self._handle_b = self._target_layer.register_full_backward_hook(bwd)

    def _remove_hooks(self):
        if self._handle_f:
            self._handle_f.remove()
        if self._handle_b:
            self._handle_b.remove()

    def compute(self, inputs: dict) -> np.ndarray:
        """
        Run forward+backward and return normalized GradCAM heatmap [H, W] in [0,1].
        inputs: dict from processor (pixel_values, input_ids, attention_mask, etc.)
        """
        self._activations.clear()
        self._gradients.clear()
        self._register_hooks()

        try:
            self.model.zero_grad()
            with torch.enable_grad():
                outputs = self.model(**inputs)
                # Force float32 for grad flow through quantized layers
                score = outputs.logits[0, -1, :].float().sum()
                score.backward()
        except RuntimeError as e:
            print(f"GradCAM backward failed: {e}. Returning uniform heatmap.")
            self._remove_hooks()
            return np.ones((self.image_size, self.image_size), dtype=np.float32) * 0.5
        finally:
            self._remove_hooks()

        if "feat" not in self._activations or "feat" not in self._gradients:
            print("GradCAM: hooks did not fire. Returning uniform heatmap.")
            return np.ones((self.image_size, self.image_size), dtype=np.float32) * 0.5

        act = self._activations["feat"]  # [B, patches, C] or [B, C, H, W]
        grad = self._gradients["feat"]

        # Handle both [B, patches, C] and [B, C, H, W] layouts
        if act.dim() == 3:
            # [B, patches, C] — typical for ViT
            weights = grad.mean(dim=(0, 1))  # [C]
            cam = (weights * act[0]).sum(dim=-1)  # [patches]
        else:
            # [B, C, H, W] — CNN-style
            weights = grad.mean(dim=(0, 2, 3))  # [C]
            cam = (weights[:, None, None] * act[0]).sum(dim=0)  # [H, W]
            cam = cam.flatten()

        cam = torch.relu(cam)
        n = cam.shape[0]
        g = int(round(n ** 0.5))
        if g * g != n:
            # Non-square: try to find factors
            for g in range(int(n ** 0.5), 0, -1):
                if n % g == 0:
                    break
            h = n // g
        else:
            h = g

        cam_2d = cam.reshape(h, g).cpu().float().numpy()
        cam_resized = cv2.resize(cam_2d, (self.image_size, self.image_size))
        cam_norm = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min() + 1e-8)
        return cam_norm.astype(np.float32)
