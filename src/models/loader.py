"""
Model loader with automatic environment detection and quantization fallbacks.

Load order:
  1. 4-bit NF4 (bitsandbytes)  — best for VRAM < 20 GB
  2. 8-bit (bitsandbytes)       — fallback if 4-bit fails
  3. fp16 no quantization       — fallback if bnb broken / unavailable
  4. cpu float32                — last resort (very slow, for debugging)
"""

import glob
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Tuple

import torch
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    LlavaForConditionalGeneration,
)

CONFIG_PATH = Path(__file__).parents[2] / "experiments" / "config.json"


# ── Environment detection ─────────────────────────────────────────────────────

def detect_env() -> dict:
    """Return dict describing runtime environment."""
    on_colab    = "google.colab" in sys.modules or os.path.exists("/content")
    on_lightning = os.path.exists("/home/zeus") or "zeus" in os.environ.get("HOME", "")
    on_kaggle   = os.path.exists("/kaggle")

    has_gpu = torch.cuda.is_available()
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if has_gpu else 0.0
    gpu_name = torch.cuda.get_device_name(0) if has_gpu else "none"

    env = {
        "platform":    "colab" if on_colab else "lightning" if on_lightning else "kaggle" if on_kaggle else "local",
        "has_gpu":     has_gpu,
        "gpu_name":    gpu_name,
        "vram_gb":     round(vram_gb, 1),
        "python":      sys.version.split()[0],
        "cuda_version": torch.version.cuda or "none",
    }
    return env


def fix_cuda_path() -> bool:
    """
    Find libcudart.so and inject its directory into LD_LIBRARY_PATH.
    Returns True if a path was found and set.
    """
    patterns = [
        "/usr/local/cuda*/lib64/libcudart.so*",
        "/usr/local/cuda/lib64/libcudart.so*",
        "/usr/lib/x86_64-linux-gnu/libcudart.so*",
        "/usr/local/lib/libcudart.so*",
        "/opt/cuda/lib64/libcudart.so*",
    ]
    for pat in patterns:
        hits = glob.glob(pat)
        if hits:
            lib_dir = os.path.dirname(hits[0])
            current = os.environ.get("LD_LIBRARY_PATH", "")
            if lib_dir not in current:
                os.environ["LD_LIBRARY_PATH"] = f"{lib_dir}:{current}"
            return True
    return False


# ── bitsandbytes probe ────────────────────────────────────────────────────────

def _probe_bnb() -> str:
    """
    Return 'ok', 'broken', or 'missing'.
    'broken' means installed but CUDA setup failed.
    """
    try:
        import bitsandbytes as bnb  # noqa: F401
        from bitsandbytes.cextension import COMPILED_WITH_CUDA  # noqa: F401
        return "ok"
    except ImportError:
        return "missing"
    except Exception:
        return "broken"


def _try_fix_bnb() -> str:
    """Try upgrading bnb and fixing CUDA path, then re-probe."""
    fixed_path = fix_cuda_path()
    status = _probe_bnb()
    if status == "ok":
        return "ok"

    # Upgrade bitsandbytes silently
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-U", "bitsandbytes"],
        check=False, capture_output=True,
    )
    # Force reimport
    for mod in list(sys.modules.keys()):
        if "bitsandbytes" in mod:
            del sys.modules[mod]

    return _probe_bnb()


def _get_bnb_config(bits: int = 4):
    from transformers import BitsAndBytesConfig
    if bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    return BitsAndBytesConfig(load_in_8bit=True)


# ── Config ────────────────────────────────────────────────────────────────────

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


# ── Main loader ───────────────────────────────────────────────────────────────

def load_model_and_processor(
    model_key: str,
    hf_token: str = None,
    device_map: str = "auto",
    force_fp16: bool = False,
) -> Tuple[object, object, dict]:
    """
    Load a VLM with automatic quantization fallback.

    Tries: 4-bit → 8-bit → fp16 → cpu-fp32
    Returns: (model, processor, model_cfg)
    """
    cfg       = load_config()
    model_cfg = cfg["models"][model_key]
    model_id  = model_cfg["model_id"]
    trust     = model_cfg.get("trust_remote_code", False)

    env = detect_env()
    print(f"[env] platform={env['platform']} gpu={env['gpu_name']} vram={env['vram_gb']}GB cuda={env['cuda_version']}")

    processor = AutoProcessor.from_pretrained(
        model_id, trust_remote_code=trust, token=hf_token,
    )

    cls = (LlavaForConditionalGeneration
           if model_cfg["model_class"] == "LlavaForConditionalGeneration"
           else AutoModelForVision2Seq)

    base_kwargs = dict(token=hf_token)
    if trust:
        base_kwargs["trust_remote_code"] = True

    # Determine device_map: use cpu if no GPU
    dm = device_map if env["has_gpu"] else "cpu"

    # ── Attempt order ─────────────────────────────────────────────────────────
    attempts = []

    if env["has_gpu"] and not force_fp16:
        bnb_status = _try_fix_bnb()
        print(f"[bnb] status={bnb_status}")

        if bnb_status == "ok":
            attempts.append(("4-bit",  dict(quantization_config=_get_bnb_config(4), device_map=dm, torch_dtype=torch.float16)))
            attempts.append(("8-bit",  dict(quantization_config=_get_bnb_config(8), device_map=dm, torch_dtype=torch.float16)))

        attempts.append(("fp16",   dict(device_map=dm, torch_dtype=torch.float16)))
    else:
        if env["has_gpu"]:
            attempts.append(("fp16", dict(device_map=dm, torch_dtype=torch.float16)))
        else:
            attempts.append(("cpu-fp32", dict(device_map="cpu", torch_dtype=torch.float32)))

    model = None
    for mode, extra_kwargs in attempts:
        kwargs = {**base_kwargs, **extra_kwargs}
        print(f"[loader] Trying {model_key} in {mode} ...")
        try:
            model = cls.from_pretrained(model_id, **kwargs)
            print(f"[loader] Loaded in {mode}.")
            break
        except (RuntimeError, ImportError, ValueError) as e:
            msg = str(e).lower()
            if any(k in msg for k in ("bitsandbytes", "bnb", "cuda setup", "quantization", "bits_and_bytes")):
                print(f"[loader] {mode} failed: {e.__class__.__name__}. Trying next ...")
                continue
            raise  # unexpected error — propagate

    if model is None:
        raise RuntimeError(f"All load attempts failed for {model_key}.")

    model.eval()
    if env["has_gpu"]:
        print(f"[loader] VRAM used: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    return model, processor, model_cfg


def unload_model(model) -> None:
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[loader] Model unloaded.")


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_prompt(model_key: str, processor, question: str, cot: bool = True) -> str:
    if cot:
        instruction = (
            "Look at the image carefully. Answer the following question step by step, "
            "explicitly describing what you see in the image at each step.\n"
            f"Question: {question}\n"
            "Step 1:"
        )
    else:
        instruction = f"Look at the image. Answer directly.\nQuestion: {question}\nAnswer:"

    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]}]

    if model_key == "llava":
        return processor.apply_chat_template(messages, add_generation_prompt=True)
    return processor.apply_chat_template(messages, add_generation_prompt=True)


# ── Step parser ───────────────────────────────────────────────────────────────

def parse_steps(text: str):
    import re
    matches = re.findall(r"Step\s*\d+[:\.\)]\s*(.+?)(?=Step\s*\d+[:\.\)]|\Z)", text, re.DOTALL)
    if len(matches) >= 2:
        return [m.strip() for m in matches]
    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    return parts if parts else [text.strip()]
