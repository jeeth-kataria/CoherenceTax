"""Model loader: 4-bit quantized VLMs for Colab A100."""

import json
import os
from pathlib import Path
from typing import Tuple

import torch
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaForConditionalGeneration,
)

CONFIG_PATH = Path(__file__).parents[2] / "experiments" / "config.json"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


def get_bnb_config(cfg: dict) -> BitsAndBytesConfig:
    q = cfg["quantization"]
    return BitsAndBytesConfig(
        load_in_4bit=q["load_in_4bit"],
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"],
    )


def load_model_and_processor(
    model_key: str,
    hf_token: str = None,
    device_map: str = "auto",
) -> Tuple[object, object, dict]:
    """
    Load a VLM in 4-bit quantization.

    Returns:
        model, processor, model_cfg
    """
    cfg = load_config()
    model_cfg = cfg["models"][model_key]
    bnb = get_bnb_config(cfg)

    model_id = model_cfg["model_id"]
    trust = model_cfg.get("trust_remote_code", False)

    print(f"Loading {model_key} ({model_id}) in 4-bit ...")

    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=trust,
        token=hf_token,
    )

    if model_cfg["model_class"] == "LlavaForConditionalGeneration":
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=bnb,
            device_map=device_map,
            torch_dtype=torch.float16,
            token=hf_token,
        )
    else:
        model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            quantization_config=bnb,
            device_map=device_map,
            torch_dtype=torch.float16,
            trust_remote_code=trust,
            token=hf_token,
        )

    model.eval()
    print(f"Loaded {model_key}. GPU mem: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
    return model, processor, model_cfg


def unload_model(model) -> None:
    """Free GPU memory between model loads."""
    del model
    torch.cuda.empty_cache()
    print("Model unloaded, cache cleared.")


def build_prompt(
    model_key: str,
    processor,
    question: str,
    cot: bool = True,
) -> str:
    """
    Build the correct chat-template prompt for each model.
    cot=True forces step-by-step output.
    """
    if cot:
        instruction = (
            "Look at the image carefully. Answer the following question step by step, "
            "explicitly describing what you see in the image at each step.\n"
            f"Question: {question}\n"
            "Step 1:"
        )
    else:
        instruction = f"Look at the image. Answer directly.\nQuestion: {question}\nAnswer:"

    if model_key == "llava":
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": instruction},
                ],
            }
        ]
        return processor.apply_chat_template(conversation, add_generation_prompt=True)
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": instruction},
                ],
            }
        ]
        return processor.apply_chat_template(messages, add_generation_prompt=True)


def parse_steps(text: str):
    """Extract numbered reasoning steps from model output."""
    import re

    matches = re.findall(
        r"Step\s*\d+[:\.\)]\s*(.+?)(?=Step\s*\d+[:\.\)]|\Z)",
        text,
        re.DOTALL,
    )
    if len(matches) >= 2:
        return [m.strip() for m in matches]
    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    return parts if parts else [text.strip()]
