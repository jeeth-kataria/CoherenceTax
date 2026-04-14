"""
Ablation experiments:
  A1: CoT vs direct-answer prompting — does forcing CoT improve GFS?
  A2: Attribution method comparison — GradCAM vs Rollout vs Occlusion
  A3: Question category analysis — GFS by ScienceQA subject

Usage:
    python scripts/run_ablations.py --model llava --output results/ablations/ --ablations A1 A2 A3
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.attribution.gradcam import GradCAM
from src.attribution.occlusion import occlusion_sensitivity
from src.attribution.rollout import attention_rollout, find_image_token_range
from src.metrics.gfs import compute_gfs_sequence, compute_decay_slope
from src.models.loader import build_prompt, load_config, load_model_and_processor, parse_steps
from src.utils.stats import save_checkpoint, load_checkpoint, bootstrap_ci


def get_image(ds, idx):
    img = ds[idx].get("image")
    if img is None:
        return None
    from PIL import Image
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    return None


def compute_heatmap(method, model, processor, model_cfg, model_key, image, step, cfg, device):
    img_size = cfg["attribution"]["image_size"]
    prompt = build_prompt(model_key, processor, step, cot=False)
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

    n_patches = model_cfg.get("n_image_patches", 576)
    img_token_id = model_cfg.get("image_token_id", -200)
    img_start, img_end = find_image_token_range(inputs["input_ids"], img_token_id, n_patches)
    layer_name = model_cfg.get("vision_encoder_layer", "")

    if method == "gradcam" and layer_name:
        try:
            gc = GradCAM(model, layer_name, image_size=img_size)
            return gc.compute(inputs)
        except Exception:
            pass
    if method == "rollout":
        return attention_rollout(model, inputs, img_start, img_end, img_size)
    if method == "occlusion":
        return occlusion_sensitivity(
            model, processor, image, prompt,
            patch_size=cfg["attribution"]["occlusion_patch_size"],
            stride=cfg["attribution"]["occlusion_stride"],
            image_size=img_size,
            device=device,
        )
    # Fallback
    return attention_rollout(model, inputs, img_start, img_end, img_size)


def ablation_a1_cot_vs_direct(model, processor, model_cfg, model_key, samples, scienceqa_ds, cfg, device, out_dir):
    """CoT vs direct-answer: compare GFS distribution."""
    results = []
    for triplet in tqdm(samples, desc="A1"):
        orig = triplet["original"]
        img = get_image(scienceqa_ds, orig["hf_index"])
        if img is None:
            continue
        question = orig["question"]
        img_size = cfg["attribution"]["image_size"]
        n_patches = model_cfg.get("n_image_patches", 576)
        img_token_id = model_cfg.get("image_token_id", -200)

        row = {"triplet_id": triplet["triplet_id"], "question": question}

        for cot_mode in [True, False]:
            prompt = build_prompt(model_key, processor, question, cot=cot_mode)
            inputs = processor(text=prompt, images=img, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=cfg["generation"]["max_new_tokens"],
                    do_sample=False,
                )
            gen_ids = out[0][inputs["input_ids"].shape[1]:]
            text = processor.decode(gen_ids, skip_special_tokens=True)
            steps = parse_steps(text) if cot_mode else [text.strip()]

            img_start, img_end = find_image_token_range(inputs["input_ids"], img_token_id, n_patches)
            layer_name = model_cfg.get("vision_encoder_layer", "")

            heatmaps = []
            for s in steps[: cfg["gfs"]["max_steps"]]:
                step_inputs = processor(
                    text=build_prompt(model_key, processor, s, cot=False),
                    images=img, return_tensors="pt"
                ).to(device)
                try:
                    gc = GradCAM(model, layer_name, image_size=img_size)
                    hm = gc.compute(step_inputs)
                except Exception:
                    hm = attention_rollout(model, step_inputs, img_start, img_end, img_size)
                heatmaps.append(hm)

            gfs_seq = compute_gfs_sequence(heatmaps, steps[: cfg["gfs"]["max_steps"]])
            label = "cot" if cot_mode else "direct"
            valid = [g for g in gfs_seq if g is not None]
            row[f"gfs_mean_{label}"] = float(np.mean(valid)) if valid else None
            row[f"gfs_decay_{label}"] = compute_decay_slope(gfs_seq)
            row[f"n_steps_{label}"] = len(steps)

        results.append(row)

    out_path = out_dir / "a1_cot_vs_direct.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"A1 saved to {out_path}")
    return results


def ablation_a2_method_comparison(model, processor, model_cfg, model_key, samples, scienceqa_ds, cfg, device, out_dir):
    """Compare GradCAM vs Rollout vs Occlusion GFS on same samples."""
    results = []
    methods = ["gradcam", "rollout"]  # occlusion is slow; add if compute allows

    for triplet in tqdm(samples[:30], desc="A2"):  # 30 samples is enough for method comparison
        orig = triplet["original"]
        img = get_image(scienceqa_ds, orig["hf_index"])
        if img is None:
            continue
        question = orig["question"]

        prompt = build_prompt(model_key, processor, question, cot=True)
        inputs_gen = processor(text=prompt, images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs_gen, max_new_tokens=cfg["generation"]["max_new_tokens"], do_sample=False)
        gen_ids = out[0][inputs_gen["input_ids"].shape[1]:]
        text = processor.decode(gen_ids, skip_special_tokens=True)
        steps = parse_steps(text)[: cfg["gfs"]["max_steps"]]

        row = {"triplet_id": triplet["triplet_id"]}
        for method in methods:
            heatmaps = [
                compute_heatmap(method, model, processor, model_cfg, model_key, img, s, cfg, device)
                for s in steps
            ]
            gfs_seq = compute_gfs_sequence(heatmaps, steps)
            valid = [g for g in gfs_seq if g is not None]
            row[f"gfs_mean_{method}"] = float(np.mean(valid)) if valid else None
            row[f"gfs_decay_{method}"] = compute_decay_slope(gfs_seq)
        results.append(row)

    out_path = out_dir / "a2_method_comparison.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"A2 saved to {out_path}")
    return results


def ablation_a3_by_subject(full_results_dir: Path, out_dir: Path):
    """Aggregate GFS by ScienceQA subject from full results."""
    results = []
    for f in full_results_dir.glob("*.json"):
        if "checkpoint" in f.name or "summary" in f.name:
            continue
        with open(f) as fh:
            results.append(json.load(fh))

    if not results:
        print("A3: no full results found.")
        return

    by_subject = {}
    for r in results:
        subj = r.get("subject", "unknown")
        by_subject.setdefault(subj, []).append(r)

    summary = {}
    for subj, recs in by_subject.items():
        gfs_means = [r["gfs_mean"] for r in recs if r.get("gfs_mean") is not None]
        slopes = [r["gfs_decay_slope"] for r in recs if r.get("gfs_decay_slope") is not None]
        if not gfs_means:
            continue
        m, lo, hi = bootstrap_ci(np.array(gfs_means))
        summary[subj] = {
            "n": len(gfs_means),
            "gfs_mean": m,
            "gfs_ci_95": (lo, hi),
            "mean_decay_slope": float(np.mean(slopes)) if slopes else None,
        }

    out_path = out_dir / "a3_by_subject.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"A3 saved to {out_path}: {len(summary)} subjects")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llava", choices=["llava", "internvl", "idefics2"])
    parser.add_argument("--output", default="results/ablations/")
    parser.add_argument("--ablations", nargs="+", default=["A1", "A2", "A3"])
    parser.add_argument("--n_samples", type=int, default=50, help="Samples for A1/A2")
    parser.add_argument("--hf_token", default=None)
    args = parser.parse_args()

    cfg = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    benchmark_path = cfg["paths"]["swap_pairs"]
    with open(benchmark_path) as f:
        triplets = json.load(f)
    samples = triplets[: args.n_samples]

    if "A3" in args.ablations:
        full_dir = Path(cfg["paths"]["results_full"]) / args.model
        ablation_a3_by_subject(full_dir, out_dir)

    if "A1" in args.ablations or "A2" in args.ablations:
        print("Loading ScienceQA ...")
        scienceqa_ds = load_dataset("derek-thomas/ScienceQA", split="train", trust_remote_code=True)
        model, processor, model_cfg = load_model_and_processor(args.model, hf_token=args.hf_token)

        if "A1" in args.ablations:
            ablation_a1_cot_vs_direct(model, processor, model_cfg, args.model, samples, scienceqa_ds, cfg, device, out_dir)
        if "A2" in args.ablations:
            ablation_a2_method_comparison(model, processor, model_cfg, args.model, samples, scienceqa_ds, cfg, device, out_dir)

    print("Ablations complete.")


if __name__ == "__main__":
    main()
