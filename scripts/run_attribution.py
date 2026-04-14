"""
Main GFS experiment script.
Loads one model, runs CoT generation + attribution for each sample.
Checkpoints every N samples. Resumes on reconnect.

Usage:
    python scripts/run_attribution.py --model llava --samples 200 --output results/full/
    python scripts/run_attribution.py --model llava --samples 20 --output results/pilot/ --pilot
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.attribution.gradcam import GradCAM
from src.attribution.rollout import attention_rollout, find_image_token_range
from src.metrics.gfs import compute_gfs_sequence, summarize_result
from src.models.loader import build_prompt, load_config, load_model_and_processor, parse_steps
from src.utils.stats import save_checkpoint, load_checkpoint


def get_image(ds, idx: int) -> Image.Image:
    img = ds[idx].get("image")
    if img is None:
        return None
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    return None


def run_single_sample(
    model,
    processor,
    model_cfg: dict,
    model_key: str,
    question: str,
    image: Image.Image,
    swap_image: Image.Image,
    cfg: dict,
    device: str,
    use_gradcam: bool = True,
) -> dict:
    gen_cfg = cfg["generation"]
    attr_cfg = cfg["attribution"]
    img_size = attr_cfg["image_size"]

    def infer(img: Image.Image):
        prompt = build_prompt(model_key, processor, question, cot=True)
        inputs = processor(
            text=prompt,
            images=img,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=gen_cfg["max_new_tokens"],
                do_sample=gen_cfg["do_sample"],
                num_beams=gen_cfg["num_beams"],
            )
        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        text = processor.decode(gen_ids, skip_special_tokens=True)
        return text, inputs

    # --- Original image ---
    gen_text, inputs_orig = infer(image)
    steps = parse_steps(gen_text)

    # --- Attribution per step ---
    n_patches = model_cfg.get("n_image_patches", 576)
    img_token_id = model_cfg.get("image_token_id", -200)
    layer_name = model_cfg.get("vision_encoder_layer", "")

    img_start, img_end = find_image_token_range(inputs_orig["input_ids"], img_token_id, n_patches)

    heatmaps_orig = []
    for step in steps[: cfg["gfs"]["max_steps"]]:
        step_inputs = processor(
            text=build_prompt(model_key, processor, step, cot=False),
            images=image,
            return_tensors="pt",
        ).to(device)

        if use_gradcam and layer_name:
            try:
                gc = GradCAM(model, layer_name, image_size=img_size)
                hm = gc.compute(step_inputs)
            except Exception:
                hm = attention_rollout(model, step_inputs, img_start, img_end, img_size)
        else:
            hm = attention_rollout(model, step_inputs, img_start, img_end, img_size)

        heatmaps_orig.append(hm)

    gfs_orig = compute_gfs_sequence(heatmaps_orig, steps[: cfg["gfs"]["max_steps"]])

    # --- Swap image ---
    gfs_swapped = None
    if swap_image is not None:
        swap_text, inputs_swap = infer(swap_image)
        swap_steps = parse_steps(swap_text)
        heatmaps_swap = []
        n_cmp = min(len(steps), len(swap_steps), cfg["gfs"]["max_steps"])
        for k in range(n_cmp):
            step = swap_steps[k]
            step_inputs = processor(
                text=build_prompt(model_key, processor, step, cot=False),
                images=swap_image,
                return_tensors="pt",
            ).to(device)
            if use_gradcam and layer_name:
                try:
                    gc = GradCAM(model, layer_name, image_size=img_size)
                    hm = gc.compute(step_inputs)
                except Exception:
                    hm = attention_rollout(model, step_inputs, img_start, img_end, img_size)
            else:
                hm = attention_rollout(model, step_inputs, img_start, img_end, img_size)
            heatmaps_swap.append(hm)

        gfs_swapped = compute_gfs_sequence(heatmaps_swap, swap_steps[:n_cmp])

    result = {
        "model_key": model_key,
        "question": question,
        "cot_text": gen_text,
        "steps": steps,
        "gfs_per_step": gfs_orig,
        "gfs_per_step_swapped": gfs_swapped,
        "n_steps": len(steps),
    }
    result.update(summarize_result(result))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["llava", "internvl", "idefics2"])
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--output", default="results/full/")
    parser.add_argument("--pilot", action="store_true", help="Pilot mode: save pilot_summary.json")
    parser.add_argument("--attribution", default="gradcam", choices=["gradcam", "rollout", "both"])
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--no_swap", action="store_true", help="Skip swap image attribution")
    args = parser.parse_args()

    cfg = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    out_dir = Path(args.output) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = str(out_dir / "checkpoint.json")

    # Resume from checkpoint
    completed = load_checkpoint(ckpt_path)
    done_ids = {r["triplet_id"] for r in completed}
    print(f"Resuming: {len(done_ids)} already done.")

    # Load benchmark
    benchmark_path = cfg["paths"]["swap_pairs"]
    if not Path(benchmark_path).exists():
        raise FileNotFoundError(f"Benchmark not found: {benchmark_path}. Run build_benchmark.py first.")
    with open(benchmark_path) as f:
        triplets = json.load(f)

    # Load dataset
    print("Loading ScienceQA ...")
    scienceqa_ds = load_dataset("derek-thomas/ScienceQA", split="train", trust_remote_code=True)

    # Load model
    model, processor, model_cfg = load_model_and_processor(
        args.model, hf_token=args.hf_token, device_map="auto"
    )

    use_gradcam = args.attribution in ("gradcam", "both")
    results = list(completed)
    start_time = time.time()

    todo = [t for t in triplets if t["triplet_id"] not in done_ids][: args.samples]
    print(f"Processing {len(todo)} samples ...")

    for i, triplet in enumerate(tqdm(todo)):
        tid = triplet["triplet_id"]
        orig = triplet["original"]
        swap = triplet.get("swap")

        orig_img = None
        swap_img = None

        if orig.get("source", triplet.get("source")) == "scienceqa":
            orig_img = get_image(scienceqa_ds, orig["hf_index"])
            if swap and not args.no_swap:
                swap_img = get_image(scienceqa_ds, swap["hf_index"])

        if orig_img is None:
            print(f"Skipping {tid}: no image")
            continue

        try:
            res = run_single_sample(
                model=model,
                processor=processor,
                model_cfg=model_cfg,
                model_key=args.model,
                question=orig["question"],
                image=orig_img,
                swap_image=swap_img if not args.no_swap else None,
                cfg=cfg,
                device=device,
                use_gradcam=use_gradcam,
            )
            res["triplet_id"] = tid
            res["subject"] = orig.get("subject", "")
            res["correct_answer"] = orig.get("answer", "")
            results.append(res)

            # Save per-sample
            sample_path = out_dir / f"{tid}.json"
            with open(sample_path, "w") as f:
                json.dump(res, f, default=str)

        except torch.cuda.OutOfMemoryError:
            print(f"OOM on {tid} — skipping")
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            print(f"Error on {tid}: {e}")
            continue

        # Checkpoint every N
        save_every = cfg["checkpointing"]["save_every_n"]
        if (i + 1) % save_every == 0:
            save_checkpoint(results, ckpt_path)
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (len(todo) - i - 1) / rate if rate > 0 else 0
            print(f"[{i+1}/{len(todo)}] ETA: {remaining/60:.1f} min")

    save_checkpoint(results, ckpt_path)

    # Pilot summary
    if args.pilot:
        valid_gfs = [r["gfs_mean"] for r in results if r.get("gfs_mean") is not None]
        valid_steps = [r["n_steps"] for r in results]
        pilot_summary = {
            "model": args.model,
            "n_samples": len(results),
            "avg_gfs": float(sum(valid_gfs) / len(valid_gfs)) if valid_gfs else None,
            "avg_steps": float(sum(valid_steps) / len(valid_steps)) if valid_steps else None,
            "samples_with_ge3_steps": sum(1 for s in valid_steps if s >= 3),
        }
        summary_path = out_dir / "pilot_summary.json"
        with open(summary_path, "w") as f:
            json.dump(pilot_summary, f, indent=2)
        print(f"\nPilot summary: {json.dumps(pilot_summary, indent=2)}")

    elapsed = time.time() - start_time
    print(f"\nDone. {len(results)} samples in {elapsed/60:.1f} min.")


if __name__ == "__main__":
    main()
