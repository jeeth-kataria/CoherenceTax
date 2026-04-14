"""
Build 200 swap triplets from ScienceQA + GQA.

Usage:
    python scripts/build_benchmark.py --n 200 --output data/swap_pairs/
    python scripts/build_benchmark.py --n 20  --output data/swap_pairs/ --dry_run
"""

import argparse
import json
import sys
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.benchmark.swap_builder import build_swap_benchmark, save_benchmark
from src.models.loader import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200, help="Number of triplets to build")
    parser.add_argument("--output", default="data/swap_pairs/", help="Output directory")
    parser.add_argument("--min_pixel_diff", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true", help="Run with n=20 for quick test")
    args = parser.parse_args()

    cfg = load_config()

    if args.dry_run:
        args.n = 20
        print("DRY RUN: n=20")

    n = args.n
    min_pd = args.min_pixel_diff or cfg["benchmark"]["min_pixel_diff"]
    out_path = Path(args.output) / "benchmark_metadata.json"

    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
        print(f"Existing benchmark: {len(existing)} triplets at {out_path}")
        if len(existing) >= n and not args.dry_run:
            print("Already sufficient. Exit.")
            return

    print("Loading ScienceQA metadata ...")
    sqa_meta_path = Path("data/scienceqa/metadata.json")
    if not sqa_meta_path.exists():
        raise FileNotFoundError("Run scripts/prepare_data.py first.")
    with open(sqa_meta_path) as f:
        scienceqa_meta = json.load(f)

    gqa_meta_path = Path("data/gqa/metadata.json")
    gqa_meta = []
    if gqa_meta_path.exists():
        with open(gqa_meta_path) as f:
            gqa_meta = json.load(f)
    else:
        print("GQA metadata not found — using ScienceQA only.")

    print("Loading ScienceQA HF dataset ...")
    scienceqa_ds = load_dataset("derek-thomas/ScienceQA", split="train", trust_remote_code=True)

    gqa_ds = None
    if gqa_meta:
        try:
            print("Loading GQA HF dataset (streaming) ...")
            gqa_ds = load_dataset("lmms-lab/GQA", "testdev_balanced", split="testdev",
                                  streaming=True, trust_remote_code=True)
            gqa_ds = list(gqa_ds)  # materialize for random access
        except Exception as e:
            print(f"GQA load failed: {e}. Skipping GQA.")
            gqa_ds = None
            gqa_meta = []

    triplets = build_swap_benchmark(
        scienceqa_meta=scienceqa_meta,
        gqa_meta=gqa_meta,
        scienceqa_ds=scienceqa_ds,
        gqa_ds=gqa_ds or [],
        n_triplets=n,
        min_pixel_diff=min_pd,
        seed=args.seed,
        scienceqa_fraction=cfg["benchmark"]["scienceqa_fraction"],
    )

    # If < 60% success, retry with relaxed threshold
    if len(triplets) < int(n * 0.6):
        fallback_pd = cfg["benchmark"]["fallback_min_pixel_diff"]
        print(f"Low yield. Retrying with min_pixel_diff={fallback_pd} ...")
        triplets = build_swap_benchmark(
            scienceqa_meta=scienceqa_meta,
            gqa_meta=gqa_meta,
            scienceqa_ds=scienceqa_ds,
            gqa_ds=gqa_ds or [],
            n_triplets=n,
            min_pixel_diff=fallback_pd,
            seed=args.seed,
            scienceqa_fraction=cfg["benchmark"]["scienceqa_fraction"],
        )

    Path(args.output).mkdir(parents=True, exist_ok=True)
    save_benchmark(triplets, str(out_path))
    print(f"Built {len(triplets)} triplets -> {out_path}")


if __name__ == "__main__":
    main()
