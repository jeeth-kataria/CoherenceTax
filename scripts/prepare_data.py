"""
Download and cache ScienceQA + GQA.
Filters: must have image, question not yes/no.
Saves filtered metadata to data/scienceqa/ and data/gqa/.
"""

import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def is_yes_no(q: str) -> bool:
    t = q.strip().lower()
    starters = (
        "is ", "are ", "was ", "were ", "do ", "does ", "did ",
        "can ", "could ", "should ", "will ", "has ", "have ",
    )
    return t.endswith("?") and t.startswith(starters)


def prepare_scienceqa(output_dir: str, max_samples: int = None):
    out = Path(output_dir) / "scienceqa"
    out.mkdir(parents=True, exist_ok=True)

    meta_path = out / "metadata.json"
    if meta_path.exists():
        print(f"ScienceQA already cached at {meta_path}")
        with open(meta_path) as f:
            return json.load(f)

    print("Downloading ScienceQA ...")
    ds = load_dataset("derek-thomas/ScienceQA", split="train", trust_remote_code=True)

    records = []
    skipped = 0
    for i, ex in enumerate(tqdm(ds, desc="ScienceQA")):
        if max_samples and i >= max_samples:
            break

        # Must have image
        if ex.get("image") is None:
            skipped += 1
            continue

        q = (ex.get("question") or "").strip()
        if not q or is_yes_no(q):
            skipped += 1
            continue

        choices = ex.get("choices", [])
        answer_idx = ex.get("answer", 0)
        if not choices or answer_idx >= len(choices):
            skipped += 1
            continue

        records.append(
            {
                "id": f"scienceqa_{i}",
                "source": "scienceqa",
                "question": q,
                "answer": choices[answer_idx],
                "answer_idx": int(answer_idx),
                "choices": choices,
                "subject": ex.get("subject", ""),
                "topic": ex.get("topic", ""),
                "hf_index": i,
            }
        )

    print(f"ScienceQA: kept={len(records)} skipped={skipped}")

    with open(meta_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved to {meta_path}")
    return records


def prepare_gqa(output_dir: str, max_samples: int = 2000):
    out = Path(output_dir) / "gqa"
    out.mkdir(parents=True, exist_ok=True)

    meta_path = out / "metadata.json"
    if meta_path.exists():
        print(f"GQA already cached at {meta_path}")
        with open(meta_path) as f:
            return json.load(f)

    print("Downloading GQA (testdev_balanced, streaming) ...")
    try:
        ds = load_dataset(
            "lmms-lab/GQA",
            "testdev_balanced",
            split="testdev",
            streaming=True,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"GQA load failed: {e}")
        print("Skipping GQA — using ScienceQA only.")
        return []

    records = []
    for i, ex in enumerate(tqdm(ds, desc="GQA", total=max_samples)):
        if i >= max_samples:
            break
        if ex.get("image") is None:
            continue
        q = (ex.get("question") or "").strip()
        if not q or is_yes_no(q):
            continue
        records.append(
            {
                "id": f"gqa_{i}",
                "source": "gqa",
                "question": q,
                "answer": str(ex.get("answer", "")),
                "subject": "visual_reasoning",
                "topic": "",
                "hf_index": i,
            }
        )

    print(f"GQA: kept={len(records)}")
    with open(meta_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved to {meta_path}")
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data", help="Root data directory")
    parser.add_argument("--max_scienceqa", type=int, default=None)
    parser.add_argument("--max_gqa", type=int, default=2000)
    args = parser.parse_args()

    sqa = prepare_scienceqa(args.output, args.max_scienceqa)
    gqa = prepare_gqa(args.output, args.max_gqa)

    summary = {
        "scienceqa_total": len(sqa),
        "gqa_total": len(gqa),
        "combined_total": len(sqa) + len(gqa),
    }
    print("\nSummary:", json.dumps(summary, indent=2))

    summary_path = Path(args.output) / "dataset_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
