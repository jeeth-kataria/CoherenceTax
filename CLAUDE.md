# CLAUDE.md — VisualCOT-Swap

## What this project does

Tests whether VLMs stay visually grounded across CoT reasoning steps.
Key claim: GFS (Grounding Faithfulness Score) decays with step depth.
Swap-pair test confirms: later steps are hallucinated, not image-grounded.

## Three models

- LLaVA-1.5-7B: `llava-hf/llava-1.5-7b-hf`
- InternVL2-8B: `OpenGVLab/InternVL2-8B`
- Idefics2-8B: `HuggingFace/idefics2-8b`

All loaded in 4-bit via BitsAndBytesConfig. A100 40GB handles all three.

## Two datasets

- ScienceQA: `derek-thomas/ScienceQA` (HuggingFace) — has images, subjects, answers
- GQA: `lmms-lab/GQA` — visual reasoning, different distribution

## Pipeline

```
scripts/prepare_data.py          # download + cache ScienceQA + GQA
scripts/build_benchmark.py       # construct 200 swap triplets
scripts/run_attribution.py       # main GFS experiment (1 model at a time)
scripts/run_ablations.py         # CoT vs direct, attribution method comparison
src/utils/visualization.py       # figures
src/utils/stats.py               # bootstrap CI, t-tests
```

## Run order

```bash
pip install -r requirements.txt

# Colab: 3 notebooks in parallel (1 per model)
python scripts/prepare_data.py --output data/
python scripts/build_benchmark.py --n 200 --output data/swap_pairs/
python scripts/run_attribution.py --model llava --samples 200 --output results/full/
python scripts/run_attribution.py --model internvl --samples 200 --output results/full/
python scripts/run_attribution.py --model idefics2 --samples 200 --output results/full/
python scripts/run_ablations.py --output results/ablations/
```

## Colab session plan

- Session 1 (~2h): LLaVA pilot (20 samples) → Week 1 deliverable
- Session 2 (~3h): LLaVA full (200 samples)
- Session 3 (~3h): InternVL2 full
- Session 4 (~3h): Idefics2 full
- Session 5 (~1h): ablations + figures + stats

## Key design decisions

- 4-bit quantization: fits all 7B/8B models on A100 40GB
- Grad-CAM on last vision encoder layer (not LM layers)
- Attention rollout as fallback if Grad-CAM hooks fail on quantized model
- Swap criterion: same subject + pixel MSE > 0.25^2 + different answer
- GFS = 1 - normalized spatial entropy of attribution heatmap
- Bootstrap CI (10k) + one-sample t-test on decay slopes = primary statistics
- Save every sample individually — checkpoint recovery on disconnect

## Known issues

- Grad-CAM backward may fail on 4-bit: fix = `outputs.logits.float().sum().backward()`
- InternVL2 requires `trust_remote_code=True`
- ScienceQA images: some samples have no image (text-only) — skip these
- GQA images: large download (~20GB) — use streaming or subset

## Fallbacks

- If InternVL2 OOM: use `OpenGVLab/InternVL2-4B`
- If Idefics2 OOM: use `HuggingFace/idefics2-8b` with `device_map="auto"`
- If < 60% valid swaps: lower `min_pixel_diff` to 0.15, relax to topic-level matching

## Linked papers

Paper 1 — CoherenceBench-IN (prior work by author)
Paper 2 — AlignTax (prior work by author)
Paper 3 — The Coherence Tax (parallel paper, ~/Research/CoherenceTax/)
Paper 4 — VisualCOT-Swap (THIS paper) — COLM 2026 target, arXiv June 1 2026
