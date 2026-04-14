# Research Plan: VisualCOT-Swap

**Start date:** April 2026
**arXiv deadline:** June 1, 2026
**COLM submission:** ~July 1, 2026

---

## Week 1 (Apr 14–20): Infrastructure + Attribution Pilot
Goal: Prove all 3 models run, all 3 attribution methods produce non-trivial heatmaps.

- [ ] Install environment: `pip install -r requirements.txt`
- [ ] Run `prepare_data.py` — verify ScienceQA + GQA download correctly
- [ ] Load LLaVA-1.5-7B in 4-bit — verify CoT generation produces ≥3 steps
- [ ] Load InternVL2-8B in 4-bit — same verification
- [ ] Load Idefics2-8B in 4-bit — same verification
- [ ] Implement Grad-CAM — verify hooks fire, heatmap is non-uniform
- [ ] Implement attention rollout — verify heatmap is non-uniform
- [ ] Implement patch occlusion — verify sensitivity map makes visual sense
- [ ] Run pilot: 20 samples on LLaVA, save results to results/pilot/
- [ ] Qualitative check: do heatmaps look reasonable for step 1 vs step 4?

**Week 1 deliverable:** results/pilot/llava/pilot_summary.json with avg_steps ≥ 3

---

## Week 2 (Apr 21–27): Benchmark Construction
Goal: 200 valid swap triplets saved to disk, verified quality.

- [ ] Implement `build_swap_benchmark.py` — test with n=20 first
- [ ] Verify swap selection: check 10 pairs manually — are swaps visually different?
- [ ] Scale to n=200 — verify ≥180 valid triplets (90% success rate)
- [ ] Save to data/swap_pairs/benchmark_metadata.json
- [ ] Optional: use GPT-4o to annotate visual references in 50 samples (~$1)
- [ ] Compute inter-rater agreement on 20 samples (you + GPT-4o)
- [ ] Write benchmark section (§3.1) draft in paper/sections/method.tex

**Week 2 deliverable:** data/swap_pairs/benchmark_metadata.json with 200 triplets

---

## Week 3 (Apr 28 – May 10): Full GFS Experiment
Goal: Main results table with GFS decay across all 3 models.

- [ ] Run full GFS on LLaVA (200 samples) — save to results/full/llava/
- [ ] Run full GFS on InternVL2 — save to results/full/internvl/
- [ ] Run full GFS on Idefics2 — save to results/full/idefics2/
- [ ] Compute decay slopes — verify direction (positive OR negative both OK)
- [ ] Compute swap sensitivity scores
- [ ] Statistical tests: bootstrap CI + t-test on slopes
- [ ] Generate Figure 1 (GFS decay curve)
- [ ] Generate Figure 2 (swap sensitivity scatter)
- [ ] Generate Figure 3 (GFS heatmap grid)
- [ ] Generate Table 1 (main results) in LaTeX
- [ ] Write results section (§5) in paper/sections/results.tex

**Week 3 deliverable:** results/full/ populated, Table 1 + Figure 1 ready for paper

---

## Week 4 (May 11–25): Ablations + Paper Writing
Goal: Submission-ready paper draft.

- [ ] Ablation 1: CoT vs direct answer — does prompting style affect GFS?
- [ ] Ablation 2: Attribution method comparison (Grad-CAM vs rollout vs occlusion)
- [ ] Ablation 3: Question category analysis (visual vs reasoning-heavy questions)
- [ ] Generate ablation figures
- [ ] Write all paper sections
- [ ] Write references.bib (all 15+ citations)
- [ ] Compile paper/main.tex — verify it compiles without errors
- [ ] Internal review: read full paper as if you're a reviewer
- [ ] Fix any gaps in novelty defense or experimental validation

**Week 4 deliverable:** Complete paper draft, ready for co-author review

---

## Final Push (May 26 – Jun 1): Polish + arXiv
- [ ] Address internal review comments
- [ ] Final proofread: every number in paper matches results/ JSON
- [ ] All figures: PDF format, readable at 1-column width
- [ ] Submit to arXiv — get arXiv ID
- [ ] Update GitHub README with arXiv link + results table
- [ ] Post on Twitter/X and tag multimodal interpretability community

---

## July: COLM Submission
- [ ] Incorporate any feedback from arXiv comments
- [ ] Format check against COLM 2026 style requirements
- [ ] Submit

---

## Risk mitigations (already baked in)
- Null result = still publishable (two-sided hypothesis)
- GPT-4o annotation = reviewer-proof benchmark quality
- 3 models = robust finding even if one model is unusual
- Public dataset + code = guaranteed citation value
- arXiv first = timestamped, on applications before COLM reviews
