"""
Paper-quality figures — Skill 8 from SKILLS.md.
COLM 2026 format: 10pt text, serif font, 300 DPI.
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ── Style ─────────────────────────────────────────────────────────────────────
mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "lines.linewidth": 1.5,
    }
)

MODEL_COLORS = {
    "llava": "#534AB7",
    "internvl": "#1D9E75",
    "idefics2": "#D85A30",
}
MODEL_LABELS = {
    "llava": "LLaVA-1.5-7B",
    "internvl": "InternVL2-8B",
    "idefics2": "Idefics2-8B",
}


def save_figure(fig, output_dir: str, name: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{output_dir}/{name}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(f"{output_dir}/{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved {output_dir}/{name}.pdf + .png")


# ── Figure 1: GFS decay curve ─────────────────────────────────────────────────
def plot_gfs_decay(
    results_by_model: Dict[str, List[dict]],
    output_dir: str,
    max_steps: int = 6,
) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    for model_key, results in results_by_model.items():
        by_step = [[] for _ in range(max_steps)]
        for r in results:
            for k, s in enumerate(r.get("gfs_per_step", [])[:max_steps]):
                if s is not None and not np.isnan(s):
                    by_step[k].append(s)

        means = [np.mean(s) if s else np.nan for s in by_step]
        sems = [np.std(s) / np.sqrt(len(s)) if len(s) > 1 else 0 for s in by_step]
        valid = [i for i, m in enumerate(means) if not np.isnan(m)]
        if not valid:
            continue

        x = np.array(valid) + 1
        y = np.array([means[i] for i in valid])
        e = np.array([sems[i] for i in valid])

        ax.plot(
            x, y,
            label=MODEL_LABELS.get(model_key, model_key),
            color=MODEL_COLORS.get(model_key, "gray"),
            marker="o",
            markersize=4,
        )
        ax.fill_between(x, y - e, y + e, color=MODEL_COLORS.get(model_key, "gray"), alpha=0.15)

    ax.set_xlabel("Reasoning step $k$")
    ax.set_ylabel("Mean GFS")
    ax.legend(frameon=False)
    ax.set_xticks(range(1, max_steps + 1))
    ax.set_ylim(0, 1)
    save_figure(fig, output_dir, "fig1_gfs_decay")


# ── Figure 2: Swap sensitivity scatter ────────────────────────────────────────
def plot_swap_sensitivity(
    results_by_model: Dict[str, List[dict]],
    output_dir: str,
) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    for model_key, results in results_by_model.items():
        gfs = [r.get("gfs_mean") for r in results if r.get("gfs_mean") is not None]
        swap = [r.get("swap_sensitivity_score") for r in results if r.get("swap_sensitivity_score") is not None]
        if not gfs or not swap:
            continue
        ax.scatter(
            gfs, swap,
            label=MODEL_LABELS.get(model_key, model_key),
            color=MODEL_COLORS.get(model_key, "gray"),
            alpha=0.5,
            s=20,
        )

    ax.set_xlabel("Mean GFS (original image)")
    ax.set_ylabel("Swap sensitivity score")
    ax.legend(frameon=False)
    save_figure(fig, output_dir, "fig2_swap_sensitivity")


# ── Figure 3: GFS heatmap grid (qualitative) ──────────────────────────────────
def plot_heatmap_grid(
    sample_results: List[dict],
    images: List,
    output_dir: str,
    n_rows: int = 2,
    max_steps: int = 4,
) -> None:
    """
    sample_results: list of result dicts with heatmaps_per_step (np.ndarray list).
    images: corresponding PIL images.
    """
    import cv2
    n_cols = max_steps + 1  # original image + step heatmaps
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 2.0, n_rows * 2.2),
    )

    for row_i, (res, img) in enumerate(zip(sample_results[:n_rows], images[:n_rows])):
        # Original image
        axes[row_i, 0].imshow(img)
        axes[row_i, 0].set_title("Image", fontsize=8)
        axes[row_i, 0].axis("off")

        heatmaps = res.get("_heatmaps", [])
        for col_i in range(1, n_cols):
            ax = axes[row_i, col_i]
            step_idx = col_i - 1
            if step_idx < len(heatmaps) and heatmaps[step_idx] is not None:
                hm = heatmaps[step_idx]
                img_arr = np.array(img.resize((hm.shape[1], hm.shape[0])))
                overlay = img_arr.copy().astype(float)
                hm_colored = cv2.applyColorMap((hm * 255).astype(np.uint8), cv2.COLORMAP_JET)
                hm_colored_rgb = cv2.cvtColor(hm_colored, cv2.COLOR_BGR2RGB)
                blended = (0.5 * overlay + 0.5 * hm_colored_rgb).astype(np.uint8)
                ax.imshow(blended)
                gfs_val = res.get("gfs_per_step", [None] * (step_idx + 1))[step_idx]
                gfs_str = f"GFS={gfs_val:.2f}" if gfs_val is not None else "GFS=—"
                ax.set_title(f"Step {step_idx+1}\n{gfs_str}", fontsize=7)
            else:
                ax.axis("off")
            ax.axis("off")

    plt.tight_layout()
    save_figure(fig, output_dir, "fig3_heatmap_grid")


# ── Figure 4: Ablation — CoT vs direct ───────────────────────────────────────
def plot_cot_vs_direct(ablation_results: List[dict], output_dir: str) -> None:
    cot_gfs = [r.get("gfs_mean_cot") for r in ablation_results if r.get("gfs_mean_cot") is not None]
    direct_gfs = [r.get("gfs_mean_direct") for r in ablation_results if r.get("gfs_mean_direct") is not None]

    if not cot_gfs or not direct_gfs:
        print("A1 data missing — skipping plot")
        return

    fig, ax = plt.subplots(figsize=(4, 3.5))
    data = [cot_gfs, direct_gfs]
    labels = ["CoT", "Direct"]
    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5)
    for patch, color in zip(bp["boxes"], ["#534AB7", "#999999"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Mean GFS")
    ax.set_title("CoT vs Direct-Answer Prompting")
    save_figure(fig, output_dir, "fig4_cot_vs_direct")


# ── Table 1 LaTeX ─────────────────────────────────────────────────────────────
def results_to_latex_table(
    results_by_model: Dict[str, List[dict]],
    decay_stats: Dict[str, dict],
) -> str:
    """Generate Table 1 (main results) as LaTeX string."""
    from src.utils.stats import bootstrap_ci

    rows = []
    for model_key, results in results_by_model.items():
        gfs = [r["gfs_mean"] for r in results if r.get("gfs_mean") is not None]
        swap = [r["swap_sensitivity_score"] for r in results if r.get("swap_sensitivity_score") is not None]
        acc = [r.get("final_answer_correct", 0) for r in results]
        slopes = [r["gfs_decay_slope"] for r in results if r.get("gfs_decay_slope") is not None]

        if not gfs:
            continue

        gfs_m, gfs_lo, gfs_hi = bootstrap_ci(np.array(gfs))
        slope_m = float(np.mean(slopes)) if slopes else float("nan")
        swap_m = float(np.mean(swap)) if swap else float("nan")
        acc_m = float(np.mean(acc)) * 100 if acc else float("nan")

        d = decay_stats.get(model_key, {})
        p_str = f"{d.get('p_value', float('nan')):.3f}" if d.get("p_value") is not None else "—"
        sig = "*" if d.get("significant") else ""

        rows.append(
            f"{MODEL_LABELS.get(model_key, model_key)} & "
            f"{gfs_m:.3f} [{gfs_lo:.3f},{gfs_hi:.3f}] & "
            f"{slope_m:+.4f}{sig} & "
            f"{swap_m:.3f} & "
            f"{acc_m:.1f}\\% & "
            f"{p_str} \\\\"
        )

    header = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Main results: GFS across models. * = $p < 0.05$ on slope t-test.}\n"
        "\\label{tab:main_results}\n"
        "\\begin{tabular}{lccccc}\n"
        "\\toprule\n"
        "Model & Mean GFS (95\\% CI) & Decay slope & Swap sens. & Accuracy & $p$ \\\\\n"
        "\\midrule\n"
    )
    footer = "\\bottomrule\n\\end{tabular}\n\\end{table}"
    return header + "\n".join(rows) + "\n" + footer
