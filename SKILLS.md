# SKILLS.md — Research Engineering Reference

This file teaches Claude Code the specific skills needed to build this project correctly.
Read this whenever implementing a new component.

---

## Skill 1: Loading VLMs in 4-bit on Colab A100

```python
from transformers import BitsAndBytesConfig, AutoProcessor
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# LLaVA-1.5
from transformers import LlavaForConditionalGeneration
model = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

# InternVL2 / Idefics2 — use AutoModelForVision2Seq with trust_remote_code=True
from transformers import AutoModelForVision2Seq
model = AutoModelForVision2Seq.from_pretrained(
    "OpenGVLab/InternVL2-8B",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
```

**Memory tips:**
- A100 40GB handles all 3 models in 4-bit easily
- Call `torch.cuda.empty_cache()` between model loads
- Use `model.eval()` and `torch.no_grad()` for inference

---

## Skill 2: Correct CoT prompting for VLMs

The prompt must force numbered step output. Templates that work:

```python
# LLaVA template (uses conversation format)
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": (
                "Look at the image carefully. Answer the following question step by step, "
                "explicitly describing what you see in the image at each step.\n"
                f"Question: {question}\n"
                "Step 1:"
            )}
        ]
    }
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

# Idefics2 template
messages = [{"role": "user", "content": [
    {"type": "image"},
    {"type": "text", "text": f"Reason step by step using visual evidence.\nQuestion: {question}\nStep 1:"}
]}]
prompt = processor.apply_chat_template(messages, add_generation_prompt=False)
```

**Parsing steps from output:**
```python
import re
def parse_steps(text):
    # Try numbered steps first
    matches = re.findall(r'Step\s*\d+[:\.\)]\s*(.+?)(?=Step\s*\d+[:\.\)]|\Z)', text, re.DOTALL)
    if len(matches) >= 2:
        return [m.strip() for m in matches]
    # Fallback: split on double newlines
    parts = [p.strip() for p in text.split('\n\n') if p.strip()]
    return parts if parts else [text.strip()]
```

---

## Skill 3: Grad-CAM on vision encoders

The key insight: hook the LAST vision encoder layer, not any language model layer.

```python
# Find the right layer name first:
for name, module in model.named_modules():
    if 'encoder.layers' in name:
        print(name)
# Look for the last one: e.g. "vision_tower.vision_model.encoder.layers.23"

# Hook implementation
activations, gradients = {}, {}

def forward_hook(module, input, output):
    activations['feat'] = output[0].detach()  # [B, patches, C]

def backward_hook(module, grad_in, grad_out):
    gradients['feat'] = grad_out[0].detach()  # [B, patches, C]

handle_f = target_layer.register_forward_hook(forward_hook)
handle_b = target_layer.register_full_backward_hook(backward_hook)

# Forward + backward
model.zero_grad()
with torch.enable_grad():
    outputs = model(**inputs)
    score = outputs.logits[0, -1, :].sum()  # Last token logit sum
    score.backward()

# Compute CAM
weights = gradients['feat'].mean(dim=(0, 1))  # [C]
cam = (weights * activations['feat'][0]).sum(dim=-1)  # [patches]
cam = torch.relu(cam)

# Reshape to spatial grid and resize
n = cam.shape[0]
g = int(n**0.5)
cam_2d = cam.reshape(g, g).cpu().float().numpy()
cam_resized = cv2.resize(cam_2d, (336, 336))
cam_norm = (cam_resized - cam_resized.min()) / (cam_resized.max() + 1e-8)

handle_f.remove()
handle_b.remove()
```

**Common failure:** hooks don't fire because quantized layers don't always support full backward.
Fix: use `outputs.logits.float().sum().backward()` to force float32 gradient flow.

---

## Skill 4: Attention Rollout

```python
def attention_rollout(model, inputs, image_token_start, image_token_end):
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    attn_list = outputs.attentions  # list of [B, heads, seq, seq] per layer

    # Initialize rollout as identity
    seq_len = attn_list[0].shape[-1]
    rollout = torch.eye(seq_len, device=attn_list[0].device)

    for attn in attn_list:
        # Average over heads
        a = attn.squeeze(0).mean(0)  # [seq, seq]
        # Add residual (Abnar & Zuidema 2020)
        a = 0.5 * a + 0.5 * torch.eye(seq_len, device=a.device)
        a = a / a.sum(dim=-1, keepdim=True)
        rollout = a @ rollout

    # Extract image token attention from the last text token
    image_attn = rollout[-1, image_token_start:image_token_end]  # [n_image_tokens]
    n = image_attn.shape[0]
    g = int(n**0.5)
    heatmap = image_attn.reshape(g, g).cpu().float().numpy()
    heatmap = cv2.resize(heatmap, (336, 336))
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() + 1e-8)
    return heatmap

# To find image token positions for LLaVA:
# input_ids contains IMAGE_TOKEN_INDEX (-200) where image tokens go
# After processor, image tokens expand to n_patches = (336/14)^2 = 576 tokens
```

---

## Skill 5: Computing GFS correctly

GFS is NOT just IoU of the full heatmap with itself. The full pipeline:

```python
import spacy
nlp = spacy.load("en_core_web_sm")

def extract_visual_references(step_text):
    """Extract noun phrases that clearly reference visual content."""
    doc = nlp(step_text)
    visual_refs = []
    spatial_words = {'left', 'right', 'top', 'bottom', 'center', 'corner',
                     'background', 'foreground', 'object', 'image', 'picture'}
    for chunk in doc.noun_chunks:
        if any(w.lower_ in spatial_words for w in chunk):
            visual_refs.append(chunk.text)
        elif chunk.root.pos_ == 'NOUN' and chunk.root.dep_ in {'nsubj', 'dobj', 'pobj'}:
            visual_refs.append(chunk.text)
    return visual_refs

def compute_gfs(heatmap, step_text, threshold_pct=0.2):
    """
    GFS_k = fraction of high-attribution area that overlaps with
    the region expected from the visual reference in step_text.

    Simple version: use top-20% of heatmap as "attended region"
    and measure its spatial consistency (entropy-based).
    For paper: use CLIP-guided localization as the grounding region.
    """
    visual_refs = extract_visual_references(step_text)
    if not visual_refs:
        return None  # No visual reference — skip this step

    # Attended region: top threshold_pct of heatmap
    flat = heatmap.flatten()
    thresh = np.percentile(flat, (1 - threshold_pct) * 100)
    attended = (heatmap >= thresh).astype(float)

    # Concentration score: how focused is the attention?
    # High GFS = attention concentrated on specific region (not diffuse)
    total = attended.sum()
    if total == 0:
        return 0.0

    # Spatial entropy (lower = more concentrated = more faithful)
    p = heatmap / (heatmap.sum() + 1e-8)
    entropy = -np.sum(p * np.log(p + 1e-8))
    max_entropy = np.log(heatmap.size)
    concentration = 1.0 - (entropy / max_entropy)

    return float(concentration)
```

**For the full paper version:** Use CLIP to localize the noun phrase in the image,
then compute IoU between the CLIP-localized region and the attribution heatmap.
This is more rigorous but requires a CLIP call per step — budget it into Colab compute.

---

## Skill 6: Constructing valid swap pairs

The swap image must satisfy ALL three criteria:
1. Same semantic category (same ScienceQA subject/topic)
2. Visually different (pixel MSE > threshold after resize to 224x224)
3. Different correct answer (otherwise the swap doesn't test anything)

```python
import numpy as np
from PIL import Image

def is_valid_swap(original, candidate, min_pixel_diff=0.25):
    # Criterion 1: same subject
    if original.get('subject') != candidate.get('subject'):
        return False

    # Criterion 2: visually different
    img_a = np.array(original['image'].resize((224, 224))).astype(float) / 255.0
    img_b = np.array(candidate['image'].resize((224, 224))).astype(float) / 255.0
    mse = np.mean((img_a - img_b) ** 2)
    if mse < min_pixel_diff ** 2:
        return False

    # Criterion 3: different answer
    if original['answer'] == candidate['answer']:
        return False

    return True
```

**If <60% of samples get valid swaps:**
- Lower min_pixel_diff to 0.15
- Relax subject matching to topic-level
- Allow cross-dataset swaps (ScienceQA ↔ GQA)

---

## Skill 7: Saving results safely on Colab

Colab disconnects. Always save incrementally:

```python
import json
from pathlib import Path
from datetime import datetime

def save_result(result, output_dir, prefix="result"):
    """Save single result JSON with timestamp."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fname = out / f"{prefix}_{result['sample_id']}.json"
    with open(fname, 'w') as f:
        json.dump(result, f, default=str)  # default=str handles PIL images, numpy, etc.

def save_checkpoint(all_results, output_dir, model_key):
    """Save running checkpoint every N samples."""
    path = Path(output_dir) / f"{model_key}_checkpoint.json"
    with open(path, 'w') as f:
        json.dump(all_results, f, default=str)

def load_checkpoint(output_dir, model_key):
    """Resume from checkpoint if it exists."""
    path = Path(output_dir) / f"{model_key}_checkpoint.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []
```

---

## Skill 8: Generating paper-quality figures

```python
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np

# Paper-quality settings (COLM uses 10pt text, figures should match)
mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 1.5,
})

MODEL_COLORS = {'llava': '#534AB7', 'internvl': '#1D9E75', 'idefics2': '#D85A30'}
MODEL_LABELS = {'llava': 'LLaVA-1.5-7B', 'internvl': 'InternVL2-8B', 'idefics2': 'Idefics2-8B'}

def save_figure(fig, output_dir, name):
    """Always save both PDF and PNG."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{output_dir}/{name}.pdf", bbox_inches='tight', dpi=300)
    fig.savefig(f"{output_dir}/{name}.png", bbox_inches='tight', dpi=200)
    plt.close(fig)

# Figure 1: GFS decay curve
def plot_gfs_decay(results_by_model, output_dir):
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    max_steps = 6

    for model_key, results in results_by_model.items():
        # Collect per-step scores
        by_step = [[] for _ in range(max_steps)]
        for r in results:
            for k, s in enumerate(r['gfs_per_step'][:max_steps]):
                if s is not None and not np.isnan(s):
                    by_step[k].append(s)

        means = [np.mean(s) if s else np.nan for s in by_step]
        sems = [np.std(s)/np.sqrt(len(s)) if len(s) > 1 else 0 for s in by_step]
        valid = [i for i, m in enumerate(means) if not np.isnan(m)]

        x = np.array(valid) + 1
        y = np.array([means[i] for i in valid])
        e = np.array([sems[i] for i in valid])

        ax.plot(x, y, label=MODEL_LABELS[model_key],
                color=MODEL_COLORS[model_key], marker='o', markersize=4)
        ax.fill_between(x, y-e, y+e, color=MODEL_COLORS[model_key], alpha=0.15)

    ax.set_xlabel('Reasoning step k')
    ax.set_ylabel('Mean GFS')
    ax.legend(frameon=False)
    ax.set_xticks(range(1, max_steps+1))
    ax.set_ylim(0, 1)
    save_figure(fig, output_dir, 'fig1_gfs_decay')
```

---

## Skill 9: Statistical testing for paper claims

```python
import numpy as np
from scipy import stats

def bootstrap_ci(data, n_bootstrap=10000, ci=0.95, seed=42):
    """Compute bootstrap confidence interval for mean."""
    rng = np.random.default_rng(seed)
    boot_means = [rng.choice(data, len(data), replace=True).mean()
                  for _ in range(n_bootstrap)]
    lo = np.percentile(boot_means, (1-ci)/2 * 100)
    hi = np.percentile(boot_means, (1+ci)/2 * 100)
    return float(np.mean(data)), float(lo), float(hi)

def test_decay_significance(gfs_per_step_list):
    """
    Test whether GFS decay slope is significantly negative.
    Uses one-sample t-test on slopes across samples.
    """
    slopes = []
    for gfs_seq in gfs_per_step_list:
        valid = [(k, s) for k, s in enumerate(gfs_seq) if s is not None]
        if len(valid) < 3:
            continue
        x = np.array([v[0] for v in valid])
        y = np.array([v[1] for v in valid])
        slope = np.polyfit(x, y, 1)[0]
        slopes.append(slope)

    t_stat, p_value = stats.ttest_1samp(slopes, 0)
    mean_slope, lo, hi = bootstrap_ci(slopes)
    return {
        'mean_slope': mean_slope,
        'ci_95': (lo, hi),
        't_stat': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'n_samples': len(slopes),
    }

# Use in paper: "Mean GFS decay slope = X (95% CI [Y, Z], t(N)=T, p<0.05)"
```

---

## Skill 10: Writing LaTeX tables from JSON results

```python
def results_to_latex_table(results_by_model):
    """
    Generate Table 1 (main results) as LaTeX.
    Columns: Model | Mean GFS | Decay slope | Swap sensitivity | Accuracy
    """
    rows = []
    for model_key, results in results_by_model.items():
        gfs = [r['gfs_mean'] for r in results]
        slopes = [r['gfs_decay_slope'] for r in results]
        swap = [r['swap_sensitivity_score'] for r in results]
        acc = [r['final_answer_correct'] for r in results]

        gfs_m, gfs_lo, gfs_hi = bootstrap_ci(gfs)
        slope_m = np.mean(slopes)
        swap_m = np.mean(swap)
        acc_m = np.mean(acc) * 100

        rows.append(
            f"{MODEL_LABELS[model_key]} & "
            f"{gfs_m:.3f} [{gfs_lo:.3f},{gfs_hi:.3f}] & "
            f"{slope_m:+.4f} & "
            f"{swap_m:.3f} & "
            f"{acc_m:.1f}\\% \\\\"
        )

    header = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Main results: GFS across models and reasoning depths.}\n"
        "\\label{tab:main_results}\n"
        "\\begin{tabular}{lcccc}\n"
        "\\toprule\n"
        "Model & Mean GFS (95\\% CI) & Decay slope & Swap sens. & Accuracy \\\\\n"
        "\\midrule\n"
    )
    footer = "\\bottomrule\n\\end{tabular}\n\\end{table}"

    return header + "\n".join(rows) + "\n" + footer
```

---

## Skill 11: Optional GPT-4o annotation (spend wisely)

Use GPT-4o ONLY to annotate which reasoning steps contain visual references.
This replaces manual annotation and costs ~$0.02 per sample × 200 samples = ~$4.

```python
from openai import OpenAI

client = OpenAI()  # Set OPENAI_API_KEY env var

ANNOTATION_PROMPT = """You are annotating reasoning steps from a Vision-Language Model.
For each step below, output JSON: {"has_visual_reference": true/false, "visual_nouns": ["list", "of", "nouns"]}
A visual reference is when the step explicitly describes something seen in the image.

Reasoning step: {step}

Output only valid JSON, nothing else."""

def annotate_step_with_gpt4o(step_text, budget_tracker):
    """
    Use GPT-4o to annotate whether a step contains visual references.
    budget_tracker: dict with 'spent_usd' key — stops if > 8.0
    """
    if budget_tracker['spent_usd'] >= 8.0:
        return None  # Budget exhausted, use rule-based fallback

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": ANNOTATION_PROMPT.format(step=step_text)}],
        max_tokens=100,
        temperature=0,
    )

    # Track cost (gpt-4o: ~$0.005 per 1K input tokens)
    input_tokens = response.usage.prompt_tokens
    budget_tracker['spent_usd'] += input_tokens * 0.000005
    return response.choices[0].message.content
```

---

## Skill 12: Running multiple Colab instances in parallel

For 3 models × 200 samples, use 3 separate Colab notebooks in parallel.
Each notebook runs one model. Share results via Google Drive.

Template for each notebook:
```python
# Notebook header for each Colab instance
import subprocess
subprocess.run(["pip", "install", "-r", "requirements.txt", "-q"])
subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm", "-q"])

# Mount Drive for shared data
from google.colab import drive
drive.mount('/content/drive')

# Clone repo
import os
if not os.path.exists('visualcot_faithfulness'):
    subprocess.run(["git", "clone", "YOUR_GITHUB_REPO_URL"])

os.chdir('visualcot_faithfulness')

# Run model-specific experiment
MODEL = "llava"  # Change to "internvl" or "idefics2" for other notebooks
subprocess.run([
    "python", "scripts/run_gfs.py",
    "--models", MODEL,
    "--benchmark", "/content/drive/MyDrive/visualcot/data/swap_pairs/benchmark_metadata.json",
    "--output", f"/content/drive/MyDrive/visualcot/results/full/",
])
```
