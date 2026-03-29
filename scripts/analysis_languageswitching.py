"""
Generates 3 comparison charts for the language switching technique results.
Reads:  annotated_results_languageswitching.csv
Writes: figures_ls/ls_01_asr_by_language_model.png
        figures_ls/ls_02_mean_asr_by_language.png
        figures_ls/ls_03_persona_adoption_by_language.png

"""

import pandas as pd
import matplotlib.pyplot as plt
import os

# Config

INPUT_FILE = "annotated_results_languageswitching.csv"
OUTPUT_DIR = "figures_ls"

# Map prompt IDs to languages — update if you add more prompts
LANG_MAP = {
    'LS01': 'Polish',   'LS02': 'Polish',   'LS03': 'Polish',   'LS04': 'Polish',
    'LS05': 'Japanese', 'LS06': 'Japanese', 'LS07': 'Japanese', 'LS08': 'Japanese',
    'LS10': 'Italian',  'LS11': 'Italian',  'LS12': 'Italian',
}

LANGUAGES = ["Polish", "Japanese", "Italian"]

MODEL_COLORS = {
    "llama-3.3-70b": "#2563EB",
    "gpt-oss-120b":  "#D97706",
    "qwen3-32b":     "#16A34A",
}

# Setup

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_FILE)
df["language"] = df["id"].map(LANG_MAP)
df["attack_success"] = pd.to_numeric(df["attack_success"], errors="coerce")
df["persona_adoption"] = pd.to_numeric(df["persona_adoption"], errors="coerce")

models = [m for m in ["llama-3.3-70b", "gpt-oss-120b", "qwen3-32b"] if m in df["model"].unique()]
n = len(models)
x = range(len(LANGUAGES))
width = 0.7 / n

def save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")

# 1. ASR by language × model (grouped bar)

pivot_asr = df.groupby(["language", "model"])["attack_success"].mean().unstack(fill_value=0)
pivot_asr = pivot_asr.reindex(LANGUAGES)

fig, ax = plt.subplots(figsize=(10, 5))
for i, model in enumerate(models):
    if model not in pivot_asr.columns:
        continue
    offset = (i - n / 2 + 0.5) * width
    bars = ax.bar(
        [xi + offset for xi in x],
        pivot_asr[model] * 100,
        width=width * 0.9,
        color=MODEL_COLORS[model],
        label=model, edgecolor="white", linewidth=0.5
    )
    ax.bar_label(bars, fmt="%.0f%%", padding=3, fontsize=8)

ax.set_xticks(list(x))
ax.set_xticklabels(LANGUAGES, fontsize=11)
ax.set_ylabel("Attack Success Rate (%)")
ax.set_ylim(0, 120)
ax.set_title("ASR by Language and Model — Language Switching Technique", fontweight="bold")
ax.legend(title="Model", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
save(fig, "ls_01_asr_by_language_model.png")

# 2. Mean ASR by language (across all models)

mean_asr = df.groupby("language")["attack_success"].mean().reindex(LANGUAGES) * 100

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(mean_asr.index, mean_asr.values,
              color=["#7C3AED", "#DB2777", "#EA580C"],
              edgecolor="white", linewidth=0.8)
ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=10)
ax.set_ylim(0, 100)
ax.set_ylabel("Mean ASR (%) across all models")
ax.set_title("Mean ASR by Language — Language Switching Technique", fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
save(fig, "ls_02_mean_asr_by_language.png")

# 3. Persona adoption rate by language × model

pivot_pa = df.groupby(["language", "model"])["persona_adoption"].apply(
    lambda x: (x >= 1).mean() * 100
).unstack(fill_value=0).reindex(LANGUAGES)

fig, ax = plt.subplots(figsize=(10, 5))
for i, model in enumerate(models):
    if model not in pivot_pa.columns:
        continue
    offset = (i - n / 2 + 0.5) * width
    bars = ax.bar(
        [xi + offset for xi in x],
        pivot_pa[model],
        width=width * 0.9,
        color=MODEL_COLORS[model],
        label=model, edgecolor="white", linewidth=0.5
    )
    ax.bar_label(bars, fmt="%.0f%%", padding=3, fontsize=8)

ax.set_xticks(list(x))
ax.set_xticklabels(LANGUAGES, fontsize=11)
ax.set_ylabel("Persona Adoption Rate (%) — PA ≥ 1")
ax.set_ylim(0, 120)
ax.set_title("Persona Adoption Rate by Language and Model", fontweight="bold")
ax.legend(title="Model", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
save(fig, "ls_03_persona_adoption_by_language.png")

print(f"\nDone — all figures saved to {OUTPUT_DIR}/")