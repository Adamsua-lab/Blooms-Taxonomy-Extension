"""Regenerate the bank-derived figures in results/figures from the delivered bank.

Same definitions as the BT4 notebook: entries per level, distribution of levels per
verb, and the mutual influence matrix. Run after retrain.py so the figures match the
bank that is actually shipped.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
BANK = REPO / "results" / "Verb List Classified by Model.csv"
FIG_DIR = REPO / "results" / "figures"
LEVELS = ["Kn", "Cm", "Ap", "An", "Sn", "Ev"]

df = pd.read_csv(BANK)
df = df[df["Accepted"] == True].reset_index(drop=True)  # noqa: E712
pat = df["AcceptedPattern"].astype(str).str.zfill(6)
for i, lv in enumerate(LEVELS):
    df[lv] = pat.str[i].astype(int)
print(f"accepted verbs: {len(df)} | entries: {int(df[LEVELS].to_numpy().sum())}")
print("entries per level:", {lv: int(df[lv].sum()) for lv in LEVELS})

FIG_DIR.mkdir(parents=True, exist_ok=True)

sums = [int(df[lv].sum()) for lv in LEVELS]
plt.figure(figsize=(7, 4))
plt.bar(LEVELS, sums)
plt.title("Entries per Cognitive Level")
plt.xlabel("Level"); plt.ylabel("# Entries")
plt.tight_layout(); plt.savefig(FIG_DIR / "fig_entries_per_level.png", dpi=200); plt.close()

nlab = df[LEVELS].sum(axis=1)
vals = nlab.value_counts().sort_index()
print("levels per verb:", dict(vals))
plt.figure(figsize=(7, 4))
plt.bar(vals.index.astype(int), vals.values)
plt.title("# of Cognitive Levels per Verb")
plt.xlabel("# levels per verb"); plt.ylabel("# verbs")
plt.tight_layout(); plt.savefig(FIG_DIR / "fig_nlabels_hist.png", dpi=200); plt.close()

total = len(df)
B = df[LEVELS].to_numpy(dtype=int)
subtotals = B.sum(axis=0)
M = np.full((6, 6), np.nan)
for i in range(6):
    for j in range(6):
        if i == j:
            continue
        count_ij = int((B[:, i] & B[:, j]).sum())
        denom = max(int(subtotals[i]) * int(subtotals[j]), 1)
        M[i, j] = (count_ij * total) / denom
plt.figure(figsize=(6, 5))
im = plt.imshow(M, interpolation="nearest")
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.xticks(range(6), LEVELS); plt.yticks(range(6), LEVELS)
plt.title("Mutual Influence Matrix")
plt.tight_layout(); plt.savefig(FIG_DIR / "fig_mutual_influence_heatmap.png", dpi=200); plt.close()

print("figures written to", FIG_DIR)
