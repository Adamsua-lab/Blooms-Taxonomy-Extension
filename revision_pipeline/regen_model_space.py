"""Regenerate results/figures/model_space_2d.png from the delivered bank.

Same projection and styling as the BT4 notebook cell (UMAP when available, PCA
otherwise), coloured by the primary domain of each accepted verb.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
LEVELS = ["Kn", "Cm", "Ap", "An", "Sn", "Ev"]
PALETTE = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
CMAP = ListedColormap(PALETTE)

df = pd.read_csv(REPO / "results" / "Verb List Classified by Model.csv")
df = df[(df["Accepted"] == True) & (df["PrimaryDomain"].isin(LEVELS))].reset_index(drop=True)  # noqa: E712

z = np.load(HERE / "all_embeddings.npz", allow_pickle=True)
emb = {str(w).strip().lower(): z["vectors"][i] for i, w in enumerate(z["words"])}
keep = [i for i, v in enumerate(df["Verb"].astype(str).str.lower()) if v in emb]
df = df.iloc[keep].reset_index(drop=True)
X = np.stack([emb[v] for v in df["Verb"].astype(str).str.lower()])
print(f"points: {len(df)} | dim: {X.shape[1]}")

try:
    import umap
    Z = umap.UMAP(n_components=2, random_state=42).fit_transform(X)
    method = "UMAP"
except Exception:
    from sklearn.decomposition import PCA
    Z = PCA(n_components=2, random_state=42).fit_transform(X)
    method = "PCA"
print("projection:", method)

y = np.array([LEVELS.index(x) for x in df["PrimaryDomain"].astype(str)], dtype=int)
plt.figure(figsize=(7.8, 6.2))
plt.scatter(Z[:, 0], Z[:, 1], c=y, cmap=CMAP, vmin=0, vmax=len(LEVELS) - 1, s=10, alpha=0.80)
plt.title(f"Bloom Model Visualization ({method})")
plt.xticks([]); plt.yticks([])
handles = [plt.Line2D([], [], marker="o", linestyle="", label=lv, markersize=6,
                      markerfacecolor=PALETTE[i], markeredgecolor="none")
           for i, lv in enumerate(LEVELS)]
plt.legend(handles=handles, title="Primary domain", loc="lower left", frameon=True)
out = REPO / "results" / "figures" / "model_space_2d.png"
plt.tight_layout(); plt.savefig(out, dpi=220); plt.close()
print("saved", out)
