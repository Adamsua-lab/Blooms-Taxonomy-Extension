"""Recompute expert-agreement statistics against the NEW pipeline outputs.

The 8x30 expert votes are unchanged (votes attach to verbs, not to model
outputs). After normalization, 'copying' -> 'copy' (a core verb, excluded from
the extension) and 'resting' -> 'rest' (still a candidate). Agreement is
recomputed for the sampled verbs still present in the regenerated bank; the
excluded verb and any now-unaccepted verbs are reported explicitly.
"""
import math, os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
import numpy as np
import pandas as pd
from scipy import stats

LEVELS = ["Kn", "Cm", "Ap", "An", "Sn", "Ev"]
PIPE = os.path.dirname(os.path.abspath(__file__))
REPO = _REPO
_cands = ([sys.argv[1]] if len(sys.argv) > 1 else []) + [
    os.path.join(_HERE, "retrain_out", "results", "Verb List Classified by Model.csv"),
    os.path.join(_REPO, "..", "pipeline_rerun", "results", "Verb List Classified by Model.csv"),
    os.path.join(_REPO, "results", "Verb List Classified by Model.csv"),
]
NEWBANK = next(p for p in _cands if os.path.exists(p))
OUT = os.path.join(_HERE, "expert_recompute.md")

np.random.seed(13)
votes = pd.read_excel(REPO + r"\results\Final File(with model & Expert Classification).xlsx")
votes["verb_clean"] = votes["Verb_clean"].astype(str).str.strip().str.lower()
votes = votes[["verb_clean"] + [f"cnt_{l}" for l in LEVELS] + ["cnt_None"]]
bank = pd.read_csv(NEWBANK)
bank["verb"] = bank["Verb"].astype(str).str.strip().str.lower()

m = votes.merge(bank, left_on="verb_clean", right_on="verb", how="left", indicator=True)
missing = m[m["_merge"] == "left_only"]["verb_clean"].tolist()
m = m[m["_merge"] == "both"].copy()

def hard_set(r):
    return {l for l in LEVELS if r.get(f"accept_{l}", 0) == 1}
def soft_set(r):
    pat = str(r.get("SoftPattern", "")).split(".")[0].zfill(6)
    return {l for l, c in zip(LEVELS, pat) if c == "1"}

lines = ["# Expert agreement recomputed against the regenerated bank", ""]
lines.append(f"Sampled verbs matched in the new bank: {len(m)}/30.")
lines.append(f"Not in the extension any more: {missing} ('copy' resolved to a core verb under normalization).")

unacc = m[m["Accepted"] != 1]
lines.append(f"Matched but no longer hard-accepted by the new model: {unacc['verb_clean'].tolist()}")
acc = m[m["Accepted"] == 1].copy()
lines.append(f"Agreement metrics computed over the {len(acc)} hard-accepted matched verbs.")
lines.append("")

acc["Mhard"] = acc.apply(hard_set, axis=1)
acc["Msoft"] = acc.apply(soft_set, axis=1)
acc["Emax"] = acc.apply(lambda r: {l for l in LEVELS
                                   if r[f"cnt_{l}"] == max(r[f"cnt_{x}"] for x in LEVELS)
                                   and r[f"cnt_{l}"] > 0}, axis=1)
acc["E25"] = acc.apply(lambda r: {l for l in LEVELS if r[f"cnt_{l}"] >= 2}, axis=1)
acc["primary"] = acc["PrimaryDomain"]

def jpr(A, E):
    inter = len(A & E)
    return (inter / len(A | E) if A | E else 0.0,
            inter / len(A) if A else 0.0,
            inter / len(E) if E else 0.0)

n = len(acc)
strict = sum(1 for _, r in acc.iterrows() if r["primary"] in r["Emax"])
endorsed = sum(1 for _, r in acc.iterrows() if r["primary"] in r["E25"])
hj, hp, hr = zip(*[jpr(r["Mhard"], r["E25"]) for _, r in acc.iterrows()])
sj, sp, sr = zip(*[jpr(r["Mhard"] | r["Msoft"], r["E25"]) for _, r in acc.iterrows()])
cov_h = sum(1 for _, r in acc.iterrows() if r["Mhard"] & r["E25"])
cov_s = sum(1 for _, r in acc.iterrows() if (r["Mhard"] | r["Msoft"]) & r["E25"])

lines += [f"- Strict top-1: {strict}/{n} = {strict/n:.1%} (old model on 30 verbs: 20.0%)",
          f"- Endorsed top-1: {endorsed}/{n} = {endorsed/n:.1%} (old: 46.7%)",
          f"- Hard Jaccard/precision/recall: {np.mean(hj):.2f} / {np.mean(hp):.2f} / {np.mean(hr):.2f} (old: 0.17/0.46/0.18)",
          f"- Hard+soft: {np.mean(sj):.2f} / {np.mean(sp):.2f} / {np.mean(sr):.2f} (old: 0.31/0.53/0.42)",
          f"- Coverage of at least one endorsed level: hard {cov_h}/{n}, hard+soft {cov_s}/{n} (old: 15/30, 23/30)", ""]

def mcnemar(pairs):
    b = sum(1 for x, y in pairs if x and not y)
    c = sum(1 for x, y in pairs if y and not x)
    t = b + c
    p = 1.0 if t == 0 else min(1.0, sum(math.comb(t, k) for k in range(min(b, c) + 1)) * 2 / 2**t)
    return b, c, p
b, c, p = mcnemar([(bool(r["Mhard"] & r["E25"]), bool((r["Mhard"] | r["Msoft"]) & r["E25"])) for _, r in acc.iterrows()])
lines.append(f"- McNemar hard vs hard+soft coverage: discordant {b} vs {c}, exact p = {p:.4f}")

# consensus-band decomposition
acc["pmax"] = acc[[f"cnt_{l}" for l in LEVELS]].max(axis=1) / 8
mod = acc[acc["pmax"] >= 0.5]; low = acc[acc["pmax"] < 0.5]
hits_mod = sum(1 for _, r in mod.iterrows() if r["primary"] in r["E25"])
hits_low = sum(1 for _, r in low.iterrows() if r["primary"] in r["E25"])
fp = stats.fisher_exact([[hits_mod, len(mod)-hits_mod], [hits_low, len(low)-hits_low]]).pvalue
ch = (mod["E25"].apply(len) / 6).to_numpy()
sims = (np.random.rand(20000, len(mod)) < ch).mean(axis=1)
p_ch = float((sims >= (hits_mod/len(mod) if len(mod) else 0)).mean())
lines += ["", f"- Consensus decomposition: moderate/high consensus {hits_mod}/{len(mod)}, low consensus {hits_low}/{len(low)}, Fisher exact p = {fp:.4f}",
          f"- Moderate/high band vs conditional chance ({ch.mean():.0%}): simulation p = {p_ch:.4f}", ""]

# score-vote permutation
S = acc[[f"cal_score_{l}" for l in LEVELS]].to_numpy(dtype=float)
V = acc[[f"cnt_{l}" for l in LEVELS]].to_numpy(dtype=float)
def mean_rho(Smat, Vmat):
    rs = [stats.spearmanr(Smat[i], Vmat[i]).statistic for i in range(len(Smat))
          if np.std(Vmat[i]) > 0 and np.std(Smat[i]) > 0]
    return float(np.mean(rs))
obs = mean_rho(S, V)
idx = np.arange(len(acc)); perm = []
for _ in range(10000):
    np.random.shuffle(idx)
    perm.append(mean_rho(S[idx], V))
p_perm = float((np.asarray(perm) >= obs).mean())
lines.append(f"- Score-vote correlation: mean rho = {obs:.3f}, permutation p = {p_perm:.4f} (old model: 0.144, p = 0.072)")

open(OUT, "w", encoding="utf-8").write("\n".join(lines))
print("\n".join(lines))
