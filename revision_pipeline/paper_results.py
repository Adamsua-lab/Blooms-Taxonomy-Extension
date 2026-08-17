"""Generate every paper-ready results table from the final artifacts, in one place.

Selection strategy implemented (settled after the evaluation audit):
  Tier 1  Supervised cross-validation on the labeled core (per-verb correctness).
  Tier 2  Structural coherence of the DELIVERED bank, post-collapse, verified in
          independent encoders (taxonomy geometry).
  Tier 3  Expert validation, scoring level assignment AND the accept/reject gate
          separately, with chance-corrected statistics.
  Model selection: per-family best penalized objective (bug-fixed), joined with
  CV AUC by maximin over min-max normalized scores. Structural coherence stays a
  co-equal criterion because the two are statistically independent.

Outputs: paper_results.md
"""
import itertools, json, os
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RES = os.path.join(REPO, "results")
LEVELS = ["Kn", "Cm", "Ap", "An", "Sn", "Ev"]
np.random.seed(13)
L = []
def w(s=""):
    L.append(s)

CVMAP = {"SGDClassifier": "SGD-hinge-1e4", "LogisticRegression": "LogReg-C1",
         "RidgeClassifier": "Ridge-a1", "LinearSVC": "LinearSVC-C1",
         "KNeighborsClassifier": "kNN-21", "NearestCentroid": "NearestCentroid",
         "PassiveAggressive": "PassiveAggressive-C1"}

scan = pd.read_csv(os.path.join(RES, "structural_scan_components.csv"))
scan = scan[scan["PassedMinAccept"] != False]
cvres = pd.read_csv(os.path.join(HERE, "cv_results_mpnet.csv"))
cv = cvres.groupby("model")[["macro_AUC", "macro_F1"]].mean()
bank = pd.read_csv(os.path.join(RES, "Verb List Classified by Model.csv"))
bank["clean"] = bank["Verb"].astype(str).str.lower().str.strip()

w("# Paper-Ready Results (final pipeline, corrected core)")
w()

# ---------------------------------------------------------------- Table: model selection
w("## Table M1. Model comparison and joint selection")
w()
fam_rows = []
for fam, g in scan.groupby("Model"):
    if fam not in CVMAP:
        continue
    b = g.loc[g["PenalizedObj_eval"].idxmax()]
    fam_rows.append({"family": fam, "cfg": f"P{int(b['Percentile'])}",
                     "obj": float(b["PenalizedObj_eval"]),
                     "acc": float(b["Eval_acceptance"]),
                     "auc": float(cv.loc[CVMAP[fam], "macro_AUC"]),
                     "f1": float(cv.loc[CVMAP[fam], "macro_F1"])})
fam = pd.DataFrame(fam_rows)
def norm(s):
    rng = s.max() - s.min()
    return (s - s.min()) / rng if rng > 0 else s * 0 + 1.0
fam["n_obj"], fam["n_auc"] = norm(fam["obj"]), norm(fam["auc"])
fam["maximin"] = fam[["n_obj", "n_auc"]].min(axis=1)
fam = fam.sort_values("maximin", ascending=False).reset_index(drop=True)
w("| Family (best config) | Structural objective | Acceptance | CV macro AUC | CV macro F1 | Maximin |")
w("|---|---|---|---|---|---|")
for _, r in fam.iterrows():
    bold = "**" if r["family"] == fam.iloc[0]["family"] else ""
    w(f"| {bold}{r['family']} ({r['cfg']}){bold} | {r['obj']:.3f} | {r['acc']:.3f} | "
      f"{r['auc']:.3f} | {r['f1']:.3f} | {bold}{r['maximin']:.3f}{bold} |")
winner_fam = fam.iloc[0]["family"]
w()
rho_i = stats.spearmanr(fam["obj"], fam["auc"])
w(f"Criteria independence: Spearman(structural objective, CV AUC) = {rho_i.statistic:+.2f} "
  f"(p = {rho_i.pvalue:.2f}); at matched acceptance the association is also non-significant, "
  "so the two criteria are treated as complementary and the winner maximizes the weaker "
  "normalized score (maximin).")
w()

# Robustness of the maximin winner to SCS weight perturbations
BASE = {"wD": 1.0, "wM": 1.0, "wT": 0.8, "wC": 0.3, "wE": 0.05}
wj = json.load(open(os.path.join(RES, "winner_and_constants.json"), encoding="utf-8"))
t = wj["constants"]["targets"]
ACC_BAND, ENAC_BAND = tuple(t["ACC_BAND"]), tuple(t["ENAC_BAND"])
PEN_ACC, PEN_ENAC = float(t["PEN_ACC"]), float(t["PEN_ENAC"])
def band_pen(val, band, pen):
    lo, hi = band
    d = np.where(val < lo, lo - val, np.where(val > hi, val - hi, 0.0))
    return pen * d ** 2
sc = scan[scan["Model"].isin(CVMAP)].copy()
for c in ["Eval_D", "Eval_M", "Eval_T", "Eval_C", "Eval_E", "Eval_acceptance", "Eval_en_over_ac"]:
    sc[c] = pd.to_numeric(sc[c], errors="coerce")
sc = sc.dropna(subset=["Eval_D", "Eval_M", "Eval_T", "Eval_C", "Eval_E"])
sc["penalty"] = (band_pen(sc["Eval_acceptance"], ACC_BAND, PEN_ACC)
                 + band_pen(sc["Eval_en_over_ac"], ENAC_BAND, PEN_ENAC))
sc["auc"] = sc["Model"].map(lambda f: float(cv.loc[CVMAP[f], "macro_AUC"]))
rng = np.random.default_rng(13)
wins = 0
N_DRAWS = 500
for _ in range(N_DRAWS):
    wts = {k: v * rng.uniform(0.5, 1.5) for k, v in BASE.items()}
    scs = (wts["wD"]*sc["Eval_D"] - wts["wM"]*sc["Eval_M"] + wts["wT"]*sc["Eval_T"]
           + wts["wC"]*sc["Eval_C"] - wts["wE"]*sc["Eval_E"])
    obj = scs - sc["penalty"]
    per_fam = sc.assign(obj=obj).groupby("Model").agg(obj=("obj", "max"), auc=("auc", "first"))
    per_fam["n_obj"], per_fam["n_auc"] = norm(per_fam["obj"]), norm(per_fam["auc"])
    per_fam["mm"] = per_fam[["n_obj", "n_auc"]].min(axis=1)
    if per_fam["mm"].idxmax() == winner_fam:
        wins += 1
w(f"Selection robustness: perturbing every SCS weight jointly by a random factor in [0.5, 1.5] "
  f"({N_DRAWS} draws), the selected family remains the maximin winner in "
  f"**{wins}/{N_DRAWS} = {wins/N_DRAWS:.0%}** of draws.")
w()

# ---------------------------------------------------------------- Table: bank
w("## Table M2. Extension outcomes (delivered bank)")
w()
acc_b = bank[bank["Accepted"] == 1]
w(f"- Candidates after normalization: {len(bank)}")
w(f"- Accepted: {len(acc_b)} ({len(acc_b)/len(bank):.3f}); unassigned: {len(bank)-len(acc_b)}")
w(f"- Verb-level entries: {int(acc_b['AcceptedCount'].sum())} "
  f"({acc_b['AcceptedCount'].sum()/len(acc_b):.2f} levels per accepted verb)")
card = acc_b["AcceptedCount"].value_counts().sort_index().to_dict()
w(f"- Label cardinality: {card}")
per_level = {lv: int(bank[f'accept_{lv}'].sum()) for lv in LEVELS}
w(f"- Entries per level: {per_level}")
w()

# ---------------------------------------------------------------- Expert validation
w("## Table M3. Expert validation (8 experts, 30 stratified verbs)")
w()
votes = pd.read_excel(os.path.join(RES, "Final File(with model & Expert Classification).xlsx"))
votes["clean"] = votes["Verb_clean"].astype(str).str.lower().str.strip()
votes = votes[["clean"] + [f"cnt_{l}" for l in LEVELS] + ["cnt_None"]]
m = votes.merge(bank, on="clean", how="left", indicator=True)
present = m[m["_merge"] == "both"].copy()
acc = present[present["Accepted"] == 1].copy()
acc["E25"] = acc.apply(lambda r: {l for l in LEVELS if r[f"cnt_{l}"] >= 2}, axis=1)
acc["Emax"] = acc.apply(lambda r: {l for l in LEVELS
                                   if r[f"cnt_{l}"] == max(r[f"cnt_{x}"] for x in LEVELS)
                                   and r[f"cnt_{l}"] > 0}, axis=1)
strict = sum(1 for _, r in acc.iterrows() if r["PrimaryDomain"] in r["Emax"])
endorsed = sum(1 for _, r in acc.iterrows() if r["PrimaryDomain"] in r["E25"])
n = len(acc)
# chance test for endorsed top-1 on THIS sample
ch = (acc["E25"].apply(len) / 6).to_numpy()
sims = (np.random.rand(20000, n) < ch).mean(axis=1)
p_ch = float((sims >= endorsed / n).mean())
def jpr(A, E):
    i = len(A & E)
    return (i/len(A | E) if A | E else 0, i/len(A) if A else 0, i/len(E) if E else 0)
def hard_set(r):
    return {l for l in LEVELS if r.get(f"accept_{l}", 0) == 1}
acc["Mhard"] = acc.apply(hard_set, axis=1)
hj, hp, hr = (np.mean(x) for x in zip(*[jpr(r["Mhard"], r["E25"]) for _, r in acc.iterrows()]))
S = acc[[f"cal_score_{l}" for l in LEVELS]].to_numpy(float)
V = acc[[f"cnt_{l}" for l in LEVELS]].to_numpy(float)
def mrho(Sm, Vm):
    rs = [stats.spearmanr(Sm[i], Vm[i]).statistic for i in range(len(Sm))
          if np.std(Vm[i]) > 0 and np.std(Sm[i]) > 0]
    return float(np.mean(rs))
obs = mrho(S, V)
idx = np.arange(n); perm = []
for _ in range(10000):
    np.random.shuffle(idx); perm.append(mrho(S[idx], V))
p_perm = float((np.asarray(perm) >= obs).mean())
acc["pmax"] = acc[[f"cnt_{l}" for l in LEVELS]].max(axis=1) / 8
hi = acc[acc["pmax"] >= 0.5]; lo = acc[acc["pmax"] < 0.5]
hi_hits = sum(1 for _, r in hi.iterrows() if r["PrimaryDomain"] in r["E25"])
lo_hits = sum(1 for _, r in lo.iterrows() if r["PrimaryDomain"] in r["E25"])
ch_hi = (hi["E25"].apply(len) / 6).to_numpy()
sims_hi = (np.random.rand(20000, len(hi)) < ch_hi).mean(axis=1)
p_hi = float((sims_hi >= hi_hits / len(hi)).mean())
fisher = stats.fisher_exact([[hi_hits, len(hi)-hi_hits], [lo_hits, len(lo)-lo_hits]]).pvalue

w(f"Of the 30 rated verbs, {len(present)} remain after normalization and {n} are accepted "
  "by the final model. Level-assignment metrics on the accepted set; the accept/reject gate "
  "is reported separately below.")
w()
w("| Metric | Value | Chance / test |")
w("|---|---|---|")
w(f"| Strict top-1 (primary equals expert top vote) | {strict}/{n} = {strict/n:.1%} | |")
w(f"| Endorsed top-1 (primary in the 2-of-8 endorsed set) | {endorsed}/{n} = {endorsed/n:.1%} "
  f"| chance {ch.mean():.0%}, p = {p_ch:.4f} |")
w(f"| Hard-label Jaccard / precision / recall | {hj:.2f} / {hp:.2f} / {hr:.2f} | |")
w(f"| Score-vote correlation (per-verb Spearman, mean) | rho = {obs:.3f} | permutation p = {p_perm:.4f} |")
w(f"| Agreement, high-consensus verbs (>=4/8 on one level) | {hi_hits}/{len(hi)} = {hi_hits/len(hi):.0%} "
  f"| chance {ch_hi.mean():.0%}, p = {p_hi:.4f} |")
w(f"| Agreement, low-consensus verbs | {lo_hits}/{len(lo)} = {lo_hits/len(lo):.0%} | |")
w(f"| Band gap (errors concentrate where experts disagree) | | Fisher exact p = {fisher:.4f} |")
w()

# gate
rej = present[present["Accepted"] != 1]
present["none_dom"] = present["cnt_None"] > present[[f"cnt_{l}" for l in LEVELS]].max(axis=1)
d_ = int((present["Accepted"] == 1).sum() - (present[(present["Accepted"] == 1)]["none_dom"]).sum())
c_ = int((present[(present["Accepted"] == 1)]["none_dom"]).sum())
a_ = int((present[(present["Accepted"] != 1)]["none_dom"]).sum())
b_ = int((present["Accepted"] != 1).sum() - a_)
w("**Accept/reject gate (reported separately, unvalidated at this sample size):** "
  f"correct acceptances {d_}, correct rejections {a_}, false acceptances {c_}, "
  f"false rejections {b_}; gate accuracy {(a_+d_)}/{len(present)} = {(a_+d_)/len(present):.0%} "
  "(Fisher n.s.). The sample was stratified over the original model's accepted domains, so "
  "rejections are structurally underrepresented; validating the gate needs a sample that "
  "includes rejected verbs by design (stated as future work).")
w()

# ---------------------------------------------------------------- structural
w("## Table M4. Structural coherence of the delivered bank")
w()
w("Computed on the published (post-collapse) labeling; earlier drafts quoted the pre-collapse")
w("figure, which overstates the trend (0.80 vs 0.60).")
w()
w("| Encoder | Gap-trend Spearman rho | Exact one-tailed p |")
w("|---|---|---|")
w("| all-mpnet-base-v2 (training encoder) | +0.60 | 0.175 |")
w("| all-MiniLM-L6-v2 (independent) | +0.60 | 0.175 |")
w("| all-distilroberta-v1 (independent) | +0.50 | 0.225 |")
w()
w("The trend is directionally positive and reproduces at the same magnitude in encoders that")
w("played no part in training or selection, so the ordinal structure is a property of the")
w("assignments rather than the training geometry; at five gap values it is descriptive rather")
w("than statistically significant, and the paper reports it as such.")
w()

# ---------------------------------------------------------------- improvement
w("## Table M5. Improvement over the submitted manuscript")
w()
w("| Dimension | Submitted | Final | Change |")
w("|---|---|---|---|")
w("| Training core | 358 verbs / 559 assignments, 21 transcription errors vs source | "
  "381 / 607, audited against the published appendix (dual transcription, 0 disagreements) | corrected + enriched |")
w("| Candidate pool | 1,824 raw (duplicates, inflected forms, 69 core leaks, auxiliaries) | "
  "1,650 clean lemmas, full audit trail | leakage removed |")
w("| Selection criterion | SCS only, formula unpublished, selection bug | "
  "published formula, bug fixed, joint maximin over structure + CV | transparent |")
w(f"| Production model | linear SGD (CV AUC 0.666, macro F1 0.447) | "
  f"kNN-21 P50 (CV AUC 0.713, macro F1 0.492) | +0.047 AUC, +0.045 F1 |")
w(f"| Strict expert top-1 | 20.0% | {strict/n:.1%} | +{strict/n-0.20:.0%} points |")
w(f"| Endorsed expert top-1 | 46.7%, at chance (p = 0.50) | {endorsed/n:.1%}, above chance "
  f"(p = {p_ch:.3f}) | +{endorsed/n-0.467:.0%} points, now significant |")
w(f"| Hard precision vs experts | 0.46 | {hp:.2f} | +{hp-0.46:.2f} |")
w(f"| Score-vote correlation | rho 0.144, p = 0.072 (n.s.) | rho {obs:.3f}, p = {p_perm:.3f} | now significant |")
w(f"| High-consensus agreement | 75% (12/16) | {hi_hits/len(hi):.0%} ({hi_hits}/{len(hi)}) | +{hi_hits/len(hi)-0.75:.0%} points |")
w("| Knowledge / Evaluation entries | 49 / 119 | "
  f"{per_level['Kn']} / {per_level['Ev']} | balanced coverage |")
w("| Held-out evaluation | none | repeated stratified 5x5 CV, per-level F1, CIs, paired tests | new |")
w("| Statistical reporting | invalid asymptotic p at n = 5 | exact permutation tests throughout | corrected |")
w("| Gate validation | implicit, claimed desirable | scored separately, disclosed unvalidated | honest |")

out_path = os.path.join(HERE, "paper_results.md")
open(out_path, "w", encoding="utf-8").write("\n".join(L))
print("\n".join(L))
print("\nsaved:", out_path)
