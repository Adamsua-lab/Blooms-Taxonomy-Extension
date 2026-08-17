"""SCS weight-perturbation sensitivity analysis (plan item 11, R1.7).

Re-aggregates SCS' = wD*D - wM*M + wT*T + wC*C - wE*E from the per-candidate
component statistics saved by retrain.py, re-applies the stored band penalty
unchanged, and reports the winner's rank under single-weight perturbations
(+-25%, +-50%) and 200 joint random perturbations. No retraining involved.
"""
import argparse, os
import numpy as np
import pandas as pd

PIPE = os.path.dirname(os.path.abspath(__file__))
BASE = {"wD": 1.0, "wM": 1.0, "wT": 0.8, "wC": 0.3, "wE": 0.05}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--components", required=True)
    ap.add_argument("--out", default=os.path.join(PIPE, "scs_sensitivity.md"))
    args = ap.parse_args()

    import json as _json
    wj = _json.load(open(os.path.join(os.path.dirname(args.components), "winner_and_constants.json"), encoding="utf-8"))
    targets = wj["constants"]["targets"]
    ACC_BAND = tuple(targets["ACC_BAND"]); ENAC_BAND = tuple(targets["ENAC_BAND"])
    PEN_ACC = float(targets.get("PEN_ACC", 5.0)); PEN_ENAC = float(targets.get("PEN_ENAC", 2.5))

    df = pd.read_csv(args.components)
    ev = df[df["PassedMinAccept"] != False].copy()
    cols = ["Eval_D", "Eval_M", "Eval_T", "Eval_C", "Eval_E", "SCS_eval",
            "Eval_acceptance", "Eval_en_over_ac"]
    for c in cols:
        ev[c] = pd.to_numeric(ev[c], errors="coerce")
    ev = ev.dropna(subset=cols)

    # Sanity: verify the stored SCS matches the documented formula
    recomputed = (BASE["wD"]*ev["Eval_D"] - BASE["wM"]*ev["Eval_M"] + BASE["wT"]*ev["Eval_T"]
                  + BASE["wC"]*ev["Eval_C"] - BASE["wE"]*ev["Eval_E"])
    err = float((recomputed - ev["SCS_eval"]).abs().max())

    # Band penalty exactly as the pipeline's _penalty_from_out:
    # squared distance to the nearest band edge times PEN, for acceptance and en/ac.
    def band_pen(val, band, pen):
        lo, hi = band
        d = np.where(val < lo, lo - val, np.where(val > hi, val - hi, 0.0))
        return pen * d ** 2
    ev["penalty"] = (band_pen(ev["Eval_acceptance"], ACC_BAND, PEN_ACC)
                     + band_pen(ev["Eval_en_over_ac"], ENAC_BAND, PEN_ENAC))

    TIE = 0.02  # SCS_TIE_EPS, from the notebook

    def ranking(w):
        scs = (w["wD"]*ev["Eval_D"] - w["wM"]*ev["Eval_M"] + w["wT"]*ev["Eval_T"]
               + w["wC"]*ev["Eval_C"] - w["wE"]*ev["Eval_E"])
        e2 = ev.assign(scs=scs, obj=scs - ev["penalty"])
        picks = []
        for model, g in e2.groupby("Model"):
            if model == "Legacy kNN":
                # Legacy kNN: one family-wide pick by max SCS, ties (within eps) by acceptance.
                top = g["scs"].max()
                near = g[g["scs"] >= top - TIE]
                sel = near.sort_values(["scs", "Eval_acceptance"], ascending=False).iloc[0]
                picks.append(sel)
                continue
            # OvR: shortlist WITHIN each hyperparameter group (eps of top SCS, then max
            # obj/acceptance); across groups of a family, pure max obj.
            group_picks = []
            for _, gp in g.groupby("Params"):
                top = gp["scs"].max()
                near = gp[gp["scs"] >= top - TIE]
                group_picks.append(near.sort_values(["obj", "Eval_acceptance"], ascending=False).iloc[0])
            picks.append(pd.DataFrame(group_picks).sort_values("obj", ascending=False).iloc[0])
        best = pd.DataFrame(picks)
        return best.sort_values("obj", ascending=False).reset_index(drop=True)

    base_rank = ranking(BASE)
    base_winner = base_rank.loc[0, "Model"]
    lines = ["# SCS sensitivity analysis", "",
             f"Formula check: max |recomputed - stored SCS| = {err:.2e}",
             f"Baseline weights {BASE}; baseline winner: {base_winner}",
             f"Baseline top 3: " + ", ".join(f"{r.Model} ({r.obj:.3f})" for r in base_rank.head(3).itertuples()), "",
             "| Perturbation | Winner | Baseline winner's rank |", "|---|---|---|"]

    ranks = []
    for key in BASE:
        for f in (0.5, 0.75, 1.25, 1.5):
            w = dict(BASE); w[key] = BASE[key] * f
            r = ranking(w)
            rk = int(r.index[r["Model"] == base_winner][0]) + 1
            ranks.append(rk)
            lines.append(f"| {key} x{f} | {r.loc[0,'Model']} | {rk} |")

    rng = np.random.default_rng(13)
    joint = []
    from collections import Counter
    first_counts, top2_counts, top3_counts = Counter(), Counter(), Counter()
    for _ in range(200):
        w = {k: v * rng.uniform(0.5, 1.5) for k, v in BASE.items()}
        r = ranking(w)
        joint.append(int(r.index[r["Model"] == base_winner][0]) + 1)
        first_counts[r.loc[0, "Model"]] += 1
        for m_ in r.head(2)["Model"]:
            top2_counts[m_] += 1
        for m_ in r.head(3)["Model"]:
            top3_counts[m_] += 1

    lines += ["", f"Single-weight perturbations: winner rank 1 in {sum(1 for x in ranks if x==1)}/{len(ranks)}, rank <=2 in {sum(1 for x in ranks if x<=2)}/{len(ranks)}",
              f"Joint random perturbations (200 draws, all weights x uniform[0.5,1.5]): rank 1 in {sum(1 for x in joint if x==1)}/200, rank <=2 in {sum(1 for x in joint if x<=2)}/200",
              "", "Who wins across the 200 joint draws (top-group stability):",
              "| Model | rank 1 | in top 2 | in top 3 |", "|---|---|---|---|"]
    for m_, c in first_counts.most_common():
        lines.append(f"| {m_} | {c} | {top2_counts[m_]} | {top3_counts[m_]} |")
    for m_ in [k for k in top3_counts if k not in first_counts]:
        lines.append(f"| {m_} | 0 | {top2_counts.get(m_, 0)} | {top3_counts[m_]} |")
    open(args.out, "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines[:6]))
    print(lines[-2]); print(lines[-1])
    print("saved:", args.out)

if __name__ == "__main__":
    main()
