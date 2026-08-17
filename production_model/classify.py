"""Label new verbs with Bloom cognitive levels using the published production model.

Examples
    python classify.py weld debug summarise
    python classify.py --file verbs.txt --out labelled.csv --soft
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bt4_scorer import BT4Scorer, LEVELS  # noqa: E402

ENCODER = "sentence-transformers/all-mpnet-base-v2"


def embed(verbs):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(ENCODER)
    return np.asarray(model.encode(list(verbs), normalize_embeddings=True), dtype=float)


def label(verbs, soft_alpha=None):
    scorer = BT4Scorer(HERE)
    X = embed(verbs)
    hard, scores = scorer.predict_binary(X)
    soft = scorer.suggest_binary(X, soft_alpha)[0] if soft_alpha else None
    rows = []
    for i, verb in enumerate(verbs):
        accepted = [LEVELS[j] for j in range(len(LEVELS)) if hard[i, j] == 1]
        primary = max(accepted, key=lambda l: scores[i, LEVELS.index(l)]) if accepted else "None"
        row = {
            "verb": verb,
            "primary": primary,
            "accepted": " ".join(accepted) if accepted else "",
        }
        if soft is not None:
            row["soft"] = " ".join(LEVELS[j] for j in range(len(LEVELS)) if soft[i, j] == 1)
        for j, lv in enumerate(LEVELS):
            row[f"score_{lv}"] = round(float(scores[i, j]), 3)
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description="Assign Bloom levels to verbs.")
    ap.add_argument("verbs", nargs="*", help="verbs to label")
    ap.add_argument("--file", help="text file with one verb per line")
    ap.add_argument("--out", help="write results to this CSV instead of the screen")
    ap.add_argument("--soft", action="store_true",
                    help="also report the soft suggestion tier (thresholds scaled by 0.85)")
    ap.add_argument("--alpha", type=float, default=0.85, help="soft tier scale (default 0.85)")
    args = ap.parse_args()

    verbs = list(args.verbs)
    if args.file:
        verbs += [ln.strip() for ln in Path(args.file).read_text(encoding="utf-8").splitlines() if ln.strip()]
    verbs = [v.lower() for v in dict.fromkeys(verbs)]
    if not verbs:
        ap.error("give at least one verb, or use --file")

    rows = label(verbs, args.alpha if args.soft else None)

    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"{len(rows)} verbs written to {args.out}")
        return

    cols = list(rows[0].keys())
    widths = {c: max(len(c), max(len(str(r[c])) for r in rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    for r in rows:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in cols))


if __name__ == "__main__":
    main()
