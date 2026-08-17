"""Merge published Bloom verb lists into the training core (plan item 17).

Sources:
  core_published.csv   Sharunova et al. 2019 Table 9 appendix, transcribed verbatim
                       (379 verbs, 580 assignments; verified by dual independent passes)
  stanny_extracted.csv Stanny 2016, Table 1 (176 unique words, 263 pairs, f = lists nominating)
  newton_extracted.csv Newton, Da Silva and Peters 2020, Table 1 master list (original taxonomy names)

Rules (all documented for the methods section):
  1. Crosswalk to original-taxonomy codes: Knowledge/Remember->Kn, Understand/Comprehension->Cm,
     Apply/Application->Ap, Analyze/Analysis->An, Evaluate/Evaluation->Ev, Create/Synthesis->Sn.
  2. Only single-token entries are eligible (no spaces or slashes); hyphens allowed.
     The pipeline embeds isolated verb lemmas, so phrases are excluded and listed in the audit.
  3. Frequencies are combined conservatively with max() across consolidations, because the
     consulted source lists overlap heavily; a sum would double-count the same underlying lists.
     Newton's master list carries no counts and contributes freq 1 where it is the only source.
  4. Labels are unioned (multi-label by design).
  5. Any NEW core verb is removed from the candidate pool (it is training data now).

Outputs: core_merged.csv, merge_audit.csv, candidates_final.txt, printed summary.
"""
import csv
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
from collections import defaultdict

DIR = _HERE
CROSS = {"Knowledge": "Kn", "Remember": "Kn", "Understand": "Cm", "Comprehension": "Cm",
         "Apply": "Ap", "Application": "Ap", "Analyze": "An", "Analysis": "An",
         "Evaluate": "Ev", "Evaluation": "Ev", "Create": "Sn", "Synthesis": "Sn"}
LEVELS = ["Kn", "Cm", "Ap", "An", "Sn", "Ev"]

freq = defaultdict(int)          # (verb, level) -> combined freq
sources = defaultdict(set)      # (verb, level) -> source names
excluded = []

with open(DIR + r"\core_published.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        key = (r["verb"], r["level"])
        freq[key] = max(freq[key], int(float(r["freq"])))
        sources[key].add("sharunova")

for src_name, fn in [("stanny2016", "stanny_extracted.csv"), ("newton2020", "newton_extracted.csv")]:
    with open(DIR + "\\" + fn, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            verb = r["verb"].strip().lower()
            if " " in verb or "/" in verb:
                excluded.append((src_name, verb, "multi-word or slash entry"))
                continue
            f_val = int(float(r["freq"])) if r.get("freq") not in (None, "",) else 1
            for lev in r["levels"].split("|"):
                code = CROSS.get(lev.strip())
                if not code:
                    excluded.append((src_name, verb, f"unmapped level {lev}"))
                    continue
                key = (verb, code)
                freq[key] = max(freq[key], f_val)
                sources[key].add(src_name)

core_verbs_old = {v for (v, l) in freq if "sharunova" in sources[(v, l)]}
all_verbs = sorted({v for (v, l) in freq})
new_verbs = sorted(set(all_verbs) - core_verbs_old)
new_assignments_existing = [(v, l) for (v, l) in freq
                            if v in core_verbs_old and "sharunova" not in sources[(v, l)]]

with open(DIR + r"\core_merged.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["verb", "level", "freq", "sources"])
    for (v, l) in sorted(freq, key=lambda k: (k[0], LEVELS.index(k[1]))):
        w.writerow([v, l, freq[(v, l)], "+".join(sorted(sources[(v, l)]))])

with open(DIR + r"\merge_audit.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["source", "entry", "reason_excluded"])
    w.writerows(excluded)

# Remove new core verbs from the candidate pool
cands = open(DIR + r"\cleaned_candidates.txt", encoding="utf-8").read().split()
promoted = sorted(set(cands) & set(new_verbs))
final_cands = [c for c in cands if c not in set(new_verbs)]
with open(DIR + r"\candidates_final.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(final_cands))

per_level = {l: sum(1 for (v, ll) in freq if ll == l) for l in LEVELS}
print(f"merged core: {len(all_verbs)} verbs, {len(freq)} assignments (was 358 / 559)")
print(f"per level: {per_level}")
print(f"new verbs added: {len(new_verbs)}")
print(f"new level-assignments on existing core verbs: {len(new_assignments_existing)}")
print(f"excluded entries: {len(excluded)} -> {[e[1] for e in excluded]}")
print(f"candidates promoted to core (removed from pool): {len(promoted)} -> {promoted[:15]}")
print(f"final candidate pool: {len(final_cands)} (was {len(cands)})")

# expert-sample implications (load the real rated verbs, clean lemma form)
import pandas as pd
REPO = _REPO
sample30 = [str(v).strip().lower() for v in
            pd.read_excel(REPO + r"\results\Final File(with model & Expert Classification).xlsx")["Verb_clean"]]
hit = [v for v in sample30 if v in set(new_verbs)]
print(f"expert-sample verbs newly promoted to core: {hit if hit else 'none'}")
