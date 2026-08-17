"""Normalization pass over the candidate verb pool (plan item 22).

Rules, in order, each recorded in the audit trail:
1. WordNet verb lemmatization (surface forms -> lemma), lowercase.
2. British spelling merged into the American form ONLY when doing so resolves a
   real duplicate (the American form exists in the core list or the candidate
   pool) or the American form has a WordNet verb synset and the British one is
   a pure -ise/-yse variant. Never invents forms.
3. Drop pure auxiliaries: be, have, do.
4. Drop candidates whose normalized lemma is a core verb (leakage removal).
5. Dedupe: multiple candidates collapsing to one lemma keep a single entry.

Outputs: cleaned_candidates.txt, normalization_audit.csv, and a printed summary.
"""
import csv
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

REPO = _REPO
OUTDIR = _HERE

wnl = WordNetLemmatizer()
AUX = {"be", "have", "do"}

core = set(open(REPO + r"\results\BTverblist_core.txt", encoding="utf-8").read().split())
with open(REPO + r"\results\Verb List Classified by Model.csv", encoding="utf-8") as f:
    candidates = [row["Verb"].strip().lower() for row in csv.DictReader(f)]
assert len(candidates) == 1824

def has_verb_synset(w):
    return bool(wn.synsets(w, pos="v"))

def american_variant(w, pool):
    """Return the American form if mapping is safe, else None."""
    subs = []
    if w.endswith("ise"):
        subs.append(w[:-3] + "ize")
    if "yse" in w:
        subs.append(w.replace("yse", "yze"))
    if "ise" in w and not w.endswith("ise"):
        subs.append(w.replace("ise", "ize"))
    for am in subs:
        if am == w:
            continue
        # Only merge when it resolves a real duplicate; never invent variants.
        if am in core or am in pool:
            return am
    return None

pool = set(candidates)
audit = []
cleaned = {}
for v in candidates:
    lemma = wnl.lemmatize(v, "v").lower()
    action = "kept" if lemma == v else "lemmatized"
    am = american_variant(lemma, pool)
    if am:
        lemma, action = am, action + "+spelling"
    if lemma in AUX:
        audit.append((v, lemma, "dropped-auxiliary", ""))
        continue
    if lemma in core:
        audit.append((v, lemma, "dropped-core-leak", lemma))
        continue
    if lemma in cleaned:
        audit.append((v, lemma, "merged-duplicate", cleaned[lemma]))
        continue
    cleaned[lemma] = v
    audit.append((v, lemma, action, ""))

with open(OUTDIR + r"\cleaned_candidates.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(sorted(cleaned)))
with open(OUTDIR + r"\normalization_audit.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["original", "normalized_lemma", "action", "merged_into_or_core"])
    w.writerows(audit)

from collections import Counter
c = Counter(a[2] for a in audit)
print("input candidates:", len(candidates))
for k, v in sorted(c.items()):
    print(f"  {k}: {v}")
print("cleaned candidate lemmas:", len(cleaned))
changed = [a for a in audit if a[2] not in ("kept",)]
print("sample changes:", changed[:12])
