# Paper-Ready Results (final pipeline, corrected core)

## Table M1. Model comparison and joint selection

| Family (best config) | Structural objective | Acceptance | CV macro AUC | CV macro F1 | Maximin |
|---|---|---|---|---|---|
| **KNeighborsClassifier (P50)** | 0.908 | 0.861 | 0.713 | 0.492 | **0.554** |
| LinearSVC (P45) | 0.898 | 0.856 | 0.701 | 0.484 | 0.496 |
| NearestCentroid (P40) | 0.860 | 0.825 | 0.716 | 0.491 | 0.268 |
| LogisticRegression (P45) | 0.857 | 0.764 | 0.720 | 0.508 | 0.252 |
| PassiveAggressive (P50) | 0.920 | 0.833 | 0.672 | 0.457 | 0.112 |
| RidgeClassifier (P40) | 0.815 | 0.857 | 0.713 | 0.501 | 0.000 |
| SGDClassifier (P55) | 0.983 | 0.804 | 0.666 | 0.447 | 0.000 |

Criteria independence: Spearman(structural objective, CV AUC) = -0.71 (p = 0.07); at matched acceptance the association is also non-significant, so the two criteria are treated as complementary and the winner maximizes the weaker normalized score (maximin).

Selection robustness: perturbing every SCS weight jointly by a random factor in [0.5, 1.5] (500 draws), the selected family remains the maximin winner in **333/500 = 67%** of draws.

## Table M2. Extension outcomes (delivered bank)

- Candidates after normalization: 1650
- Accepted: 1420 (0.861); unassigned: 230
- Verb-level entries: 1647 (1.16 levels per accepted verb)
- Label cardinality: {1: 1243, 2: 133, 3: 38, 4: 6}
- Entries per level: {'Kn': 167, 'Cm': 86, 'Ap': 445, 'An': 257, 'Sn': 413, 'Ev': 279}

## Table M3. Expert validation (8 experts, 30 stratified verbs)

Of the 30 rated verbs, 29 remain after normalization and 26 are accepted by the final model. Level-assignment metrics on the accepted set; the accept/reject gate is reported separately below.

| Metric | Value | Chance / test |
|---|---|---|
| Strict top-1 (primary equals expert top vote) | 11/26 = 42.3% | |
| Endorsed top-1 (primary in the 2-of-8 endorsed set) | 18/26 = 69.2% | chance 46%, p = 0.0080 |
| Hard-label Jaccard / precision / recall | 0.32 / 0.71 / 0.38 | |
| Score-vote correlation (per-verb Spearman, mean) | rho = 0.171 | permutation p = 0.0144 |
| Agreement, high-consensus verbs (>=4/8 on one level) | 13/14 = 93% | chance 51%, p = 0.0008 |
| Agreement, low-consensus verbs | 5/12 = 42% | |
| Band gap (errors concentrate where experts disagree) | | Fisher exact p = 0.0093 |

**Accept/reject gate (reported separately, unvalidated at this sample size):** correct acceptances 22, correct rejections 1, false acceptances 4, false rejections 2; gate accuracy 23/29 = 79% (Fisher n.s.). The sample was stratified over the original model's accepted domains, so rejections are structurally underrepresented; validating the gate needs a sample that includes rejected verbs by design (stated as future work).

## Table M4. Structural coherence of the delivered bank

Computed on the published (post-collapse) labeling; earlier drafts quoted the pre-collapse
figure, which overstates the trend (0.80 vs 0.60).

| Encoder | Gap-trend Spearman rho | Exact one-tailed p |
|---|---|---|
| all-mpnet-base-v2 (training encoder) | +0.60 | 0.175 |
| all-MiniLM-L6-v2 (independent) | +0.60 | 0.175 |
| all-distilroberta-v1 (independent) | +0.50 | 0.225 |

The trend is directionally positive and reproduces at the same magnitude in encoders that
played no part in training or selection, so the ordinal structure is a property of the
assignments rather than the training geometry; at five gap values it is descriptive rather
than statistically significant, and the paper reports it as such.

## Table M5. Improvement over the submitted manuscript

| Dimension | Submitted | Final | Change |
|---|---|---|---|
| Training core | 358 verbs / 559 assignments, 21 transcription errors vs source | 381 / 607, audited against the published appendix (dual transcription, 0 disagreements) | corrected + enriched |
| Candidate pool | 1,824 raw (duplicates, inflected forms, 69 core leaks, auxiliaries) | 1,650 clean lemmas, full audit trail | leakage removed |
| Selection criterion | SCS only, formula unpublished, selection bug | published formula, bug fixed, joint maximin over structure + CV | transparent |
| Production model | linear SGD (CV AUC 0.666, macro F1 0.447) | kNN-21 P50 (CV AUC 0.713, macro F1 0.492) | +0.047 AUC, +0.045 F1 |
| Strict expert top-1 | 20.0% | 42.3% | +22% points |
| Endorsed expert top-1 | 46.7%, at chance (p = 0.50) | 69.2%, above chance (p = 0.008) | +23% points, now significant |
| Hard precision vs experts | 0.46 | 0.71 | +0.25 |
| Score-vote correlation | rho 0.144, p = 0.072 (n.s.) | rho 0.171, p = 0.014 | now significant |
| High-consensus agreement | 75% (12/16) | 93% (13/14) | +18% points |
| Knowledge / Evaluation entries | 49 / 119 | 167 / 279 | balanced coverage |
| Held-out evaluation | none | repeated stratified 5x5 CV, per-level F1, CIs, paired tests | new |
| Statistical reporting | invalid asymptotic p at n = 5 | exact permutation tests throughout | corrected |
| Gate validation | implicit, claimed desirable | scored separately, disclosed unvalidated | honest |