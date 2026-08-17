# Superseded material

Everything in this folder comes from the original submission and has been replaced. It is
kept so the earlier state stays inspectable, not because it is still in use. If you want
the current versions, use the files named on the right.

| Here | Superseded by |
| --- | --- |
| `original-submission-model/` (SGDClassifier P60, saved 2025-10-12) | `production_model/` (kNN, k = 21, P50) |
| `BTverblist_new.txt` (1,824 raw candidate verbs) | `revision_pipeline/candidates_final.txt` (1,650 after normalization) |
| `ExpertSample_single (Model Classified Sample).csv` | `results/ExpertSample_single.csv` |
| `Final File(with model & Expert Classification).xlsx` | `results/Verb List Classified by Model.csv` |

Why they were replaced:

- The training core was audited against its published source and enriched from two further
  published consolidations, so every model trained before that audit sits on a different
  core (381 verbs and 607 assignments now, against 358 and 559 before).
- The candidate pool was normalized, which removed core-verb variants, within-pool
  duplicates and auxiliaries. `revision_pipeline/normalization_audit.csv` records what
  happened to each of the 1,824 raw candidates.
- Model selection changed from the structural score alone to a two-stage rule that also
  uses cross-validated generalization and held-out end-to-end accuracy, which is what
  moved the production model from SGDClassifier to k-nearest neighbours.

Do not load the model in `original-submission-model/` expecting the published results. It
predates all of the above.
