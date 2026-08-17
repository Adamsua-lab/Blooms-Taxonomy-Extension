# Extending the action verb inventory of Bloom's cognitive domain

Code, data and models behind the paper *Extending the Action Verb Inventory of Bloom's
Cognitive Domain for Engineering Education with Machine Learning* (Talha, Shi and Qureshi).

The problem is simple to state. People write learning outcomes with action verbs, but the
Bloom verb lists in circulation are short, generic and rarely cover the vocabulary a
discipline actually uses. This repository builds a larger, discipline-specific verb bank
by learning from the verbs that already carry expert labels and applying that model to
verbs mined from engineering literature. It delivers 1,420 accepted verbs carrying 1,647
verb-level assignments, plus the 230 candidates the model declined.

## Start here

If you want to **label verbs**, go to [`production_model/`](production_model/). Nothing
else needs to run.

```
pip install -r requirements.txt
python production_model/classify.py weld debug appraise
```

If you want the **finished verb bank**, open
[`results/Verb List Classified by Model.csv`](results/Verb%20List%20Classified%20by%20Model.csv).
Rows with `Accepted = True` are the bank; the rest are the declined candidates, kept so
the accept/reject behaviour stays inspectable.

If you want to **rerun the study**, see Reproduction below.

## What is in here

| Folder | Contents |
| --- | --- |
| `production_model/` | the published model, packaged with a command line tool and usage notes |
| `revision_pipeline/` | the scripts that produced every number in the paper, plus their inputs and audit trails |
| `results/` | the delivered verb bank, model comparisons, structural scans, figures |
| `results/core_structure/` | the core-verb distance tables and figures used in Section 2.1 |
| `BT 1` to `BT 5` notebooks | the narrative walkthrough, executed with outputs |
| `Papers & Books Used for Extraction/` | the source corpus, for transparency (see the notice in that folder) |
| `legacy/` | material from the original submission, kept for reference and clearly superseded |

## The notebooks

They are meant to be read in order, and they drive the scripts in `revision_pipeline/`
rather than duplicating them.

1. **BT 1 - Analysis on core verb**: the labelled core list, its embedding geometry, and
   the centroid and gap analyses reported in Section 2.1.
2. **BT 2 - Data collection**: extracting candidate verbs from the corpus with Stanza and
   filtering them through WordNet.
3. **BT 3 - Model training and classification**: imports `retrain.py` and runs the model
   zoo, calibration, threshold scan and selection.
4. **BT 4 - Validation and reflection**: expert agreement and the structural checks.
5. **BT 5 - Revision analyses**: cross-validation, encoder ablation, held-out evaluation,
   significance tests and sensitivity analyses added during revision.

## Reproduction

Full rerun, in order. Each step writes into the folder you pass it, so nothing is
overwritten by accident.

```
cd revision_pipeline
python normalize.py                 # 1,824 raw candidates -> 1,652 clean lemmas + audit trail
python merge_core.py                # audited core + two published consolidations -> 381 verbs, 607 assignments
python retrain.py --core-csv core_merged.csv --candidates candidates_final.txt --outdir ../results/rerun
python cv_metrics.py --winner-json ../results/rerun/results/winner_and_constants.json
python methodology_upgrades.py      # held-out full-pipeline evaluation
python scs_sensitivity.py --components ../results/rerun/results/structural_scan_components.csv
python expert_recompute.py          # expert agreement against the regenerated bank
python paper_results.py             # the core-structure tables and figures
```

Embeddings are served from `all_embeddings.npz` so a rerun takes minutes rather than
re-encoding every verb. Delete that file if you want to regenerate embeddings from
scratch. Runs are seeded (seed 13); small differences can still appear across BLAS or
scikit-learn builds.

### Where each result comes from

| Paper item | Produced by |
| --- | --- |
| Tables 2 to 4, Figures 1 and 2 (core structure) | `paper_results.py`, outputs in `results/core_structure/` |
| Table 5 (corpus composition) | `BT 2` notebook |
| Table 6 (cross-validation) | `cv_metrics.py` -> `cv_results_mpnet.csv` |
| Table 7 (encoder ablation) | `cv_metrics.py --encoder ...` -> `cv_results_minilm.csv`, `cv_results_droberta.csv` |
| Tables 8a and 8b (model selection) | `retrain.py` -> `winner_and_constants.json`, `model_comparison_table5.csv` |
| Table 9 (extension outcomes) | `retrain.py` -> `Verb List Classified by Model.csv` |
| Table 10 (gap trend across encoders) | `paper_results.py` |
| Table 11 (held-out pipeline) | `methodology_upgrades.py` -> `heldout_results.csv` |
| Expert agreement in Section 3.3.3 | `expert_recompute.py` |
| Appendix C (full verb bank) | `results/Verb List Classified by Model.csv` |

## Method in one paragraph

Verbs are embedded with `all-mpnet-base-v2` and classified by one-vs-rest multi-label
models. Raw scores are mapped to [0, 1] through per-level empirical CDFs fitted
out-of-fold, and a verb is accepted at a level when its calibrated score clears that
level's threshold, set by a percentile scan. Model selection uses two stages that never
touch the expert data: families first have to pass a structural adequacy floor and are
compared on a penalised structural score and cross-validated macro AUC, then the shortlist
is compared again with held-out end-to-end accuracy added. The winner was a k-nearest
neighbour classifier at the 50th percentile. Full detail, including every constant, is in
the paper and in the header of `retrain.py`.

## Data notes

- `revision_pipeline/table9_pass1.csv` and `table9_pass2.csv` are two independent
  transcriptions of the published source table. They are byte-identical, which is the
  evidence behind the claim that the transcriptions agreed on all 580 assignments.
- `revision_pipeline/core_notebook_original.csv` is the superseded working copy, kept so
  the discrepancies we corrected remain checkable.
- `revision_pipeline/normalization_audit.csv` records what happened to every one of the
  1,824 raw candidates, one row each.

## Citing this work

See `CITATION.cff`. Please cite the paper if you use the verb bank or the model.

## Licence

MIT, and it covers the code, the data files and the models produced by this project. It
does **not** cover the third-party publications in `Papers & Books Used for Extraction/`,
which remain under their publishers' terms; see the notice in that folder.
