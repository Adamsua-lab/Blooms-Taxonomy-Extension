# Revision pipeline (major revision, Discover Education)

Standalone scripts that produced every number in the revised manuscript. They reproduce
the BT1-BT4 notebook pipeline faithfully (all constants documented in retrain.py's header)
and add the analyses requested by the reviewers.

## Reproduction order

```
python normalize.py                      # 1824 raw candidates -> cleaned_candidates.txt (1652) + audit
python merge_core.py                     # audited core + Stanny 2016 + Newton 2020 -> core_merged.csv (381 verbs) + candidates_final.txt (1650)
python retrain.py --core-csv core_merged.csv --candidates candidates_final.txt --outdir <out>
python cv_metrics.py --winner-json <out>/results/winner_and_constants.json            # add --encoder/--tag for the ablation
python scs_sensitivity.py --components <out>/results/structural_scan_components.csv
python expert_recompute.py               # expert agreement vs the regenerated bank
```

Requirements: see ../requirements.txt (pinned to the versions this was run with).

## Files

- normalize.py, normalization_audit.csv: candidate-pool cleanup rules and audit trail.
- merge_core.py, merge_audit.csv: published-list merge with the 1956-2001 crosswalk;
  stanny_extracted.csv and newton_extracted.csv are verbatim transcriptions of
  Stanny (2016) Table 1 and Newton, Da Silva and Peters (2020) Table 1.
- core_merged.csv: the enriched 381-verb, 607-assignment training core.
- retrain.py: faithful standalone Algorithm 1 (embedding, model zoo, OOF-CDF calibration,
  percentile threshold scan, SCS selection, top-2 collapse delta 0.060, soft tier 0.85).
  Deviations from the notebooks are listed in the header block. Embeddings are served
  from all_embeddings.npz (all-mpnet-base-v2, normalized) to avoid per-file I/O stalls.
- cv_metrics.py, cv_results_*.csv, cv_summary_*.md: repeated stratified multi-label CV
  mirroring the production decision rule, three encoders.
- scs_sensitivity.py, scs_sensitivity.md: SCS weight-perturbation analysis.
- expert_recompute.py: agreement of the regenerated bank with the 8x30 expert votes.

Note on the notebooks: BT3's committed state is missing the cell that defined the
production post-processing constants (the top-2 margin delta); retrain.py restores the
published value (0.060) and documents it. BT2 reads its corpus from a `data/` folder
that does not exist in the repo; the corpus manifest replaces it in this revision.
