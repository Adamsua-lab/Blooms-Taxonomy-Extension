# Production model

This folder holds the model that produced the published verb bank, packaged so it can be
used on its own without running the training pipeline.

The model is a one-vs-rest k-nearest-neighbour classifier (k = 21, cosine distance,
distance weighting) trained on 381 core Bloom verbs carrying 607 verb-level assignments,
with per-level thresholds set at the 50th percentile of the calibrated core scores.
Verbs are represented by `all-mpnet-base-v2` sentence embeddings (768 dimensions,
L2 normalised).

## Files

| File | What it is |
| --- | --- |
| `model.pkl` | the fitted one-vs-rest classifier (joblib) |
| `calibration_cdfs.json` | per-level empirical CDFs that map raw scores to [0, 1] |
| `thresholds.json` | per-level acceptance thresholds (P50 of the calibrated core scores) |
| `row_weights.npy` | frequency weights used during training |
| `metadata.json` | model parameters, level order, provenance |
| `bt4_scorer.py` | loader and scoring class used by the pipeline |
| `classify.py` | command line tool for labelling new verbs |

## Quick use

```
pip install -r ../requirements.txt
python classify.py weld debug summarise appraise
```

Output is one line per verb with the accepted levels, the primary level and the
calibrated score for each of the six levels:

```
verb        primary  accepted     Kn     Cm     Ap     An     Sn     Ev
weld        Ap       Ap         0.31   0.52   0.71   0.44   0.63   0.29
```

A verb with no level above its threshold is reported as `None`, which is the model
declining to place it rather than a failure. Add `--soft` to also show the wider
suggestion tier (thresholds scaled by 0.85), which trades precision for recall.

To label a file of verbs, one per line:

```
python classify.py --file my_verbs.txt --out labelled.csv
```

## Using it from Python

```python
from bt4_scorer import BT4Scorer
from sentence_transformers import SentenceTransformer
import numpy as np

enc = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
X = enc.encode(["weld", "debug"], normalize_embeddings=True)

scorer = BT4Scorer(".")
Y, scores = scorer.predict_binary(X)      # hard labels
Ysoft, _ = scorer.suggest_binary(X, 0.85) # soft suggestions
```

`Y` is an n x 6 binary matrix over the levels in `metadata.json` order
(Kn, Cm, Ap, An, Sn, Ev) and `scores` holds the calibrated scores.

## What to expect from it

On held-out core verbs the acceptance gate keeps 92.4 per cent of known Bloom verbs and
78.4 per cent of primary predictions land within one level of the published level. Against
an eight-expert panel the primary level fell inside the expert-endorsed set for 69.2 per
cent of accepted verbs. Verb-to-level mapping depends partly on context, so treat the
output as a ranked prior to be read alongside the full learning outcome, not as a
context-free verdict. The precision side of the accept/reject decision has not been
validated against labelled negatives, so unassigned verbs are best sent to human review.

The model is stored as a pickle, which means it is tied to the scikit-learn version in
`requirements.txt`. If you load it under a different version and see a warning, retrain
with `revision_pipeline/retrain.py` instead of trusting the loaded object.
