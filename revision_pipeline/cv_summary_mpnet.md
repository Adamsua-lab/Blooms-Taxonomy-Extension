# CV summary (mpnet, 381 core verbs, repeated 5x5 stratified CV, winner percentile P50)

| Model | macro AUC | macro F1 (mean over repeats) | 95% CI (seed 13 repeat) | micro F1 | subset acc |
|---|---|---|---|---|---|
| SGD-hinge-1e4 | 0.666 | 0.447 | [0.412, 0.481] | 0.450 | 0.093 |
| SGD-log-1e5 | 0.674 | 0.459 | [0.417, 0.484] | 0.460 | 0.097 |
| SGD-hinge-1e5 | 0.665 | 0.455 | [0.430, 0.495] | 0.460 | 0.097 |
| LinearSVC-C1 | 0.701 | 0.484 | [0.429, 0.501] | 0.485 | 0.128 |
| LogReg-C1 | 0.720 | 0.508 | [0.476, 0.548] | 0.508 | 0.156 |
| Ridge-a1 | 0.713 | 0.501 | [0.464, 0.533] | 0.500 | 0.135 |
| PassiveAggressive-C1 | 0.672 | 0.457 | [0.410, 0.475] | 0.458 | 0.098 |
| kNN-21 | 0.713 | 0.492 | [0.470, 0.537] | 0.496 | 0.142 |
| kNN-11 | 0.706 | 0.473 | [0.454, 0.519] | 0.478 | 0.120 |
| NearestCentroid | 0.716 | 0.491 | [0.474, 0.541] | 0.494 | 0.148 |
| GaussianNB | 0.711 | 0.501 | [0.481, 0.552] | 0.502 | 0.133 |
| LDA | 0.705 | 0.480 | [0.459, 0.525] | 0.484 | 0.127 |
| QDA | 0.677 | 0.459 | [0.439, 0.504] | 0.464 | 0.104 |
| CalibratedSVC | 0.725 | 0.512 | [0.481, 0.552] | 0.512 | 0.159 |
| majority-level | nan | 0.090 | [0.081, 0.098] | 0.283 | 0.150 |
| freq-random | nan | 0.271 | [0.243, 0.311] | 0.297 | 0.043 |

Per-level F1 (winner SGD-hinge-1e4, mean over 5 repeats):
| Kn | Cm | Ap | An | Sn | Ev |
|---|---|---|---|---|---|
| 0.452 | 0.434 | 0.455 | 0.479 | 0.466 | 0.393 |

Paired exact McNemar (winner vs competitor, per verb-level decision, seed-13 repeat):
- vs SGD-log-1e5: winner-only correct 92, competitor-only correct 94, exact p = 0.9416
- vs SGD-hinge-1e5: winner-only correct 161, competitor-only correct 126, exact p = 0.04457
- vs LinearSVC-C1: winner-only correct 106, competitor-only correct 151, exact p = 0.00595
- vs LogReg-C1: winner-only correct 157, competitor-only correct 270, exact p = 5.013e-08
- vs Ridge-a1: winner-only correct 129, competitor-only correct 221, exact p = 1.003e-06
- vs PassiveAggressive-C1: winner-only correct 103, competitor-only correct 89, exact p = 0.3482
- vs kNN-21: winner-only correct 228, competitor-only correct 321, exact p = 8.315e-05
- vs kNN-11: winner-only correct 249, competitor-only correct 305, exact p = 0.01937
- vs NearestCentroid: winner-only correct 203, competitor-only correct 313, exact p = 1.466e-06
- vs GaussianNB: winner-only correct 209, competitor-only correct 318, exact p = 2.352e-06
- vs LDA: winner-only correct 282, competitor-only correct 348, exact p = 0.009552
- vs QDA: winner-only correct 317, competitor-only correct 322, exact p = 0.8743
- vs CalibratedSVC: winner-only correct 168, competitor-only correct 295, exact p = 3.821e-09
- vs majority-level: winner-only correct 397, competitor-only correct 401, exact p = 0.9154
- vs freq-random: winner-only correct 523, competitor-only correct 391, exact p = 1.424e-05