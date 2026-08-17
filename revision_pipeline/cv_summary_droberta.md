# CV summary (droberta, 381 core verbs, repeated 5x5 stratified CV, winner percentile P50)

| Model | macro AUC | macro F1 (mean over repeats) | 95% CI (seed 13 repeat) | micro F1 | subset acc |
|---|---|---|---|---|---|
| SGD-hinge-1e4 | 0.649 | 0.427 | [0.399, 0.458] | 0.432 | 0.064 |
| SGD-log-1e5 | 0.658 | 0.439 | [0.408, 0.468] | 0.442 | 0.066 |
| SGD-hinge-1e5 | 0.640 | 0.431 | [0.400, 0.459] | 0.438 | 0.072 |
| LinearSVC-C1 | 0.669 | 0.455 | [0.425, 0.487] | 0.459 | 0.078 |
| LogReg-C1 | 0.683 | 0.469 | [0.438, 0.501] | 0.472 | 0.103 |
| Ridge-a1 | 0.677 | 0.465 | [0.437, 0.498] | 0.468 | 0.097 |
| PassiveAggressive-C1 | 0.655 | 0.432 | [0.394, 0.457] | 0.436 | 0.072 |
| kNN-21 | 0.660 | 0.438 | [0.415, 0.476] | 0.443 | 0.081 |
| kNN-11 | 0.664 | 0.442 | [0.401, 0.467] | 0.446 | 0.077 |
| NearestCentroid | 0.681 | 0.472 | [0.440, 0.505] | 0.476 | 0.118 |
| majority-level | nan | 0.090 | [0.081, 0.098] | 0.283 | 0.150 |
| freq-random | nan | 0.271 | [0.243, 0.311] | 0.297 | 0.043 |

Per-level F1 (winner SGD-hinge-1e4, mean over 5 repeats):
| Kn | Cm | Ap | An | Sn | Ev |
|---|---|---|---|---|---|
| 0.426 | 0.378 | 0.437 | 0.460 | 0.497 | 0.362 |

Paired exact McNemar (winner vs competitor, per verb-level decision, seed-13 repeat):
- vs SGD-log-1e5: winner-only correct 71, competitor-only correct 134, exact p = 1.285e-05
- vs SGD-hinge-1e5: winner-only correct 107, competitor-only correct 164, exact p = 0.0006425
- vs LinearSVC-C1: winner-only correct 96, competitor-only correct 190, exact p = 2.89e-08
- vs LogReg-C1: winner-only correct 137, competitor-only correct 293, exact p = 4.086e-14
- vs Ridge-a1: winner-only correct 117, competitor-only correct 247, exact p = 8.257e-12
- vs PassiveAggressive-C1: winner-only correct 80, competitor-only correct 159, exact p = 3.577e-07
- vs kNN-21: winner-only correct 287, competitor-only correct 361, exact p = 0.0041
- vs kNN-11: winner-only correct 282, competitor-only correct 377, exact p = 0.0002449
- vs NearestCentroid: winner-only correct 193, competitor-only correct 342, exact p = 1.198e-10
- vs majority-level: winner-only correct 385, competitor-only correct 526, exact p = 3.365e-06
- vs freq-random: winner-only correct 498, competitor-only correct 503, exact p = 0.8994