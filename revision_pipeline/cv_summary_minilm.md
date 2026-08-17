# CV summary (minilm, 381 core verbs, repeated 5x5 stratified CV, winner percentile P50)

| Model | macro AUC | macro F1 (mean over repeats) | 95% CI (seed 13 repeat) | micro F1 | subset acc |
|---|---|---|---|---|---|
| SGD-hinge-1e4 | 0.644 | 0.437 | [0.422, 0.482] | 0.435 | 0.065 |
| SGD-log-1e5 | 0.649 | 0.437 | [0.393, 0.457] | 0.432 | 0.077 |
| SGD-hinge-1e5 | 0.634 | 0.437 | [0.405, 0.464] | 0.433 | 0.062 |
| LinearSVC-C1 | 0.674 | 0.460 | [0.424, 0.488] | 0.456 | 0.087 |
| LogReg-C1 | 0.698 | 0.489 | [0.459, 0.529] | 0.486 | 0.113 |
| Ridge-a1 | 0.687 | 0.473 | [0.432, 0.499] | 0.468 | 0.100 |
| PassiveAggressive-C1 | 0.647 | 0.436 | [0.389, 0.455] | 0.431 | 0.075 |
| kNN-21 | 0.703 | 0.481 | [0.444, 0.511] | 0.485 | 0.110 |
| kNN-11 | 0.701 | 0.478 | [0.462, 0.525] | 0.483 | 0.111 |
| NearestCentroid | 0.702 | 0.490 | [0.461, 0.528] | 0.490 | 0.117 |
| majority-level | nan | 0.090 | [0.081, 0.098] | 0.283 | 0.150 |
| freq-random | nan | 0.271 | [0.243, 0.311] | 0.297 | 0.043 |

Per-level F1 (winner SGD-hinge-1e4, mean over 5 repeats):
| Kn | Cm | Ap | An | Sn | Ev |
|---|---|---|---|---|---|
| 0.502 | 0.363 | 0.458 | 0.441 | 0.439 | 0.418 |

Paired exact McNemar (winner vs competitor, per verb-level decision, seed-13 repeat):
- vs SGD-log-1e5: winner-only correct 94, competitor-only correct 100, exact p = 0.7197
- vs SGD-hinge-1e5: winner-only correct 155, competitor-only correct 132, exact p = 0.194
- vs LinearSVC-C1: winner-only correct 84, competitor-only correct 161, exact p = 9.875e-07
- vs LogReg-C1: winner-only correct 131, competitor-only correct 306, exact p = 3.21e-17
- vs Ridge-a1: winner-only correct 100, competitor-only correct 217, exact p = 4.402e-11
- vs PassiveAggressive-C1: winner-only correct 83, competitor-only correct 116, exact p = 0.02306
- vs kNN-21: winner-only correct 227, competitor-only correct 373, exact p = 2.711e-09
- vs kNN-11: winner-only correct 218, competitor-only correct 382, exact p = 2.149e-11
- vs NearestCentroid: winner-only correct 179, competitor-only correct 355, exact p = 2.134e-14
- vs majority-level: winner-only correct 387, competitor-only correct 497, exact p = 0.0002422
- vs freq-random: winner-only correct 498, competitor-only correct 472, exact p = 0.4222