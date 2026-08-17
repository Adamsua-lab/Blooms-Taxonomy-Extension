# SCS sensitivity analysis

Formula check: max |recomputed - stored SCS| = 4.44e-16
Baseline weights {'wD': 1.0, 'wM': 1.0, 'wT': 0.8, 'wC': 0.3, 'wE': 0.05}; baseline winner: SGDClassifier
Baseline top 3: SGDClassifier (0.945), KNeighborsClassifier (0.908), PassiveAggressive (0.868)

| Perturbation | Winner | Baseline winner's rank |
|---|---|---|
| wD x0.5 | SGDClassifier | 1 |
| wD x0.75 | SGDClassifier | 1 |
| wD x1.25 | SGDClassifier | 1 |
| wD x1.5 | SGDClassifier | 1 |
| wM x0.5 | SGDClassifier | 1 |
| wM x0.75 | SGDClassifier | 1 |
| wM x1.25 | SGDClassifier | 1 |
| wM x1.5 | SGDClassifier | 1 |
| wT x0.5 | SGDClassifier | 1 |
| wT x0.75 | SGDClassifier | 1 |
| wT x1.25 | SGDClassifier | 1 |
| wT x1.5 | SGDClassifier | 1 |
| wC x0.5 | SGDClassifier | 1 |
| wC x0.75 | SGDClassifier | 1 |
| wC x1.25 | SGDClassifier | 1 |
| wC x1.5 | SGDClassifier | 1 |
| wE x0.5 | SGDClassifier | 1 |
| wE x0.75 | SGDClassifier | 1 |
| wE x1.25 | SGDClassifier | 1 |
| wE x1.5 | SGDClassifier | 1 |

Single-weight perturbations: winner rank 1 in 20/20, rank <=2 in 20/20
Joint random perturbations (200 draws, all weights x uniform[0.5,1.5]): rank 1 in 200/200, rank <=2 in 200/200

Who wins across the 200 joint draws (top-group stability):
| Model | rank 1 | in top 2 | in top 3 |
|---|---|---|---|
| SGDClassifier | 200 | 200 | 200 |
| PassiveAggressive | 0 | 176 | 197 |
| KNeighborsClassifier | 0 | 23 | 147 |
| LinearSVC | 0 | 0 | 49 |
| NearestCentroid | 0 | 1 | 4 |
| RidgeClassifier | 0 | 0 | 3 |