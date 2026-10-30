# Témoin — Evaluation results

Run at 2026-10-30 18:03 UTC with `meta-models/muse-spark-open-1026` 
(constat v2, compare v1, agent v2).

Command: `python -m eval.run_eval`

## Headline

- Pairs: 20 (0 errors)
- Accuracy: **18/20 (90%)**
- False accusations: **0 (rate 0% over same-object pairs)** — goal: 0 %
- Fraud confirmed: 1, cautious: 5, missed: 0
- Model raw verdicts naming a different object: 1

## Confusion matrix (expected × predicted)

| expected \ predicted | CONFORME | ENDOMMAGÉ | À VÉRIFIER | NON CONFORME |
|---|---|---|---|---|
| CONFORME | 9 | 0 | 1 | 0 |
| ENDOMMAGÉ | 0 | 0 | 0 | 0 |
| À VÉRIFIER | 0 | 0 | 9 | 1 |
| NON CONFORME | 0 | 0 | 0 | 0 |

## Pairs

| pair | relation | expected | verdict | model | conf | time |
|---|---|---|---|---|---|---|
| same-01 | same | CONFORME | À VÉRIFIER | À VÉRIFIER | 62 | 0s |
| same-02 | same | CONFORME | CONFORME | CONFORME | 96 | 0s |
| same-03 | same | CONFORME | CONFORME | CONFORME | 98 | 0s |
| same-04 | same | CONFORME | CONFORME | CONFORME | 93 | 0s |
| same-05 | same | CONFORME | CONFORME | CONFORME | 93 | 0s |
| same-06 | same | CONFORME | CONFORME | CONFORME | 92 | 0s |
| same-07 | same | CONFORME | CONFORME | CONFORME | 98 | 0s |
| same-08 | same | CONFORME | CONFORME | CONFORME | 98 | 0s |
| same-09 | same | CONFORME | CONFORME | CONFORME | 93 | 0s |
| same-10 | same | CONFORME | CONFORME | CONFORME | 96 | 0s |
| same-11 | same | À VÉRIFIER | À VÉRIFIER | CONFORME | 86 | 0s |
| same-12 | same | À VÉRIFIER | À VÉRIFIER | CONFORME | 87 | 0s |
| same-13 | same | À VÉRIFIER | À VÉRIFIER | À VÉRIFIER | 62 | 0s |
| same-14 | same | À VÉRIFIER | À VÉRIFIER | CONFORME | 93 | 0s |
| diff-01 | different | À VÉRIFIER | NON CONFORME | NON CONFORME | 98 | 0s |
| diff-02 | different | À VÉRIFIER | À VÉRIFIER | À VÉRIFIER | 95 | 0s |
| diff-03 | different | À VÉRIFIER | À VÉRIFIER | À VÉRIFIER | 92 | 0s |
| diff-04 | different | À VÉRIFIER | À VÉRIFIER | À VÉRIFIER | 95 | 0s |
| diff-05 | different | À VÉRIFIER | À VÉRIFIER | À VÉRIFIER | 92 | 0s |
| diff-06 | different | À VÉRIFIER | À VÉRIFIER | À VÉRIFIER | 85 | 0s |

## Known gaps

- Damaged-between-photos (ENDOMMAGÉ): 0 pairs — needs longitudinal photos of the same object before/after damage.
- Same-model-different-unit: 0 pairs — needs two physical units of one model.
- Add pairs by dropping two photos under `eval/pairs/<id>/` and one row in `eval/labels.csv`, then rerun.

