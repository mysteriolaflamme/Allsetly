# Témoin — Evaluation results

Run at 2026-10-30 12:37 UTC with `meta-models/muse-spark-open-1026` 
(constat v1, compare v1, agent v2).

Command: `python -m eval.run_eval`

## Headline

- Pairs: 20 (0 errors)
- Accuracy: **19/20 (95%)**
- False accusations: **0 (rate 0% over same-object pairs)** — goal: 0 %
- Fraud confirmed: 0, cautious: 6, missed: 0
- Model raw verdicts naming a different object: 0

## Confusion matrix (expected × predicted)

| expected \ predicted | CONFORME | ENDOMMAGÉ | À VÉRIFIER | NON CONFORME |
|---|---|---|---|---|
| CONFORME | 9 | 1 | 0 | 0 |
| ENDOMMAGÉ | 0 | 0 | 0 | 0 |
| À VÉRIFIER | 0 | 0 | 10 | 0 |
| NON CONFORME | 0 | 0 | 0 | 0 |

## Pairs

| pair | relation | expected | verdict | model | conf | time |
|---|---|---|---|---|---|---|
| same-01 | same | CONFORME | ENDOMMAGÉ | ENDOMMAGÉ | 71 | 0s |
| same-02 | same | CONFORME | CONFORME | CONFORME | 98 | 27s |
| same-03 | same | CONFORME | CONFORME | CONFORME | 99 | 59s |
| same-04 | same | CONFORME | CONFORME | CONFORME | 96 | 67s |
| same-05 | same | CONFORME | CONFORME | CONFORME | 98 | 97s |
| same-06 | same | CONFORME | CONFORME | CONFORME | 97 | 76s |
| same-07 | same | CONFORME | CONFORME | CONFORME | 96 | 86s |
| same-08 | same | CONFORME | CONFORME | CONFORME | 98 | 96s |
| same-09 | same | CONFORME | CONFORME | CONFORME | 92 | 68s |
| same-10 | same | CONFORME | CONFORME | CONFORME | 98 | 39s |
| same-11 | same | À VÉRIFIER | À VÉRIFIER | CONFORME | 92 | 60s |
| same-12 | same | À VÉRIFIER | À VÉRIFIER | ENDOMMAGÉ | 74 | 142s |
| same-13 | same | À VÉRIFIER | À VÉRIFIER | À VÉRIFIER | 65 | 126s |
| same-14 | same | À VÉRIFIER | À VÉRIFIER | CONFORME | 96 | 69s |
| diff-01 | different | À VÉRIFIER | À VÉRIFIER | À VÉRIFIER | 95 | 58s |
| diff-02 | different | À VÉRIFIER | À VÉRIFIER | À VÉRIFIER | 98 | 0s |
| diff-03 | different | À VÉRIFIER | À VÉRIFIER | À VÉRIFIER | 98 | 48s |
| diff-04 | different | À VÉRIFIER | À VÉRIFIER | À VÉRIFIER | 98 | 47s |
| diff-05 | different | À VÉRIFIER | À VÉRIFIER | À VÉRIFIER | 95 | 55s |
| diff-06 | different | À VÉRIFIER | À VÉRIFIER | À VÉRIFIER | 95 | 50s |

## Known gaps

- Damaged-between-photos (ENDOMMAGÉ): 0 pairs — needs longitudinal photos of the same object before/after damage.
- Same-model-different-unit: 0 pairs — needs two physical units of one model.
- Add pairs by dropping two photos under `eval/pairs/<id>/` and one row in `eval/labels.csv`, then rerun.

