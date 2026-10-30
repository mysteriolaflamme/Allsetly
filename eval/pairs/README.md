# Témoin evaluation pairs

20 annotated pairs under `eval/pairs/<id>/` (`sale.jpg` + `delivery.jpg`),
labeled in `eval/labels.csv`. All photos are our own, derived from two real
shots of one headphone set (`photo.jpg` code K7, `livraison.jpg` code A8).
Derived views are documented per pair in the `notes` column — no hidden edits.

## Groups

| Pairs | Relation | Codes | Expected | Construction |
|---|---|---|---|---|
| same-01 | same | ok | CONFORME | golden: the two real shots |
| same-02/03 | same | ok | CONFORME | identical photo twice (sanity) |
| same-04..10 | same | ok | CONFORME | rotation, brightness, contrast, border crop |
| same-11..14 | same | missing | À VÉRIFIER | challenge code cropped out (freshness KO) |
| diff-01..03 | different | ok | À VÉRIFIER | paper-with-code only vs object (see below) |
| diff-04..06 | different | missing | À VÉRIFIER | empty table/floor vs object |

## Ground-truth rules

- Same object + readable codes → CONFORME (identical state).
- Any code missing or unreadable → À VÉRIFIER by the freshness guardrail
  (deterministic, section 5 of PROJET.md).
- Paper-only vs object (diff-01..03): the delivery photo shows no object, so
  no two clear contradictions can be established; per section 5
  ("dans le doute → À VÉRIFIER") and pillar 3 (guided collection), the
  correct system answer is À VÉRIFIER with a photo request — it must never
  clear the pair as the same object. Fraud confirmation (NON CONFORME) needs
  two fully visible different objects; those pairs are still missing (gaps).
- The headline metric is the **false accusation rate**: NON CONFORME on a
  same-object pair. Goal: 0 %.

## Known gaps

- Damaged-between-photos (ENDOMMAGÉ): 0 pairs, needs longitudinal photos.
- Same-model-different-unit: 0 pairs, needs two physical units.

Add pairs by dropping two photos under `eval/pairs/<id>/`, appending one row
to `eval/labels.csv`, and rerunning `python -m eval.run_eval`.
