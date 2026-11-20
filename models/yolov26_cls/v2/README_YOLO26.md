# YOLO26n-cls(2) — Classification

> Variante : **YOLO26n-cls** (nano) en v2 ; la v1 utilise YOLO26**s**-cls.
> ⚠ **Métrique de validation, pas de test** : le split `val` (2 305 images, 4 classes) a aussi servi à l'early stopping et à la sélection du meilleur epoch.

## Résultats (validation)
- Top-5 : 1.0000
- Top-1 : **trois valeurs incohérentes dans le dépôt** — 0.9784 (ce README, version précédente), 96.75 % (`rapport_YOLO26.md`), 0.97674 maximum dans `Resultats/results.csv` (22 epochs seulement, alors que le rapport indique un arrêt à l'epoch 64). Ne citer aucune de ces valeurs comme définitive tant que le notebook et `best.pt` ne sont pas fournis (`EXPERIMENT_INDEX.md`, A5).
