# ResNet50(2) — Classification

> ⚠ **Résultat sur la classe corrosion uniquement (602 images), PAS un résultat à quatre classes.** Le `val/` utilisé ne contenait que des images `corrosion` (support 0 pour cracks, normal, spalling). L'accuracy de 99.17 % est donc le rappel sur une classe ; elle ne doit pas être comparée aux résultats v1 ni aux autres modèles. Les moyennes macro de `classification_report.json` (≈0.25) sont des artefacts de ce biais.

## Résultats (corrosion uniquement)
- Accuracy sur 602 images de corrosion : 0.9917
- Partition : validation (`val/`), utilisée aussi pour l'early stopping — ce n'est pas un test indépendant.
- Détail : `Resultats/classification_report.json`
- Évaluation à quatre classes : **non disponible** ; à produire comme nouvelle exécution identifiée (voir `EXPERIMENT_INDEX.md`).
