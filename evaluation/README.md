# evaluation/

| Script | Rôle | Produit |
|---|---|---|
| `compute_metrics.py` | accuracy, P/R/F1, macro-F1, AUC, matrices de confusion à partir d'un CSV de prédictions | `metrics_<tag>.json`, `confusion_matrix*_<tag>.png` |
| `mcnemar_test.py` | McNemar exact + IC bootstrap sur l'écart d'accuracy entre 2 modèles (mêmes images) | sortie console |
| `make_figures.py` | courbes d'apprentissage depuis `training_log_*` / `results.csv` | PNG |

Chaîne : `inference/predict_folder.py` → `compute_metrics.py` → `mcnemar_test.py`.
Les fichiers de `models/*/v1/Resultats/` ont été produits directement par `code/v1/<modèle>/test.py` (voir `EXPERIMENT_INDEX.md`), ces scripts permettent de les recalculer de façon indépendante et d'ajouter les tests statistiques (non exécutés à ce jour).
