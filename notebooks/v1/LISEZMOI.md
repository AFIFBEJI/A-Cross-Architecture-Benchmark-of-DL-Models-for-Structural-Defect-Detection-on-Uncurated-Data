# notebooks/v1 — notebooks de ré-exécution

Les notebooks d'origine (Kaggle, v1) ont été supprimés et sont introuvables (déclaré par l'auteur le 20/11/2026). Les fichiers `rerun_v1_*_kaggle.ipynb` sont des **ré-exécutions** du code de `code/v1/`, avec les hyperparamètres d'origine, **sans sorties tant qu'ils n'ont pas été exécutés** : ils ne servent de preuve qu'une fois exécutés sur Kaggle et ré-exportés avec leurs sorties. Les fichiers de `models/*/v1/Resultats/` restent ceux de l'exécution d'origine ; un écart de score est attendu et doit être consigné.

## Notebooks d'origine retrouvés (ne pas confondre avec les `rerun_*`)
| Fichier | Contenu | Sorties | Exécution |
|---|---|---|---|
| `v1_efficientnet_b2_cleaning_kaggle_original.ipynb` | visualisation + audit + nettoyage B2 (sans resize) | oui | 2026-07-28, Kaggle, 993 s |
| `v1_yolo26s_kaggle_original_code_only.ipynb` | nettoyage, tuning (yolo26n), entraînement (yolo26s), test, export ONNX | **non** (code seul) | jamais exécuté en Save & Run All |
| `../extension/dataset_extension_v3_kaggle_original.ipynb` | extension v3 (MDMCS, RCS, Surface Crack) | partielles | dernière exécution 2026-11-20, arrêtée sur une erreur de cellule d'essai |

Manquent toujours : notebooks d'origine d'entraînement/test B0, B2, ResNet50 (v1) et tous les notebooks v2.
