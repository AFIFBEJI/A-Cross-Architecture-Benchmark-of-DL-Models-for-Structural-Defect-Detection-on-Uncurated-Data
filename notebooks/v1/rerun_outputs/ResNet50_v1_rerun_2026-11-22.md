# Ré-exécution v1 — ResNet50 (Keras, Kaggle, 2026-11-22/23)

**Ce n'est pas l'entraînement d'origine** : nouvelle exécution du code `code/v1/resnet50/` avec les hyperparamètres d'origine (`models/aigis_resnet50/v1/Hyperparametres/best_hyperparameters_resnet50.json`), sans nouvelle recherche. Les résultats d'origine ne sont pas modifiés.

| Élément | Valeur |
|---|---|
| Dépôt cloné | commit `95f435a` (fork Yacindaikhi) |
| Environnement | Kaggle, 2× Tesla T4, Python 3.13.16, TensorFlow 2.21.0, Keras 3.13.2 (`environments/kaggle_v1_resnet50_rerun_env.json`) |
| Nettoyage | doublons exclus 3, fuites 3, converties RGB 626, 21 010 images |
| Partition obtenue | **train 14 707 / val 3 151 / test 3 152** |
| Phase 1 (tête seule) | 15 époques ; val acc 0,9330 à la dernière |
| Phase 2 (20 dernières couches) | arrêt anticipé après 6 époques ; val acc max 0,9429 (époque 2) |
| **Test** | accuracy **0,9362**, AUC macro **0,9915** |
| Par classe (test) | corrosion P 0,9703 / R 0,9301 ; cracks 0,9150 / 0,9087 ; normal 1,0000 / 0,9987 ; spalling 0,8637 / 0,9078 (support 844 / 734 / 771 / 803) |
| `aigis_resnet50_final.keras` | 147 591 208 octets, SHA-256 `dc5bed70ed5056f8dfe97101292b8dd795a903c380fc8476615ae272a66633db` |

## Comparaison avec l'exécution d'origine

| | Origine | Ré-exécution | Écart |
|---|---|---|---|
| Accuracy test | 93,69 % | 93,62 % | −0,07 pt (≈ 2 images sur 3 152) |
| AUC macro | 0,9908 | 0,9915 | +0,0007 |

Écart compatible avec la variabilité d'un ré-entraînement (TensorFlow/CUDA non déterministes, versions différentes). La ré-exécution **confirme** l'ordre de grandeur du résultat v1 ; elle ne prouve pas l'identité des poids d'origine, dont le checkpoint final est absent du Drive (voir `docs/WEIGHTS_REGISTRY.md`). Ce nouveau `.keras` devient le checkpoint **traçable** pour la métrique ResNet50 v1, clairement identifié comme issu d'une ré-exécution.

## Vérifications

- Le `aigis_resnet50_final.keras` téléchargé a le SHA-256 ci-dessus (recalculé sur la copie téléchargée, identique au hash calculé sur Kaggle).
- `manifests/v1_clean_resnet50_manifest.csv` est **identique ligne à ligne** à `v1_clean_yolo26_manifest.csv` (21 010 images, MD5 compris) : le nettoyage est reproductible d'une exécution à l'autre.
- Notebook exécuté : `rerun_v1_resnet50_kaggle_executed.ipynb` (les 3 dernières cellules, `ls` et `rm -rf`, sont des diagnostics ajoutés après l'exécution complète).
- Poids intermédiaires de cette exécution (non versionnés) : `best_model_resnet50_phase1.keras` 111857220 o, SHA-256 `3868b5e2f8bd790f8e1b9508848d5ac24c8ba3f497baf6d081e7606924ab3a3d` ; `best_model_resnet50_phase2.keras` 147591208 o, SHA-256 `09e720e784178bc0801f406103fc779d86b53eb1b514fcd4a001c2529f5219bf`.
