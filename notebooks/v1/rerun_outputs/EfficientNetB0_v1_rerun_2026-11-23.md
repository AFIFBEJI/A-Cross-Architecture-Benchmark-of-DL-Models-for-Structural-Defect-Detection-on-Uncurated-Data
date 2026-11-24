# Ré-exécution v1 — EfficientNetB0 (Keras, Kaggle, 2026-11-23)

**Ce n'est pas l'entraînement d'origine** : nouvelle exécution du code `code/v1/efficientnet_b0/` avec les hyperparamètres d'origine (`models/efficientnet_b0/v1/Hyperparametres/best_hyperparameters_b0.json`), sans nouvelle recherche. Les résultats d'origine ne sont pas modifiés.

| Élément | Valeur |
|---|---|
| Dépôt cloné | commit `95f435a` (fork Yacindaikhi) |
| Environnement | Kaggle, 2× Tesla T4, Python 3.13.16, TensorFlow 2.21.0, Keras 3.13.2 (`environments/kaggle_v1_efficientnet_b0_rerun_env.json`) |
| Nettoyage | doublons exclus 3, fuites 3, converties RGB 626, 21 010 images (resize 256, JPEG q85) |
| Partition obtenue | **train 14 707 / val 3 151 / test 3 152** |
| **Test** | accuracy **0,9223**, AUC macro **0,9894** |
| Par classe (test) | corrosion P 0,9701 / R 0,9230 ; cracks 0,8747 / 0,8937 ; normal 1,0000 / 0,9961 ; spalling 0,8472 / 0,8767 (support 844 / 734 / 771 / 803) |
| `aigis_efficientnetb0_final.keras` | 43 655 933 octets, SHA-256 `0c91248d4584a85f6a280120f0831251a83fc6067a603815050a70f240877984` (recalculé sur la copie téléchargée, identique au hash Kaggle) |

## Comparaison avec l'exécution d'origine

| | Origine | Ré-exécution | Écart |
|---|---|---|---|
| Accuracy test | 92,32 % | 92,23 % | −0,09 pt (≈ 3 images sur 3 152) |
| AUC macro | 0,98894 | 0,98936 | +0,0004 |

Écart compatible avec la variabilité d'un ré-entraînement (TensorFlow/CUDA non déterministes, versions différentes). La ré-exécution confirme l'ordre de grandeur du résultat v1 ; elle ne prouve pas l'identité des poids d'origine (`aigis_efficientnetb0_final.keras` du Drive : 43 655 963 o, fichier différent de celui-ci : 43 655 933 o).

## Vérifications

- `manifests/v1_clean_b0_manifest.csv` est **identique ligne à ligne** (21 010 images, MD5 compris) à `v1_clean_yolo26_manifest.csv` : le nettoyage est reproductible, et B0, ResNet50 et YOLO partagent exactement la même partition.
- Notebook exécuté : `rerun_v1_efficientnet_b0_kaggle_executed.ipynb` (la dernière cellule `ls` est un diagnostic ajouté après l'exécution).
- Des messages CUDA `Delay kernel timed out` apparaissent pendant l'évaluation : avertissements sans effet sur les scores.
