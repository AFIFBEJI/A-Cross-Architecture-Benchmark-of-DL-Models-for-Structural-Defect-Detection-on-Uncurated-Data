# Ré-exécution v1 — EfficientNetB2 (Keras, Kaggle, 2026-11-24)

**Ce n'est pas l'entraînement d'origine** : nouvelle exécution du code `code/v1/efficientnet_b2/` avec les hyperparamètres d'origine (`models/efficientnet_b2/v1/Hyperparametres/best_hyperparameters_b2.json` : dense 1024, dropout 0,3, lr 0,001, 50 couches dégelées, rmsprop, sans batchnorm), sans nouvelle recherche. Les résultats d'origine ne sont pas modifiés.

| Élément | Valeur |
|---|---|
| Dépôt cloné | commit `95f435a` (fork Yacindaikhi) |
| Environnement | Kaggle, 2× Tesla T4, Python 3.13.16, TensorFlow 2.21.0, Keras 3.13.2 (`environments/kaggle_v1_efficientnet_b2_rerun_env.json`) |
| Nettoyage | variante **B2 : sans redimensionnement, format d'origine** (`code/v1/data/02_clean_dataset_b2_fullres.py`) ; 3 doublons retirés (étiquettes en conflit spalling/cracks), 21 010 images copiées, 626 converties RGB |
| Partition obtenue | **train 14 707 / val 3 151 / test 3 152** (mêmes effectifs par classe que B0/R50/YOLO) |
| Entraînement | phase 1 : 10 époques (tête seule) ; phase 2 : 4 époques (50 dernières couches), val acc 0,9432 à la dernière |
| **Test** | accuracy **0,9289**, AUC macro **0,9898** |
| Par classe (test) | corrosion P 0,9630 / R 0,9242 ; cracks 0,8782 / 0,9237 ; normal 0,9987 / 0,9987 ; spalling 0,8761 / 0,8717 (support 844 / 734 / 771 / 803) |
| `aigis_efficientnetb2_final.keras` | 60 586 848 octets, SHA-256 `52c91d304a0811bd2f4861d175ad20664874dcef124b0bb6f3458682463af272` (calculé sur Kaggle) |

## Comparaison avec l'exécution d'origine

| | Origine | Ré-exécution | Écart |
|---|---|---|---|
| Accuracy test | 93,05 % | 92,89 % | −0,16 pt (≈ 5 images sur 3 152) |
| AUC macro | 0,99088 | 0,9898 | −0,0011 |

Écart compatible avec la variabilité d'un ré-entraînement (TensorFlow/CUDA non déterministes, versions différentes). La ré-exécution confirme l'ordre de grandeur du résultat v1 ; elle ne prouve pas l'identité des poids d'origine (`aigis_efficientnetb2_final.keras` du Drive : 60 586 878 o ; celui-ci : 60 586 848 o).

## Vérifications

- `aigis_efficientnetb2_final.keras` téléchargé : SHA-256 recalculé `52c91d304a0811bd2f4861d175ad20664874dcef124b0bb6f3458682463af272`, 60 586 848 octets, identique au hash calculé sur Kaggle.
- `manifests/v1_clean_b2_manifest.csv` : 21 010 images (train 14 707 / val 3 151 / test 3 152). **La partition est identique à celle de YOLO/ResNet50/B0** : en comparant les noms de fichiers sans extension, les 21 010 images ont la même classe et la même partition. En revanche **les MD5 sont tous différents** (pas de redimensionnement ni de recompression JPEG pour B2, et l'extension d'origine est conservée : 18 713 chemins seulement sont identiques mot pour mot). C'est attendu : B2 est la variante « résolution d'origine ».
- Notebook exécuté : `rerun_v1_efficientnet_b2_kaggle_executed.ipynb` (la dernière cellule `ls` est un diagnostic ajouté après l'exécution).
- Les messages CUDA `Delay kernel timed out` sont des avertissements sans effet sur les scores.
