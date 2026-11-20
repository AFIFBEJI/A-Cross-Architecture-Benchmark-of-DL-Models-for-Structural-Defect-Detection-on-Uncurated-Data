# Registre des poids (checkpoints)

## Règle uniforme

1. Les **checkpoints d'entraînement** (`.keras`, `.pt`, `.pth`) ne sont **jamais** versionnés dans Git (taille, `.gitignore`) ; ils vivent sur Google Drive et sont identifiés ci-dessous par chemin, nom, taille et SHA-256.
2. Seuls les **exports de déploiement convertis** (`.tflite`) de moins de 30 Mo sont versionnés, à titre de commodité. **Ils n'ont produit aucune métrique rapportée** : `code/v1/<modèle>/test.py` évalue le `.keras` (`MODEL_PATH`), le `.tflite` est généré ensuite par `train.py` à partir de ce même `.keras` ; l'accuracy d'un `.tflite` n'a pas été mesurée.
3. Une métrique rapportée est rattachée au fichier de poids exact via cette table. Toute ligne marquée ⚠ doit être complétée avec `tools/hash_files.py`.

## Fichiers versionnés dans le dépôt (calculés)

| Fichier | Taille (octets) | SHA-256 | Nature |
|---|---|---|---|
| `models/aigis_resnet50/v1/Modeles/aigis_resnet50.tflite` (anciennement `aigis_resnet50 (1).tflite`) | 26034760 | `01898ca411ca62dedcbadce6805cd19a47c55029b3a73c5f601352743982d66b` | export converti |
| `models/efficientnet_b0/v1/Modeles/aigis_efficientnetb0.tflite` | 5870256 | `54c7007d01aab0552447ce488dec2bbef08c65c445283746bc2881dc9d8d7369` | export converti |
| `models/efficientnet_b2/v1/Modeles/aigis_efficientnetb2.tflite` | 10002872 | `bcc9f8708dc9e49f8739d0260b6ef4d4b6e14594ea058b7ba9173a5d5b3ed284` | export converti |

## Checkpoints sur Google Drive (les 8 modèles)

Dossier racine : https://drive.google.com/drive/folders/1F4AUqhfDRvCa-3bZb30VMvE_XZP_OPIN. Je n'ai pas pu lire son contenu (page vide hors connexion) : **chemin exact, taille et SHA-256 sont à compléter par l'auteur** ; aucune valeur n'a été inventée.

| ID | Métrique rapportée | Partition | Fichier attendu (nom d'après `train.py`) | Chemin Drive exact | Taille | SHA-256 | Présent ? |
|---|---|---|---|---|---|---|---|
| B0-v1 | 92.32 % acc | test | `aigis_efficientnetb0_final.keras` | ⚠ | ⚠ | ⚠ | ⚠ |
| B2-v1 | 93.05 % | test | `aigis_efficientnetb2_final.keras` | ⚠ | ⚠ | ⚠ | ⚠ |
| R50-v1 | 93.69 % | test | `aigis_resnet50_final.keras` | ⚠ | ⚠ | ⚠ | ⚠ |
| Y26s-v1 | 94.70 % top-1 | test | `best.pt` (`yolo26_train/aigis_yolo26_final/weights/`) | ⚠ | ⚠ | ⚠ | ⚠ (+ ONNX 20,8 Mo) |
| B0-v2 | 97.01 % (corrosion seule) | val | ⚠ (non documenté) | ⚠ | ⚠ | ⚠ | ⚠ |
| B2-v2 | 97.18 % (corrosion seule) | val | ⚠ | ⚠ | ⚠ | ⚠ | ⚠ |
| R50-v2 | 99.17 % (corrosion seule) | val | ⚠ | ⚠ | ⚠ | ⚠ | ⚠ |
| Y26n-v2 | 96.75 % / 97.84 % / 97.67 % (incohérent) | val | `best.pt` (~3,2 Mo, `/content/runs/classify/YOLO26_2/weights/`) | ⚠ | ⚠ | ⚠ | ⚠ |

Procédure : télécharger chaque fichier, `python tools/hash_files.py <fichier>`, coller la ligne ici ; recalculer ensuite la métrique avec `inference/predict_folder.py` + `evaluation/compute_metrics.py` sur le manifeste du split correspondant pour prouver le lien poids → métrique.
