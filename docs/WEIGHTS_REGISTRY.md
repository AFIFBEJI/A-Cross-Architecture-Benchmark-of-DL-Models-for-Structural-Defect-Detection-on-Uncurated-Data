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

Dossier racine `MonProjet` : https://drive.google.com/drive/folders/1F4AUqhfDRvCa-3bZb30VMvE_XZP_OPIN. Inventaire relevé le 2026-11-22 par lecture directe du Drive (nom, taille, identifiant, dossier). **Drive ne fournit pas de SHA-256** : les hash restent à calculer sur une copie téléchargée (`Get-FileHash -Algorithm SHA256`). Aucune valeur n'a été inventée.

Les chemins sont donnés relativement à `MonProjet/`. Les dossiers `*(2)` et `YOLO26(2)` sont ceux de la v2.

| ID | Métrique rapportée | Partition | Chemin Drive | Fichier | Taille (octets) | ID Drive | SHA-256 |
|---|---|---|---|---|---|---|---|
| B0-v1 | 92.32 % acc | test | `EfficientNetB0/Modeles/` | `aigis_efficientnetb0_final.keras` | 43655963 | `14A56an3F2G0GQGT_Auv9JBJJigwkbd9z` | `687762f1e4636716c290c16313e44f274ced153c66e75c316dbb16286eec8628` (copie locale ; taille identique au Drive) |
| B2-v1 | 93.05 % | test | `EfficientNetB2/Modeles/` | `aigis_efficientnetb2_final.keras` | 60586878 | `1chd4IoT9fOfNI_6ZqO6mKTFASOQlxdXy` | `a80ca973fa9b8fb5f645a28ebea1ce82bf6409d9cd1dbf8e264fe64d9b272e1b` (copie locale ; taille identique au Drive) |
| R50-v1 | 93.69 % | test | `ResNet50/Modeles/` | **`aigis_resnet50_final.keras` ABSENT** ; seul `best_model_resnet50_phase1.keras` est présent | 111857238 (phase 1) | `1aK3Alf_zHyA3bJoSr1-zxFV2pS3wvOPC` | ⚠ à calculer ; ne prouve PAS la métrique 93,69 % (phase 1 seulement) |
| Y26s-v1 | 94.70 % top-1 | test | `Yolo26/Modeles/` | `best.pt` (+ `best.onnx` 21803864 o, export) | 11032699 | `1f-6MVKv97FcnHYvbW6krDXn_1Vx4b6-9` | ⚠ à calculer |
| B0-v2 | 97.01 % (corrosion seule) | val | `EfficientNetB0(2)/Modeles/` | `best_efficientnetb0.pt` | 16353495 | `1vFXfaiH12A8IBne7A4mT4mlCYq891dfH` | ⚠ à calculer |
| B2-v2 | 97.18 % (corrosion seule) | val | `EfficientNetB2(2)/Modeles/` | `best_efficientnetb2.pt` | 31286239 | `1yrxDeLFhNbPQy0e9EO0FnVbmrXyhYnTF` | ⚠ à calculer |
| R50-v2 | 99.17 % (corrosion seule) | val | `ResNet50(2)/Modeles/` | `best_resnet50.pt` | 94384867 | `1gOXjrFBHuWOmSY-gtT5SaZCGHO1drTj3` | ⚠ à calculer |
| Y26n-v2 | 96.75 % / 97.84 % / 97.67 % (incohérent) | val | `YOLO26(2)/Modeles/` | `best_YOLO26_2.pt` | 3193346 | `1Gh-hwz_-so-sthzkBLlJzk7JtTPVTSNn` | ⚠ à calculer |

### Poids issus de ré-exécutions (nov. 2026, ne remplacent aucun original)

| ID | Métrique (test) | Fichier | Taille | SHA-256 | Emplacement |
|---|---|---|---|---|---|
| Y26s-v1 rerun | 94,89 % / AUC 0,9940 | `best.pt` | 11032386 | `f4ec145c5cd9d7584e87dc3a5ac91adb78fd79d809bcf8318ce862f213852b5d` | à déposer sur Drive `rerun_2026-11/` ; sortie Kaggle `yolo26_train/aigis_yolo26_final/weights/` |
| R50-v1 rerun | 93,62 % / AUC 0,9915 | `aigis_resnet50_final.keras` | 147591208 | `dc5bed70ed5056f8dfe97101292b8dd795a903c380fc8476615ae272a66633db` | à déposer sur Drive `rerun_2026-11/` |
| B0-v1 rerun | 92,23 % / AUC 0,9894 | `aigis_efficientnetb0_final.keras` | 43655933 | `0c91248d4584a85f6a280120f0831251a83fc6067a603815050a70f240877984` | à déposer sur Drive `rerun_2026-11/` |
| B2-v1 rerun | 92,89 % / AUC 0,9898 | `aigis_efficientnetb2_final.keras` | 60586848 | `52c91d304a0811bd2f4861d175ad20664874dcef124b0bb6f3458682463af272` (hash Kaggle) | à déposer sur Drive `rerun_2026-11/` |

Voir `notebooks/v1/rerun_outputs/YOLO26s_v1_rerun_2026-11-22.md`.
Voir aussi `notebooks/v1/rerun_outputs/ResNet50_v1_rerun_2026-11-22.md` `EfficientNetB0_v1_rerun_2026-11-23.md` et `EfficientNetB2_v1_rerun_2026-11-24.md`.

### Autres fichiers présents (à ne pas confondre avec les poids rapportés)

| Dossier | Fichier | Taille | Remarque |
|---|---|---|---|
| `EfficientNetB0/Modeles/` | `best_model_b0_phase1.keras` (32834559), `best_model_b0_phase2.keras` (43655963), `aigis_efficientnetb0.tflite` (5870256) | | phase2 a la même taille que `_final` : probablement identique (à vérifier par hash) |
| `EfficientNetB2/Modeles/` | `best_model_phase1.keras` (43845968), `best_model_phase2.keras` (60586878), `aigis_efficientnetb2.tflite` (10002872) | | idem |
| `ResNet50/Modeles/` | `aigis_resnet50 (1).tflite` (26034760) | | nom à nettoyer sur Drive aussi |
| `Yolo26/Modeles/` | `best.onnx` (21803864) | | export |
| `YOLO26(2)/Modeles/` | `last_YOLO26_2.pt` (3193346), **`best (1).pt` (9383971)** | | `last_` a la même taille que `best_` ; `best (1).pt` est un autre fichier de taille différente, d'origine inconnue : à identifier ou supprimer |

### Points d'attention

1. **ResNet50 v1** : le checkpoint final `aigis_resnet50_final.keras` n'existe pas sur le Drive. La métrique 93,69 % ne peut donc pas être rattachée à un fichier. Soit le retrouver (autre dossier/PC), soit refaire `code/v1/resnet50/train.py` puis `test.py` et le déposer dans `rerun_2026-11/`.
2. **Poids v2 PyTorch** : les trois `.pt` ne contiennent que le meilleur point de contrôle ; aucun `.keras`. Ils correspondent aux entraînements « corrosion seule » (voir `EXPERIMENT_INDEX.md`, A1).
3. **Hash des poids B0/B2 v1** : calculés sur le PC de l'auteur ; les tailles sont identiques à celles du Drive, mais seule la comparaison avec un hash du fichier Drive téléchargé le prouve complètement.
4. Le nouveau `best.pt` issu de la ré-exécution Kaggle (nov. 2026) doit aller dans un dossier séparé `rerun_2026-11/` et **ne remplace aucun fichier ci-dessus**.

Procédure : télécharger chaque fichier ⚠, `Get-FileHash <fichier> -Algorithm SHA256 | Format-List` (ou `python tools/hash_files.py <fichier>`), coller la valeur ici ; recalculer ensuite la métrique avec `inference/predict_folder.py` + `evaluation/compute_metrics.py` sur le manifeste du split correspondant pour prouver le lien poids → métrique.
