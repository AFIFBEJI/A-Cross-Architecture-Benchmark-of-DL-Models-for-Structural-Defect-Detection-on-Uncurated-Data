# manifests/

Fichiers fournis : `v1_clean_yolo26_manifest.csv`, `v1_clean_resnet50_manifest.csv`, `v1_clean_b0_manifest.csv` (identiques ligne à ligne : mêmes chemins, classes, partitions et MD5, 21 010 images), `v1_clean_b2_manifest.csv` (même partition par nom de fichier, mais MD5 et extensions différents : variante sans redimensionnement), `v2_concrete_damage_v3_manifest.csv` (dossier Drive `concrete-damage-v3`, 25 549 images). Manquent : la copie Kaggle de `concrete-damage-v3` (35 641 images) et le jeu d'entraînement d'origine du v2 (non identifié).

Génération (cellule Kaggle/Colab, dataset monté) :
```bash
python tools/make_manifest.py /kaggle/working/dataset_clean_b0 manifests/v1_clean_b0.csv
# attendu v1 : train=14707 val=3151 test=3152
python tools/verify_manifest.py /kaggle/working/dataset_clean_b0 manifests/v1_clean_b0.csv
```
Colonnes : `relative_path, class, split, md5`. Les manifestes v1 ont été générés lors des ré-exécutions Kaggle de novembre 2026, et celui du v2 dans Colab le 2026-11-26.
