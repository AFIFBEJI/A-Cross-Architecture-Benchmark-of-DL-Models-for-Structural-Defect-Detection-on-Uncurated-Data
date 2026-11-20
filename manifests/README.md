# manifests/

Un CSV par jeu de données réellement utilisé : `v1_clean_b0.csv`, `v1_clean_yolo26.csv`, `v1_clean_aigis.csv`, `v1_clean_b2.csv` (⚠ si le nettoyage a été refait par modèle, les 4 doivent être identiques), `v2_drive_newdataset.csv`, `v2_kaggle_b2_yolo.csv`, `v3_concrete_damage.csv`.

Génération (cellule Kaggle/Colab, dataset monté) :
```bash
python tools/make_manifest.py /kaggle/working/dataset_clean_b0 manifests/v1_clean_b0.csv
# attendu v1 : train=14707 val=3151 test=3152
python tools/verify_manifest.py /kaggle/working/dataset_clean_b0 manifests/v1_clean_b0.csv
```
Colonnes : `relative_path, class, split, md5`. Aucun manifeste n'est fourni à ce jour : les images n'étaient pas disponibles lors de la préparation du dépôt.
