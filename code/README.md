# code/

| Dossier | Contenu | Origine |
|---|---|---|
| `v1/data/` | `01_visualize_dataset.py` (audit visuel), `02_clean_dataset.py` (MD5, fuites inter-splits, RGB, resize 256×256, JPEG q85), `03a_extract_crops_labelme.py` + `03b_merge_dataset_v3.py` (extension → *concrete-damage-v3*, non utilisée par v1) | Scripts extraits des cellules exécutées sur Kaggle |
| `v1/<modèle>/{tune,train,test}.py` | Recherche d'hyperparamètres, entraînement 2 phases, évaluation test + matrices de confusion (`efficientnet_b0`, `efficientnet_b2`, `resnet50`, `yolo26`) | Idem. B2 et ResNet50 sont dérivés de B0 (taille 260, backbone, `preprocess_input`, couleur de la heatmap, noms de fichiers) |
| `v2/` | **Reconstructions de référence** PyTorch/Ultralytics à partir des JSON d'hyperparamètres v2 | **Pas les notebooks d'origine** (voir `v2/README.md`) |

Le partitionnement v1 (70/15/15, seed 42) est fait dans `v1/data/02_clean_dataset.py` : le split d'origine du dataset Kaggle est conservé, seuls les doublons/fuites sont retirés (priorité train > val > test).

`v1/data/02_clean_dataset_b2_fullres.py` : variante de nettoyage réellement utilisée pour EfficientNet-B2 (sans resize). Notebooks d'origine retrouvés : voir `../notebooks/`.

Ordre d'exécution et correspondance fichier de résultat → script : `../EXPERIMENT_INDEX.md` et `../notebooks/EXECUTION_ORDER.md`.
