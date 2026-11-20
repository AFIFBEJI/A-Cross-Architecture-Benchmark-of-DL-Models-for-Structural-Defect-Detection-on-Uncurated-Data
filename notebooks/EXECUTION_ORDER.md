# Ordre d'exécution et export des notebooks

## Export (obligatoire pour la reproductibilité)
Kaggle : *File → Download notebook* (.ipynb) **après** « Save & Run All » pour garder les sorties. Colab : *File → Download → .ipynb*. Ne pas modifier les sorties à la main. Déposer dans `notebooks/v1/` ou `notebooks/v2/` (nom : `v<N>_<modèle>_<plateforme>.ipynb`). Aucun notebook n'a été fabriqué ici : seuls les exports réels des sessions font foi.

## v1 (Kaggle, dataset `daikhiyassine/dataset`, 21 013 images brutes → 21 010 nettoyées)
Pour chaque modèle M ∈ {efficientnet_b0, efficientnet_b2, resnet50, yolo26} :
1. `code/v1/data/01_visualize_dataset.py` — audit visuel (distribution, échantillons)
2. `code/v1/data/02_clean_dataset.py` (avec `CLEAN_PATH=/kaggle/working/dataset_clean_<suffixe_M>` : `_b0`, `_yolo26`, `_aigis` pour ResNet50, voir les `DATASET_PATH` de `tune/train/test`) — nettoyage + partition → `dataset_clean_<M>` + `rapport_nettoyage_final_<M>.json`
3. `code/v1/<M>/tune.py` — recherche d'hyperparamètres → `best_hyperparameters_<M>.json` (⚠ sur Kaggle `/kaggle/working` n'est pas persistant : relancer 2 et 3 après tout redémarrage)
4. `code/v1/<M>/train.py` — entraînement → `.keras`/`best.pt`, `training_log_*`, `.tflite`/`.onnx`
5. `code/v1/<M>/test.py` — évaluation test → `test_results_<M>.json`, matrices de confusion
6. `tools/make_manifest.py` sur le dataset nettoyé → `manifests/` ; `tools/capture_env.py` → `environments/`
7. (optionnel) `evaluation/make_figures.py`, `evaluation/mcnemar_test.py`

Environnement : voir `reports/Environnement_Technique/` (P100 ⇒ `torch==2.4.1+cu121`).

## Extension de données (hors v1 et hors v2 des tableaux de résultats)
`code/v1/data/03a_extract_crops_labelme.py` (MDMCS, RCS-2119, RCS-1841) puis `03b_merge_dataset_v3.py` → *concrete-damage-v3* (35 641 images).

## v2
Notebooks d'origine à ajouter (voir `code/v2/README.md`).
