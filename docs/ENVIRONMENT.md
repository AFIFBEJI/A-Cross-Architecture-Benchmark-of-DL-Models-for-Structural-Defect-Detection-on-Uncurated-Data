# Environnements d'exécution et reconstruction

Détails historiques (P100, incompatibilités) : `reports/Environnement_Technique/Documentation_Environnement_Technique.md`.

## 1. Environnements utilisés

| Env. | Expériences | Plateforme / GPU | Python | Frameworks | CUDA | Source des versions |
|---|---|---|---|---|---|---|
| **kaggle_p100_v1** | v1 : EfficientNet-B0/B2, ResNet50 (Keras), YOLO26s-cls | Kaggle, Tesla P100-PCIE-16GB (sm_60, 16 269 MiB) ⚠ GPU de B0/B2/ResNet50 à confirmer (README B0/B2 : « P100/T4 ») | 3.12.13 | YOLO : ultralytics 8.4.110, torch 2.4.1+cu121, torchvision 0.19.1+cu121 ; TF/Keras, Keras Tuner : ⚠ non relevées | image Kaggle cu128 ; torch downgradé cu121 pour YOLO | rapport d'environnement |
| **colab_t4_v2** | v2 : ResNet50, EfficientNet-B0 (PyTorch) | Google Colab, GPU T4 | ⚠ | ⚠ | ⚠ | rapports v2 (plateforme, GPU) uniquement |
| **kaggle_t4_v2** | v2 : EfficientNet-B2, YOLO26n-cls | Kaggle, GPU T4 | ⚠ | ⚠ (Ultralytics, `/content/runs/classify` suggère que YOLO v2 a tourné sur Colab : à confirmer) | ⚠ | rapports v2 |

Toutes les cases ⚠ se remplissent en une commande par notebook (§3) ; le dépôt ne les contient pas et je ne les ai pas estimées.

## 2. Seeds (réconciliation)

- Le rapport d'environnement indiquait « seed 42 pour toutes les exécutions ». C'est inexact pour YOLO26 v1 : `models/yolov26_cls/v1/Hyperparametres/args.yaml` enregistre `seed: 0` (valeur par défaut d'Ultralytics, `deterministic: true`) ; `code/v1/yolo26/train.py` ne fixe pas de seed. **Seed réelle de YOLO26 v1 = 0.**
- Keras v1 : `seed=42` n'est passé qu'à `image_dataset_from_directory` (ordre de mélange) ; aucune graine globale (`tf.keras.utils.set_random_seed`) n'est posée, donc l'entraînement n'est pas bit-à-bit reproductible. Fusions/échantillonnage de données : `random.seed(42)`.
- v2 : seed 42 déclarée dans les JSON d'hyperparamètres (YOLO26n : `seed: 42`).

## 3. Figer les versions (à faire dans chaque notebook)

Cellule à exécuter une fois les installations terminées :
```bash
!git clone https://github.com/AFIFBEJI/A-Cross-Architecture-Benchmark-of-DL-Models-for-Structural-Defect-Detection-on-Uncurated-Data.git repo
%cd repo
!python tools/capture_env.py kaggle_p100_v1      # ou colab_t4_v2, kaggle_t4_v2
```
Produit `environments/<nom>_freeze.txt` (pip freeze) et `environments/<nom>_env.json` (Python, OS, GPU, CUDA, versions TF/Keras/torch/ultralytics). Commiter ces fichiers, puis remplacer les `requirements-*.txt` par leur contenu filtré.

## 4. Reconstruire une session

**Kaggle** (v1, P100)
1. Nouveau notebook, *Accelerator = GPU P100* (ou T4), *Internet on*.
2. **Avant de démarrer la session**, *Add Input* : `daikhiyassine/dataset` (et `concrete-damage-v3`, `mdmcs-dataset`, RCS-2119/1841 pour l'extension). Les datasets ajoutés après le démarrage sont invisibles.
3. Chemins : `/kaggle/input/datasets/daikhiyassine/<slug>`.
4. YOLO sur P100 : `pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 --index-url https://download.pytorch.org/whl/cu121`, puis `pip install ultralytics==8.4.110`, puis **redémarrer le noyau**. Export TFLite impossible dans cette configuration → ONNX.
5. Exécuter l'ordre de `notebooks/EXECUTION_ORDER.md`. `/kaggle/working` est effacé à chaque redémarrage : relancer nettoyage et tuning, ou *Save Version* / publier un dataset Kaggle pour conserver `best_hyperparameters_*.json` et les checkpoints.
6. Vérifier le jeu de données : `python tools/verify_manifest.py <dataset_clean> manifests/<nom>.csv`.

**Colab** (v2, T4)
1. *Runtime → Change runtime type → T4 GPU*.
2. Monter Drive : `from google.colab import drive; drive.mount('/content/drive')` ; copie du dataset `NewDataSet` dans `/content` pour la vitesse.
3. `pip install -r requirements-torch.txt` (une fois figé), puis *Runtime → Restart session*.
4. Lancer `tools/capture_env.py colab_t4_v2`, puis le notebook ; sauvegarder les poids et `runs/` sur Drive avant la fin de session (le disque `/content` est éphémère).
