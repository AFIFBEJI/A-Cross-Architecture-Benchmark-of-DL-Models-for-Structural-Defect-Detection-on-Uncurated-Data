# Documentation de l'Environnement Technique — Projet Aigis

**Auteur :** Yacin
**Date :** Août 2026
> **Périmètre** : ce document ne décrit que l'environnement **Kaggle P100 (v1)**. Les environnements Colab T4 / Kaggle T4 de la v2, les versions figées et la procédure de reconstruction sont dans `docs/ENVIRONMENT.md`.

**Objectif :** Garantir la reproductibilité et la transparence scientifique des expérimentations, conformément à la demande du superviseur.

---

## 1. Plateforme d'exécution

| Élément | Détail |
|---|---|
| Plateforme | Kaggle Notebooks |
| Système d'exploitation | Ubuntu (image Kaggle standard, Linux) |
| Python | 3.12.13 |
| Accélérateur GPU | NVIDIA Tesla **P100-PCIE-16GB** |
| Architecture GPU | Pascal (capacité de calcul CUDA **sm_60**) |
| VRAM | 16 269 MiB (~16 Go) |
| Stockage de travail | `/kaggle/working/` — quota observé : 19.5 GiB |
| Persistance | ⚠️ Non permanente : le contenu de `/kaggle/working/` est réinitialisé à chaque redémarrage de session (voir §5) |

## 2. Frameworks et bibliothèques principales

### 2.1 Modèles TensorFlow/Keras (EfficientNetB0, EfficientNetB2, ResNet50)

| Bibliothèque | Version | Usage |
|---|---|---|
| TensorFlow / Keras | Version par défaut Kaggle | Construction et entraînement des modèles CNN |
| Keras Tuner | Installée via `pip install keras-tuner` | Recherche d'hyperparamètres (optimisation bayésienne) |
| scikit-learn | Version par défaut Kaggle | Métriques d'évaluation (classification_report, confusion_matrix, roc_auc_score) |
| Pillow (PIL) | Version par défaut Kaggle | Chargement, redimensionnement, conversion RGB des images |
| matplotlib / seaborn | Version par défaut Kaggle | Visualisation (distribution des classes, matrices de confusion) |

**Preprocessing spécifique par architecture** (point de vigilance documenté suite à erreurs rencontrées) :
- `tensorflow.keras.applications.efficientnet.preprocess_input` pour EfficientNetB0/B2
- `tensorflow.keras.applications.resnet50.preprocess_input` pour ResNet50 (normalisation différente, style "caffe")

### 2.2 Modèle YOLO26 (Ultralytics / PyTorch)

| Bibliothèque | Version utilisée | Remarque |
|---|---|---|
| Ultralytics | 8.4.110 | Framework d'entraînement YOLO26 |
| PyTorch | **2.4.1+cu121** (downgradé) | Voir incompatibilité §3 |
| torchvision | 0.19.1+cu121 | Downgradé en cohérence avec PyTorch |
| CUDA (image par défaut Kaggle) | cu128 (PyTorch 2.10.0 natif) | Incompatible avec le GPU P100, cf. §3 |

## 3. Incompatibilité matérielle rencontrée et solution appliquée

**Problème identifié** : l'image Kaggle par défaut installe **PyTorch 2.10.0+cu128**, qui a abandonné le support des GPU d'architecture Pascal (capacité de calcul **sm_60**, dont fait partie le Tesla P100). PyTorch 2.10 ne supporte que sm_70 et supérieur (Turing, Ampere, Hopper, etc.).

**Erreur observée** :
```
torch.AcceleratorError: CUDA error: no kernel image is available for execution on the device
```

**Solution appliquée** : downgrade explicite de PyTorch/torchvision avant l'installation d'Ultralytics :
```bash
pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics
```

**Limite de cette solution** : ce downgrade doit être **refait à chaque redémarrage de session** (voir §5), car `/kaggle/working/` et l'état de l'environnement pip ne sont pas persistants entre sessions. Une alternative plus stable, documentée mais non retenue par choix pratique, consiste à basculer l'accélérateur sur **GPU T4** (architecture Turing, sm_75), nativement compatible avec PyTorch 2.10 sans aucune modification.

**Conséquence secondaire** : l'export TFLite de YOLO26 échoue avec ce downgrade, les dépendances d'export (`litert-torch`, `torchao`) nécessitant PyTorch ≥ 2.11. Export **ONNX** utilisé en alternative fonctionnelle (`best.onnx`, 20.8 Mo).

## 4. Configuration matérielle des expérimentations

| Modèle | GPU | Framework | Temps d'inférence mesuré |
|---|---|---|---|
| EfficientNetB0 | P100 | TensorFlow/Keras | Non mesuré précisément |
| EfficientNetB2 | P100 | TensorFlow/Keras | Non mesuré précisément |
| ResNet50 | P100 | TensorFlow/Keras | Non mesuré précisément |
| YOLO26s-cls | P100 | PyTorch/Ultralytics | **0.9 ms/image** (mesuré) |

⚠️ **Limite à noter** : seule la vitesse d'inférence de YOLO26 a été mesurée précisément à ce stade. Une mesure comparable pour les 3 autres modèles (sur le même GPU P100, même méthode) est nécessaire pour une comparaison rigoureuse — identifiée comme prochaine étape.

## 5. Limitations de persistance de l'environnement Kaggle

Point critique pour la reproductibilité, rencontré à plusieurs reprises durant le projet :

- Le contenu de `/kaggle/working/` (datasets nettoyés, modèles entraînés, hyperparamètres sauvegardés) est **perdu à chaque redémarrage de session ou de kernel**.
- Les packages installés via `pip install` dans une session ne persistent pas non plus après un redémarrage.
- **Bonne pratique adoptée** : sauvegarde systématique via "Save Version" dès qu'une étape critique (nettoyage de dataset, résultats de tuning, modèle entraîné) est terminée, pour figer l'état de `/kaggle/working/` de façon durable.
- **Bonne pratique recommandée pour la suite** : publier les datasets nettoyés comme "Kaggle Dataset" persistant (via le bouton "New Dataset" depuis le panneau Output), pour éviter de régénérer le nettoyage à chaque nouvelle session.

## 6. Structure des datasets utilisés

| Dataset | Chemin Kaggle | Taille | Résolution de retraitement |
|---|---|---|---|
| Dataset principal (Aigis) | `/kaggle/input/datasets/daikhiyassine/dataset` | 21 013 images brutes → 21 010 après nettoyage | 256×256 (retraitement commun), entrée modèle 224×224 ou 260×260 selon architecture |

**Pipeline de nettoyage appliqué** (commun à tous les modèles) :
1. Détection d'images corrompues (`PIL.Image.verify()`)
2. Détection de doublons par hash MD5, avec vérification spécifique des fuites entre splits train/val/test
3. Conversion en RGB (images non-RGB détectées et converties)
4. Redimensionnement (LANCZOS) et compression JPEG (qualité 85)

## 7. Reproductibilité

- **Seeds** : `42` pour l'ordre de mélange Keras (`image_dataset_from_directory`) et `random.seed()` des fusions de datasets ; **YOLO26 v1 : seed 0** (`args.yaml`, défaut Ultralytics) — correction d'une mention antérieure « 42 pour toutes les exécutions ». Pas de graine globale TensorFlow. Détail : `docs/ENVIRONMENT.md` §2.
- **Hyperparamètres sauvegardés** en JSON après chaque recherche (Keras Tuner / Ultralytics `.tune()`), permettant de reproduire l'entraînement final sans refaire la recherche.
- **Callbacks d'entraînement standardisés** : `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint`, `CSVLogger` pour tous les modèles TensorFlow, garantissant une trace complète de l'historique d'entraînement.

---
*Document rédigé pour le rapport technique du projet Aigis, en réponse au point 6 des consignes du superviseur (09/08/2026).*
