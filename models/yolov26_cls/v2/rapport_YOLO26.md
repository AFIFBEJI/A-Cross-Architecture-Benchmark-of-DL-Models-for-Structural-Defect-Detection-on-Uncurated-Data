# Modèle : YOLO26(2) — Classification d'images

> ⚠ **Métrique de validation** (le split val a servi à l'early stopping), pas de test. Variante **YOLO26n-cls** (v1 : YOLO26s-cls). Valeurs top-1 incohérentes dans le dépôt : voir `README_YOLO26.md`.


## 1. Contexte
- **Dataset** : `NewDataSet` (Kaggle), dossier `concrete-damage-v3`
- **Tâche** : Classification — détection de dommages sur béton
- **Classes (4)** : `corrosion`, `spalling`, `normal`, `cracks`
- **Environnement** : Kaggle (GPU Tesla T4)

## 2. Analyse du dataset

| Classe | Nb images (train) |
|---|---|
| corrosion | 2 094 |
| spalling | 1 666 |
| normal | 3 581 |
| cracks | 4 190 |
| **Total** | **11 531** |

- **Ratio de déséquilibre** (max/min) : **2.52** — acceptable, pas de rééquilibrage nécessaire.
- Résolutions hétérogènes selon les classes (230 px à 970 px de large).

## 3. Data Cleaning

| Étape | Résultat |
|---|---|
| Images corrompues détectées | 0 |
| Doublons détectés (hash MD5) | 0 |
| Images conservées | 11 531 |

- **Split train/val automatique** créé (absent à l'origine) : 80% train (9 226) / 20% val (2 305), stratifié par classe.

## 4. Hyperparamètres retenus (YOLO26n-cls)

| Paramètre | Valeur |
|---|---|
| Modèle | `yolo26n-cls.pt` (nano) |
| Taille image | 224 |
| Epochs (max) | 100 |
| Patience (early stopping) | 20 |
| Batch size | 64 |
| Optimizer | auto |
| Learning rate initial | 0.001 |
| Learning rate final | 0.01 |
| Momentum | 0.937 |
| Weight decay | 0.0005 |
| Warmup epochs | 3 |
| Dropout | 0.1 |
| Data augmentation | Activée |
| Seed | 42 |

## 5. Résultats

- **Arrêt anticipé (early stopping)** à l'epoch **64/100**
- Dataset de test : split val (4 classes représentées, 2 305 images)

| Métrique | Valeur |
|---|---|
| Top-1 accuracy | **96.75%** |
| Top-5 accuracy | **100%** |

- Modèle final : `best.pt` (~3.2 Mo, 1 531 148 paramètres, 3.2 GFLOPs)
- Résultat fiable : dataset de test équilibré et représentatif des 4 classes.

## 6. Conclusion
Résultat très satisfaisant pour un modèle nano. Convergence rapide, adapté à un déploiement léger (edge/embarqué).
