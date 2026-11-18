# Modèle : EfficientNetB2(2) — Classification d'images

## 1. Contexte
- **Dataset** : `NewDataSet` (Kaggle), dossier `concrete-damage-v3`
- **Tâche** : Classification — détection de dommages sur béton
- **Classes (4)** : `corrosion`, `cracks`, `normal`, `spalling`
- **Environnement** : Kaggle (GPU Tesla T4)

## 2. Analyse du dataset

| Classe | Nb images (train) |
|---|---|
| corrosion | 2 094 |
| spalling | 1 666 |
| normal | 3 581 |
| cracks | 4 190 |
| **Total** | **11 531** |

- **Ratio de déséquilibre** (max/min) : **2.52**.
- Même dataset équilibré que YOLO26 (source Kaggle, pas la copie Drive).

## 3. Data Cleaning

| Étape | Résultat |
|---|---|
| Images corrompues/tronquées détectées | 0 |
| Doublons détectés | 0 |
| Images conservées | 11 531 |
| Split val automatique | 80/20, stratifié (9 226 train / 2 305 val) |

## 4. Hyperparamètres retenus (EfficientNetB2, fine-tuning complet)

| Paramètre | Valeur |
|---|---|
| Modèle | `efficientnet_b2` (poids ImageNet) |
| Taille image | 260 (résolution native B2) |
| Epochs (max) | 30 |
| Patience (early stopping) | 7 |
| Batch size | 24 |
| Optimizer | AdamW |
| Learning rate | 0.0003 |
| Weight decay | 0.0001 |
| Scheduler | CosineAnnealingLR |
| Label smoothing | 0.1 |
| Dropout | 0.3 (par défaut B2) |
| Data augmentation | Flip horizontal, ColorJitter, RandomResizedCrop |
| Seed | 42 |

## 5. Résultats

- **Arrêt anticipé (early stopping)** à l'epoch **9/30** (meilleur résultat)
- Test effectué sur split val équilibré (2 305 images, 4 classes représentées)

| Métrique | Valeur |
|---|---|
| Test accuracy | **97.09%** |

### Détail par classe (rapport de classification)

| Classe | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| corrosion | 0.96 | 0.95 | 0.96 | 418 |
| cracks | 0.98 | 0.99 | 0.98 | 838 |
| normal | 1.00 | 1.00 | 1.00 | 716 |
| spalling | 0.91 | 0.90 | 0.91 | 333 |
| **Accuracy globale** | | | **0.97** | 2 305 |
| Macro avg | 0.96 | 0.96 | 0.96 | 2 305 |
| Weighted avg | 0.97 | 0.97 | 0.97 | 2 305 |

## 6. Conclusion
Meilleur résultat des 4 modèles à ce stade (97.09%), sur un test set équilibré et fiable. La classe `spalling` reste la plus difficile à distinguer (91% precision/90% recall) — piste d'amélioration possible : plus de données ou d'augmentation ciblée sur cette classe.
