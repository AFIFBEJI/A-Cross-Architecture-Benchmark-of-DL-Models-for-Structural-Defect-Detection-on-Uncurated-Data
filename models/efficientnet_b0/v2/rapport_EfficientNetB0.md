# Modèle : EfficientNetB0(2) — Classification d'images

## 1. Contexte
- **Dataset** : `NewDataSet` (copie stockée sur Google Drive)
- **Tâche** : Classification — détection de dommages sur béton
- **Classes (4)** : `corrosion`, `cracks`, `normal`, `spalling`
- **Environnement** : Google Colab (GPU T4)

## 2. Analyse du dataset

| Classe | Nb images (train) |
|---|---|
| corrosion | 5 208 |
| normal | 7 094 |
| cracks | 7 648 |
| spalling | 4 996 |
| **Total** | **~25 050** |

- **Ratio de déséquilibre** (max/min) : **1.53** — bien équilibré.
- ⚠️ Cette copie du dataset (sur Drive) est différente et plus volumineuse que celle utilisée pour YOLO26/EfficientNetB2 (sur Kaggle).

## 3. Data Cleaning

| Étape | Résultat |
|---|---|
| Images corrompues/tronquées détectées | 0 → 1 selon la version du script (correction apportée en cours de route pour mieux détecter les images tronquées) |
| Doublons détectés (hash MD5) | ~499 |
| Images conservées | ~25 049–25 050 |

## 4. Hyperparamètres retenus (EfficientNetB0, fine-tuning complet)

| Paramètre | Valeur |
|---|---|
| Modèle | `efficientnet_b0` (poids ImageNet) |
| Taille image | 224 |
| Epochs (max) | 30 |
| Patience (early stopping) | 7 |
| Batch size | 32 |
| Optimizer | AdamW |
| Learning rate | 0.0003 |
| Weight decay | 0.0001 |
| Scheduler | CosineAnnealingLR |
| Label smoothing | 0.1 |
| Dropout | 0.2 (par défaut B0) |
| Data augmentation | Flip horizontal, ColorJitter, RandomResizedCrop |
| Seed | 42 |

## 5. Résultats

- **Arrêt anticipé (early stopping)** à l'epoch **11/30**

| Métrique | Valeur |
|---|---|
| Test accuracy | **97.01%** |

### ⚠️ Réserve méthodologique importante
Le dossier `val` déjà présent dans la copie Drive du dataset **ne représente pas les 4 classes de façon complète** (un déséquilibre a provoqué une erreur lors du calcul du rapport de classification détaillé, indiquant que certaines classes étaient absentes ou sous-représentées dans ce split). **Le chiffre de 97.01% doit donc être interprété avec prudence** — il n'est pas garanti que les 4 classes aient été évaluées de façon équilibrée, contrairement à YOLO26 et EfficientNetB2 qui utilisent un split de test entièrement équilibré et vérifié.

## 6. Conclusion
Résultat globalement cohérent avec les autres modèles (~97%), mais nécessiterait une ré-évaluation sur un split de test strictement équilibré pour une comparaison rigoureuse entre modèles.
