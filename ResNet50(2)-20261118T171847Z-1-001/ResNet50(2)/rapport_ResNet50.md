# Modèle : ResNet50(2) — Classification d'images

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
| **Total** | **~25 049** |

- **Ratio de déséquilibre** (max/min) : **1.53**.
- Même copie Drive du dataset que pour EfficientNetB0 (différente de celle de Kaggle utilisée pour YOLO26/EfficientNetB2).

## 3. Data Cleaning

| Étape | Résultat |
|---|---|
| Images corrompues/tronquées détectées | 1 |
| Doublons détectés (hash MD5) | 499 |
| Images conservées | 25 049 |

## 4. Hyperparamètres retenus (ResNet50, fine-tuning complet)

| Paramètre | Valeur |
|---|---|
| Modèle | `resnet50` (poids ImageNet V2) |
| Taille image | 224 |
| Epochs (max) | 30 |
| Patience (early stopping) | 7 |
| Batch size | 32 |
| Optimizer | AdamW |
| Learning rate | 0.0003 |
| Weight decay | 0.0001 |
| Scheduler | CosineAnnealingLR |
| Label smoothing | 0.1 |
| Data augmentation | Flip horizontal, ColorJitter, RandomResizedCrop |
| Seed | 42 |

## 5. Résultats

- **Arrêt anticipé (early stopping)** à l'epoch **8/30** (meilleur résultat epoch 1)

| Métrique | Valeur affichée |
|---|---|
| Test accuracy | 99.17% |

### Détail par classe (rapport de classification)

| Classe | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| corrosion | 1.00 | 0.99 | 1.00 | 602 |
| cracks | 0.00 | 0.00 | 0.00 | 0 |
| normal | 0.00 | 0.00 | 0.00 | 0 |
| spalling | 0.00 | 0.00 | 0.00 | 0 |

### ⚠️ RÉSERVE MÉTHODOLOGIQUE MAJEURE
Le dossier `val` déjà présent dans cette copie Drive du dataset **ne contient que des images de la classe `corrosion`** (support = 602 pour corrosion, **0 pour les 3 autres classes**). Le test n'a donc porté que sur une seule classe, ce qui rend le score de **99.17% non représentatif** de la performance réelle du modèle sur les 4 classes. Ce chiffre ne doit **pas** être comparé directement aux résultats des autres modèles (YOLO26 : 96.75%, EfficientNetB2 : 97.09%), qui ont été évalués sur des splits de test équilibrés et complets.

**Cause probable** : la structure du dossier `val/` dans la copie du dataset hébergée sur Google Drive est incomplète/mal organisée (contrairement à la version hébergée sur Kaggle, qui ne contenait qu'un dossier `train/` complet, forçant une répartition automatique équilibrée).

**Recommandation** : pour obtenir un résultat fiable et comparable, ré-entraîner ResNet50 en forçant la génération d'un split de validation équilibré à partir de `train/` (comme cela a été fait automatiquement pour YOLO26 et EfficientNetB2), plutôt que d'utiliser le dossier `val/` existant mais incomplet.

## 6. Conclusion
Le fine-tuning technique s'est déroulé sans erreur (convergence rapide, meilleur résultat dès l'epoch 1), mais l'évaluation finale n'est pas exploitable en l'état pour comparer ce modèle aux trois autres. Une nouvelle évaluation sur un test set équilibré est nécessaire avant de tirer des conclusions sur la performance réelle de ResNet50 sur ce problème.
