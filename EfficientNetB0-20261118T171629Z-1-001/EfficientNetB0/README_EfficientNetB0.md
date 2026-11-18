# Aigis — Détection de Défauts Structurels en Béton
## Rapport Final — Modèle EfficientNetB0

**Projet :** Aigis — Pipeline de deep learning pour la détection automatique de défauts structurels (corrosion, fissures, écaillage) sur infrastructures en béton
**Stage :** ESPRIT — Recherche en vision par ordinateur
**Auteur :** Yacin
**Architecture :** EfficientNetB0 (transfer learning, poids ImageNet)
**Statut :** ✅ Entraînement et évaluation terminés

---

## 1. Objectif

Ce dossier regroupe l'ensemble du pipeline et des résultats pour le modèle **EfficientNetB0**, entraîné à classifier des images de surfaces en béton en 4 catégories :

| Classe | Description |
|---|---|
| `corrosion` | Corrosion visible (armatures, taches de rouille) |
| `cracks` | Fissures |
| `spalling` | Écaillage / éclatement du béton |
| `normal` | Surface saine, sans défaut |

EfficientNetB0 est la version la plus légère de la famille EfficientNet (~4M paramètres), retenue ici pour évaluer le compromis **précision / vitesse d'inférence**, notamment en vue d'un déploiement mobile ou embarqué (export TFLite).

## 2. Pipeline

Le traitement suit 4 étapes, chacune tracée dans les sous-dossiers de ce répertoire :

1. **Visualisation du dataset** — distribution des classes, échantillons visuels par classe
2. **Nettoyage des données** — détection des doublons, vérification des fuites entre splits train/val/test, conversion RGB, resize et compression
3. **Recherche d'hyperparamètres** — recherche bayésienne (Keras Tuner, 10 essais)
4. **Entraînement et évaluation** — entraînement en 2 phases (tête gelée puis fine-tuning), évaluation sur le test set

## 3. Dataset

| Propriété | Valeur |
|---|---|
| Source | Dataset Kaggle (`daikhiyassine/dataset`) |
| Total après nettoyage | 21 010 images |
| Répartition | Train 14 707 / Val 3 151 / Test 3 152 (≈70/15/15) |
| Classes | Équilibrées (~24-26% par classe) |
| Résolution de retraitement | 256×256 (marge pour data augmentation), entrée modèle 224×224 |
| Nettoyage appliqué | 3 doublons exacts retirés, 3 fuites train/val/test corrigées, 626 images converties en RGB |

Détails complets : voir `01_Dataset/rapport_nettoyage_final_b0.json`

## 4. Configuration du modèle

Architecture de la tête de classification (après la base EfficientNetB0) :

```
EfficientNetB0 (ImageNet, base convolutive)
      ↓
GlobalAveragePooling2D
      ↓
Dense(1024, activation="relu")
      ↓
Dropout(0.5)
      ↓
Dense(4, activation="softmax")
```

**Meilleurs hyperparamètres** (issus de la recherche bayésienne, `val_accuracy` best = 0.9295) :

| Hyperparamètre | Valeur |
|---|---|
| Unités denses | 1024 |
| Dropout | 0.5 |
| Learning rate (phase 1) | 1e-4 |
| Couches dégelées (fine-tuning) | 20 dernières couches |
| BatchNormalization | Non |
| Optimiseur | Adam |

Détails complets : voir `02_Hyperparametres/best_hyperparameters_b0.json`

## 5. Entraînement

Entraînement réalisé en deux phases, méthodologie identique à celle utilisée pour EfficientNetB2 :

- **Phase 1** — base EfficientNetB0 entièrement gelée, entraînement de la tête de classification uniquement (jusqu'à 15 epochs, arrêt anticipé sur `val_loss`)
- **Phase 2** — fine-tuning des 20 dernières couches de la base, avec un learning rate réduit d'un facteur 10 (jusqu'à 20 epochs, arrêt anticipé sur `val_loss`)

Callbacks utilisés : `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint` (meilleur modèle sauvegardé sur `val_accuracy`).

Historique complet des courbes loss/accuracy par epoch : voir `04_Resultats/training_log_b0_phase1.csv` et `phase2.csv`.

## 6. Résultats sur le test set (3 152 images)

### 6.1 Métriques globales

| Métrique | Valeur |
|---|---|
| **Accuracy** | **92.32 %** |
| **Loss** | 0.2164 |
| **AUC (Keras, multi-label)** | 0.9874 |
| **AUC macro (sklearn, one-vs-rest)** | 0.9889 |

### 6.2 Résultats détaillés par classe

| Classe | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Corrosion | 96.66 % | 92.54 % | 94.55 % | 844 |
| Cracks (fissures) | 88.38 % | 89.10 % | 88.74 % | 734 |
| Normal | 100.00 % | 99.61 % | 99.81 % | 771 |
| Spalling (écaillage) | 84.57 % | 88.04 % | 86.27 % | 803 |

**Moyennes globales**
- Macro avg (F1) : 92.34 %
- Weighted avg (F1) : 92.37 %

### 6.3 Analyse

**Points forts**
- La classe **normal** est reconnue quasi parfaitement (F1 = 99.81 %, precision 100 %) — aucun faux positif sur cette classe.
- La classe **corrosion** reste très bien détectée (F1 = 94.55 %), malgré un modèle presque 2x plus léger que B2.
- L'AUC macro de 0.9889 confirme une excellente séparabilité globale entre les 4 classes malgré la taille réduite du modèle.

**Point faible : confusion cracks / spalling**
- Comme pour EfficientNetB2, ce sont les deux classes les plus difficiles à distinguer (F1 = 88.74 % et 86.27 %), avec `spalling` en retrait (precision 84.57 %, la plus basse de tout le tableau).
- Ce résultat confirme l'hypothèse déjà posée sur B2 : une possible ambiguïté de labellisation à la source entre ces deux classes (2 cas d'étiquetage contradictoire déjà détectés lors du nettoyage, voir `01_Dataset/rapport_nettoyage_final_b0.json`), plutôt qu'une faiblesse propre à l'un ou l'autre modèle.
- Voir `04_Resultats/confusion_matrix_normalized_efficientnetb0.png` pour visualiser précisément où se concentrent les erreurs.

Matrices de confusion disponibles : `04_Resultats/confusion_matrix_efficientnetb0.png` et `confusion_matrix_normalized_efficientnetb0.png`

## 7. Comparaison avec les autres modèles du projet

| Modèle | Paramètres | Résolution | Accuracy (test) | AUC macro |
|---|---|---|---|---|
| ResNet50 (Hamza) | ~25M | 224×224 | ~95.66 % | — |
| EfficientNetB2 | ~7.7M | 260×260 | 93.05 % | 0.9909 |
| **EfficientNetB0** | **~4M** | **224×224** | **92.32 %** | **0.9889** |

**Lecture** : EfficientNetB0 perd seulement **0.73 point d'accuracy** par rapport à EfficientNetB2, alors qu'il utilise environ **2 fois moins de paramètres** et des images plus petites (224 vs 260). L'écart entre les deux modèles EfficientNet est faible sur toutes les classes, y compris sur les classes les plus difficiles (`cracks`, `spalling`), ce qui suggère que la difficulté vient davantage de la qualité/cohérence des labels sur ces deux classes que de la capacité du modèle.

Pour un déploiement embarqué ou mobile où la vitesse d'inférence et la taille du modèle comptent (fichier `.tflite` plus petit, moins de calculs), **EfficientNetB0 constitue un excellent compromis** face à la perte de précision minime observée.

## 8. Contenu du dossier

```
EfficientNetB0/
├── 01_Dataset/
│   └── rapport_nettoyage_final_b0.json
├── 02_Hyperparametres/
│   └── best_hyperparameters_b0.json
├── 03_Modeles/
│   ├── best_model_b0_phase1.keras
│   ├── best_model_b0_phase2.keras
│   ├── aigis_efficientnetb0_final.keras
│   └── aigis_efficientnetb0.tflite
├── 04_Resultats/
│   ├── distribution_classes_b0.png
│   ├── echantillons_dataset_b0.png
│   ├── test_results_efficientnetb0.json
│   ├── confusion_matrix_efficientnetb0.png
│   ├── confusion_matrix_normalized_efficientnetb0.png
│   ├── training_log_b0_phase1.csv
│   └── training_log_b0_phase2.csv
└── README.md
```

## 9. Prochaines étapes

1. Comparer le temps d'inférence réel (.tflite) entre EfficientNetB0, EfficientNetB2 et ResNet50 sur du matériel représentatif du déploiement final
2. Examiner manuellement un échantillon d'images `cracks`/`spalling` mal classées pour confirmer l'hypothèse d'ambiguïté de labellisation à la source
3. Envisager une révision ciblée des labels `cracks`/`spalling` dans le dataset source, au-delà des doublons déjà corrigés
4. Décider du modèle à retenir pour la suite du projet (B0 pour la légèreté, B2 pour la précision maximale, ou ResNet50 comme référence) selon les contraintes de déploiement définies avec le superviseur

---
*Pipeline exécuté sur Kaggle (GPU P100/T4), projet Aigis — ESPRIT.*
