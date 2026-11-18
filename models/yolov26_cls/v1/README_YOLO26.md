# Aigis — Détection de Défauts Structurels en Béton
## Rapport Final — Modèle YOLO26 (classification)

**Projet :** Aigis — Pipeline de deep learning pour la détection automatique de défauts structurels (corrosion, fissures, écaillage) sur infrastructures en béton
**Stage :** ESPRIT — Recherche en vision par ordinateur
**Auteur :** Yacin
**Architecture :** YOLO26s-cls (Ultralytics, transfer learning ImageNet)
**Statut :** ✅ Entraînement et évaluation terminés

---

## 1. Objectif

Ce dossier regroupe l'ensemble du pipeline et des résultats pour **YOLO26**, la dernière génération de modèles Ultralytics (sortie janvier 2026), testée ici en variante **classification** (`-cls`) sur les 4 classes du projet Aigis :

| Classe | Description |
|---|---|
| `corrosion` | Corrosion visible (armatures, taches de rouille) |
| `cracks` | Fissures |
| `spalling` | Écaillage / éclatement du béton |
| `normal` | Surface saine, sans défaut |

YOLO26 a été retenu pour évaluer une architecture différente de la famille EfficientNet/ResNet déjà testée, avec un écosystème d'entraînement natif (Ultralytics) optimisé pour le déploiement edge/mobile (inférence NMS-free, export multi-format).

## 2. Pipeline

1. **Visualisation du dataset** — distribution des classes, échantillons visuels par classe
2. **Nettoyage des données** — détection des doublons, vérification des fuites entre splits, conversion RGB, resize et compression (identique aux autres modèles du projet)
3. **Recherche d'hyperparamètres** — algorithme génétique natif Ultralytics (`model.tune()`, 15 itérations, modèle nano pour la recherche)
4. **Entraînement final** — `yolo26s-cls`, 50 epochs max avec early stopping (patience=10), hyperparamètres issus du tuning
5. **Évaluation** — validation native Ultralytics + rapport de classification détaillé sur le test set

## 3. Dataset

| Propriété | Valeur |
|---|---|
| Source | Dataset Kaggle (`daikhiyassine/dataset`) |
| Total après nettoyage | 21 010 images |
| Répartition | Train 14 707 / Val 3 151 / Test 3 152 (≈70/15/15) |
| Résolution de retraitement | 256×256, entrée modèle 224×224 |
| Nettoyage appliqué | 3 doublons exacts retirés (dont 2 cas d'étiquetage contradictoire `spalling`/`cracks`), 3 fuites train/val/test corrigées, 626 images converties en RGB |

Détails complets : voir `01_Dataset/rapport_nettoyage_final_yolo26.json`

## 4. Configuration du modèle

| Propriété | Valeur |
|---|---|
| Architecture | YOLO26s-cls (5,439,252 paramètres, 47 couches, 12.0 GFLOPs) |
| Backbone | C3k2 / C2PSA (architecture native YOLO26) |
| Entrée | 224×224×3 |
| Poids initiaux | ImageNet (pré-entraîné) |

**Meilleurs hyperparamètres** (recherche génétique, 15 itérations) :

| Hyperparamètre | Valeur |
|---|---|
| Learning rate initial (lr0) | 0.00056 |
| Learning rate final (lrf) | 0.01 |
| Momentum | 0.9008 |
| Weight decay | 0.00098 |
| Warmup epochs | 1.93 |
| Mosaic augmentation | 0.853 |
| HSV saturation | 0.812 |
| Optimiseur | AdamW |

Détails complets : voir `02_Hyperparametres/best_hyperparameters_yolo26.json` et `args.yaml`

## 5. Entraînement

Entraînement en une phase (contrairement à l'approche 2 phases de B0/B2), avec fine-tuning direct de l'ensemble du réseau — méthodologie native à l'écosystème Ultralytics. Arrêt anticipé (`patience=10`) sur absence d'amélioration.

Historique complet des courbes loss/accuracy par epoch : voir `04_Resultats/results.csv` et `results.png`.

## 6. Résultats sur le test set (3 152 images)

### 6.1 Métriques globales

| Métrique | Valeur |
|---|---|
| **Top-1 Accuracy** | **94.70 %** |
| **Top-5 Accuracy** | 100.00 % |
| **AUC macro (sklearn, one-vs-rest)** | 0.9942 |
| Vitesse d'inférence | 0.9 ms/image (GPU P100) |

### 6.2 Résultats détaillés par classe

| Classe | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Corrosion | 97.82 % | 95.50 % | 96.64 % | 844 |
| Cracks (fissures) | 91.40 % | 92.64 % | 92.02 % | 734 |
| Normal | 100.00 % | 99.74 % | 99.87 % | 771 |
| Spalling (écaillage) | 89.57 % | 90.91 % | 90.23 % | 803 |

**Moyennes globales**
- Macro avg (F1) : 94.69 %
- Weighted avg (F1) : 94.72 %

### 6.3 Analyse

**Points forts**
- **Meilleur résultat global des trois modèles testés dans ce projet** (94.70% vs 93.05% pour B2 et 92.32% pour B0), malgré un modèle de taille intermédiaire (~5.4M paramètres, entre B0 et B2).
- Vitesse d'inférence remarquable : **0.9 ms/image** sur GPU P100, cohérent avec l'architecture NMS-free et l'optimisation edge native de YOLO26.
- Amélioration nette sur toutes les classes par rapport à B0 et B2, y compris sur les classes historiquement difficiles.

**Point faible persistant : confusion cracks / spalling**
- Comme observé sur B0 et B2, `cracks` (F1 = 92.02%) et surtout `spalling` (F1 = 90.23%) restent les deux classes les plus faibles, bien qu'améliorées par rapport aux modèles précédents.
- Ce résultat confirme, sur un **troisième modèle indépendant**, l'hypothèse d'une ambiguïté de labellisation à la source entre ces deux classes plutôt qu'une limite architecturale — les 2 cas d'étiquetage contradictoire détectés lors du nettoyage (`spalling` vs `cracks`) restent le point d'attention prioritaire pour améliorer encore ces scores.

Matrices de confusion disponibles : `04_Resultats/confusion_matrix_yolo26.png` et `confusion_matrix_normalized_yolo26.png` (sur test set — à ne pas confondre avec les matrices générées automatiquement par Ultralytics sur le val set)

## 7. Comparaison avec les autres modèles du projet

| Modèle | Paramètres | Résolution | Accuracy (test) | AUC macro |
|---|---|---|---|---|
| ResNet50 (Hamza) | ~25M | 224×224 | ~95.66 % | — |
| **YOLO26s-cls** | **~5.4M** | **224×224** | **94.70 %** | **0.9942** |
| EfficientNetB2 | ~7.7M | 260×260 | 93.05 % | 0.9909 |
| EfficientNetB0 | ~4M | 224×224 | 92.32 % | 0.9889 |

**Lecture** : YOLO26 offre le **meilleur compromis précision/légèreté** du projet parmi les modèles entraînés from scratch en interne — il approche la performance de ResNet50 (25M paramètres) avec **~4.6 fois moins de paramètres**, tout en dépassant nettement les deux modèles EfficientNet testés. Combiné à sa vitesse d'inférence native (0.9 ms/image) et son export multi-format (ONNX validé, TFLite à finaliser), YOLO26 apparaît comme le candidat le plus solide pour un déploiement en production à ce stade du projet.

## 8. Points techniques rencontrés (pour référence future)

- **Incompatibilité GPU P100 / PyTorch récent** : PyTorch 2.10+ a abandonné le support de l'architecture Pascal (sm_60) du P100. Solution appliquée : downgrade vers `torch==2.4.1+cu121` + `torchvision==0.19.1+cu121`. Alternative plus stable : utiliser un GPU T4 (architecture Turing, sm_75), nativement compatible sans downgrade.
- **Export TFLite en échec** : l'auto-update des dépendances d'export (`litert-torch`, `torchao 0.17`) nécessite PyTorch ≥ 2.11, incompatible avec le downgrade ci-dessus. **Solution de contournement** : export ONNX réussi (`best.onnx`, 20.8 MB) en alternative ; export TFLite à refaire sur une session T4 avec PyTorch par défaut (2.10+), sans downgrade.

## 9. Contenu du dossier

```
YOLO26/
├── 01_Dataset/
│   └── rapport_nettoyage_final_yolo26.json
├── 02_Hyperparametres/
│   ├── best_hyperparameters_yolo26.json
│   └── args.yaml
├── 03_Modeles/
│   ├── best.pt
│   └── best.onnx
├── 04_Resultats/
│   ├── distribution_classes_yolo26.png
│   ├── echantillons_dataset_yolo26.png
│   ├── test_results_yolo26.json
│   ├── confusion_matrix_yolo26.png
│   ├── confusion_matrix_normalized_yolo26.png
│   ├── results.csv
│   └── results.png
└── README.md
```

## 10. Prochaines étapes

1. Finaliser l'export TFLite (session T4, PyTorch par défaut, sans downgrade) pour compléter la comparaison de taille/vitesse d'inférence sur mobile
2. Comparer précisément les temps d'inférence des 4 modèles sur un même matériel de référence
3. Examiner manuellement les erreurs `cracks`/`spalling` (communes aux 3 modèles internes) pour confirmer la piste d'un problème de labellisation à la source
4. Recommandation à ce stade pour le superviseur : **YOLO26 comme modèle candidat principal**, avec ResNet50 comme référence de précision maximale si la taille du modèle n'est pas contrainte

---
*Pipeline exécuté sur Kaggle (GPU P100), projet Aigis — ESPRIT.*
