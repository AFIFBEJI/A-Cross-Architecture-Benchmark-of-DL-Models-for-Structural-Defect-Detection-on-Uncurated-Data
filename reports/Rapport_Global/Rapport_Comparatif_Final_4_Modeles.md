# Aigis — Détection de Défauts Structurels en Béton
## Rapport Comparatif Final — 4 Modèles

**Projet :** Aigis — Pipeline de deep learning pour la détection automatique de défauts structurels (corrosion, fissures, écaillage) sur infrastructures en béton
**Stage :** ESPRIT — Recherche en vision par ordinateur
**Auteur :** Yacin
**Date :** Août 2026
**Réunion de suivi prévue :** semaine du 17 août 2026

---

## 1. Objectif de ce rapport

Ce document synthétise et compare les performances des **4 modèles** entraînés et évalués dans des conditions strictement identiques (même dataset nettoyé de 21 010 images, même split train/val/test 70/15/15), conformément à la demande du superviseur (email du 09/08/2026, point 2 : *"Reproduire... les résultats de l'étude précédente en évaluant les différents modèles sur le même dataset... afin de disposer d'une base de comparaison fiable et rigoureuse"*).

## 2. Protocole expérimental commun

| Élément | Détail |
|---|---|
| Dataset | `daikhiyassine/dataset` (Kaggle), nettoyé (21 010 images après suppression de 3 doublons dont 2 cas d'étiquetage contradictoire) |
| Classes | corrosion, cracks, spalling, normal (4 classes équilibrées, ~24-26% chacune) |
| Split | Train 14 707 / Val 3 151 / Test 3 152 (70/15/15) |
| Recherche d'hyperparamètres | Bayésienne (Keras Tuner, 10 essais) pour B0/B2/ResNet50 ; génétique (Ultralytics, 15 itérations) pour YOLO26 |
| Stratégies anti-overfitting | Data augmentation (flip, rotation, zoom, contraste), Dropout, EarlyStopping, ReduceLROnPlateau, entraînement en 2 phases (base gelée puis fine-tuning partiel) |
| Métriques | Accuracy, Precision, Recall, F1-score (par classe et moyennes macro/weighted), AUC macro (one-vs-rest) |
| Infrastructure | Kaggle Notebooks, GPU P100/T4, TensorFlow/Keras (B0/B2/ResNet50), PyTorch/Ultralytics (YOLO26) |

## 3. Tableau comparatif global

| Modèle | Paramètres | Résolution | Accuracy (test) | AUC macro | Vitesse d'inférence |
|---|---|---|---|---|---|
| **YOLO26s-cls** 🏆 | ~5.4M | 224×224 | **94.70 %** | **0.9942** | 0.9 ms/image |
| ResNet50 (réentraîné) | ~25M | 224×224 | 93.69 % | 0.9908 | — |
| EfficientNetB2 | ~7.7M | 260×260 | 93.05 % | 0.9909 | — |
| EfficientNetB0 | ~4M | 224×224 | 92.32 % | 0.9889 | — |

**YOLO26 obtient la meilleure performance globale**, avec **4.6 fois moins de paramètres** que ResNet50 tout en le dépassant de 1.01 point d'accuracy.

## 4. Résultats détaillés par classe et par modèle

### 4.1 Classe `corrosion`

| Modèle | Precision | Recall | F1-score |
|---|---|---|---|
| YOLO26s-cls | 97.82 % | 95.50 % | 96.64 % |
| ResNet50 | 96.01 % | 94.08 % | 95.03 % |
| EfficientNetB2 | 97.28 % | 93.13 % | 95.16 % |
| EfficientNetB0 | 96.66 % | 92.54 % | 94.55 % |

### 4.2 Classe `cracks`

| Modèle | Precision | Recall | F1-score |
|---|---|---|---|
| YOLO26s-cls | 91.40 % | 92.64 % | 92.02 % |
| ResNet50 | 90.28 % | 92.37 % | 91.31 % |
| EfficientNetB2 | 88.48 % | 91.01 % | 89.72 % |
| EfficientNetB0 | 88.38 % | 89.10 % | 88.74 % |

### 4.3 Classe `normal`

| Modèle | Precision | Recall | F1-score |
|---|---|---|---|
| YOLO26s-cls | 100.00 % | 99.74 % | 99.87 % |
| ResNet50 | 100.00 % | 99.87 % | 99.94 % |
| EfficientNetB2 | 99.87 % | 99.87 % | 99.87 % |
| EfficientNetB0 | 100.00 % | 99.61 % | 99.81 % |

### 4.4 Classe `spalling`

| Modèle | Precision | Recall | F1-score |
|---|---|---|---|
| YOLO26s-cls | 89.57 % | 90.91 % | 90.23 % |
| ResNet50 | 88.43 % | 88.54 % | 88.49 % |
| EfficientNetB2 | 86.67 % | 88.29 % | 87.48 % |
| EfficientNetB0 | 84.57 % | 88.04 % | 86.27 % |

## 5. Analyse et discussion critique

### 5.1 Tendance générale observée

Sur les **4 modèles indépendants**, testés avec des architectures et écosystèmes d'entraînement différents (TensorFlow/Keras pour B0/B2/ResNet50, PyTorch/Ultralytics pour YOLO26), un même schéma se répète systématiquement :

1. **`normal` est quasi parfaitement classée** par tous les modèles (F1 > 99.8% partout) — le modèle distingue très facilement une surface saine d'une surface endommagée.
2. **`corrosion` est bien classée** (F1 entre 94.5% et 96.6%).
3. **`cracks` et surtout `spalling` restent les classes les plus difficiles** pour tous les modèles, avec `spalling` systématiquement la plus faible (F1 entre 86.3% et 90.2%).

### 5.2 Interprétation : limite du modèle ou du dataset ?

La récurrence de cette même faiblesse sur **4 architectures totalement différentes** (allant de 4M à 25M paramètres, deux frameworks distincts) suggère fortement que la cause n'est **pas architecturale**, mais liée à la **qualité de l'annotation du dataset source**.

Preuve concrète : lors du nettoyage du dataset, **2 cas d'images strictement identiques mais étiquetées différemment** ont été détectés entre les classes `spalling` et `cracks` (voir `rapport_nettoyage_final.json`). Si de tels cas existent parmi les doublons détectables, il est probable que des cas similaires existent aussi parmi les images uniques (non détectables par simple hash), contribuant au bruit d'étiquetage dans ces deux classes.

**Recommandation** : une révision manuelle d'un échantillon d'images `cracks`/`spalling` mal classées par plusieurs modèles à la fois (erreurs communes) permettrait de vérifier cette hypothèse.

### 5.3 Comparaison avec le ResNet50 original (Hamza)

Le ResNet50 réentraîné dans ce rapport (93.69%) est **inférieur** au ResNet50 original de l'étude précédente (~95.66%, sur le dataset brut non nettoyé). Cet écart n'indique pas une régression, mais s'explique par la méthodologie plus rigoureuse appliquée ici :

- Le dataset original contenait 3 doublons, dont **3 cas de fuite entre train/val/test** — un même contenu pouvant apparaître à la fois dans les données d'entraînement et de test, ce qui gonfle artificiellement les métriques de validation.
- Le score de 93.69% obtenu ici, sur un dataset nettoyé sans fuite, est donc **plus représentatif de la capacité de généralisation réelle** du modèle.

Ce point constitue un exemple concret à mettre en avant dans la discussion critique demandée par le superviseur (point 3 de l'email) : un chiffre plus bas peut refléter une méthodologie plus rigoureuse plutôt qu'une moins bonne performance.

### 5.4 Avantages et limites de chaque modèle

| Modèle | Avantages | Limites |
|---|---|---|
| **YOLO26s-cls** | Meilleure accuracy globale, très léger (~5.4M param.), inférence très rapide (0.9ms), écosystème natif orienté déploiement edge | Écosystème différent (PyTorch/Ultralytics) à maintenir en parallèle de TensorFlow, export TFLite non finalisé (problème de compatibilité de version rencontré) |
| **ResNet50** | Bonne performance, architecture éprouvée et largement documentée dans la littérature | Le plus lourd des 4 modèles (~25M param.), inférence plus coûteuse, peu adapté à l'embarqué |
| **EfficientNetB2** | Bon compromis précision/taille, entrée plus grande (260×260) capte plus de détails | Légèrement moins performant que YOLO26 et ResNet50, plus lourd que B0 |
| **EfficientNetB0** | Le plus léger des modèles TensorFlow testés (~4M param.), rapide à entraîner | Accuracy la plus faible des 4, notamment sur `spalling` |

## 6. Recommandation

Sur la base de ces résultats, **YOLO26s-cls constitue le meilleur candidat actuel** pour la suite du projet, en particulier compte tenu de l'objectif à terme de déploiement sur plateformes embarquées (drones/UAV, mobile — cf. point 4 de l'email du superviseur sur la veille technologique). Il combine :
- La meilleure précision (94.70%)
- Le meilleur AUC macro (0.9942)
- Un nombre de paramètres très inférieur à ResNet50
- Une vitesse d'inférence native optimisée

**ResNet50** reste une référence utile comme modèle de comparaison "haute précision, sans contrainte de taille".

## 7. Limites de l'étude et pistes d'amélioration

1. **Confusion `cracks`/`spalling` non résolue** — nécessite une investigation de la qualité d'étiquetage du dataset source, au-delà des doublons déjà corrigés.
2. **Export TFLite incomplet pour YOLO26** — à finaliser sur un environnement avec PyTorch à jour (sans contrainte de compatibilité GPU P100).
3. **Comparaison de vitesse d'inférence incomplète** — seul YOLO26 a une mesure précise ; il manque les benchmarks comparables pour B0/B2/ResNet50 sur le même matériel.
4. **Élargissement du dataset en cours** — plusieurs datasets externes ont été identifiés et vérifiés (licences confirmées : Historical_Building_Crack_2019, BFDD, Damage Detection Dataset, Surface Crack Detection — voir `Suivi_Nouveaux_Datasets.xlsx`), une nouvelle itération d'entraînement sur dataset élargi permettra de vérifier si l'ajout de données réduit la confusion cracks/spalling.

## 8. Prochaines étapes

1. Finaliser la fusion du dataset élargi (v2) avec les nouvelles images collectées et validées par le superviseur
2. Ré-entraîner les 4 modèles sur le dataset élargi et comparer l'évolution des performances, notamment sur `cracks`/`spalling`
3. Compléter les benchmarks de vitesse d'inférence sur un matériel de référence commun
4. Finaliser l'export TFLite de YOLO26
5. Présenter ces résultats au superviseur lors de la réunion de la semaine du 17 août

---
*Rapport généré à partir des évaluations sur Kaggle (GPU P100/T4), pipeline Aigis — ESPRIT.*
