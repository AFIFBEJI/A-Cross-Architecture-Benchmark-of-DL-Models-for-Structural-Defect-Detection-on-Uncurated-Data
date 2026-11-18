# Recherche Bibliographique et État de l'Art

## Classification de défauts structurels en béton par apprentissage profond, avec perspective de déploiement embarqué

**Projet :** Aigis — ESPRIT
**Auteur :** Yacin
**Date :** Août 2026

---

## 1. Introduction et contexte scientifique

Le projet Aigis porte sur la **classification automatique de défauts structurels** sur infrastructures en béton (corrosion, fissures, écaillage/spalling, surfaces saines) à partir d'images, en vue d'un futur déploiement sur plateformes à ressources limitées (drones/UAV, appareils mobiles). Contrairement à un problème de détection d'objets avec localisation (bounding boxes), la tâche traitée ici attribue **une étiquette de classe à l'image entière**.

Cette revue de littérature vise à :
1. Synthétiser l'état de l'art des architectures de classification d'image appliquées à l'inspection de défauts structurels
2. Identifier les architectures légères adaptées à un déploiement embarqué
3. Mettre en évidence les research gaps que le projet Aigis peut contribuer à combler

Une section complémentaire (§5) couvre les architectures de **détection d'objets** identifiées dans une veille initiale — pertinentes si le projet évolue vers une localisation précise des défauts sur l'image, au-delà de la classification globale actuelle.

## 2. Classification de défauts structurels : état de l'art

### 2.1 Architectures CNN classiques (ResNet, VGG)

ResNet50, largement utilisé comme référence dans la littérature sur l'inspection structurelle, sert de base de comparaison dans de nombreuses études de détection de fissures et corrosion par transfer learning. Dans le cadre du projet Aigis, un ResNet50 réentraîné sur le dataset nettoyé (21 010 images, 4 classes) atteint **93.69% d'accuracy** sur le test set, avec un AUC macro de 0.9908.

### 2.2 EfficientNet : compromis précision/paramètres

La famille EfficientNet (Tan & Le, 2019, arXiv:1905.11946) introduit le *compound scaling*, équilibrant profondeur, largeur et résolution du réseau. Sur le dataset Aigis :
- **EfficientNetB0** (~4M paramètres, 224×224) : 92.32% accuracy, AUC macro 0.9889
- **EfficientNetB2** (~7.7M paramètres, 260×260) : 93.05% accuracy, AUC macro 0.9909

La version EfficientNetV2 (Tan & Le, 2021, arXiv:2104.00298) améliore la vitesse d'entraînement via des blocs Fused-MBConv, avec des accuracies ImageNet officielles de 83.9% (V2-S, ~22M paramètres) et 85.1% (V2-M, ~54M paramètres) — non testée expérimentalement dans Aigis à ce stade, mais candidate pour extension future.

### 2.3 YOLO en mode classification (`-cls`)

Bien que la famille YOLO soit historiquement associée à la détection d'objets, Ultralytics propose depuis plusieurs générations une variante **classification pure** (`-cls`), qui réutilise le backbone du détecteur sans tête de régression de boîtes.

**YOLO26** (Sapkota et al., 2025, arXiv:2509.25164), annoncé en septembre 2025 lors de YOLO Vision 2025 (Londres) et officiellement lancé par Ultralytics le **14 janvier 2026**, introduit plusieurs innovations : suppression de la Distribution Focal Loss (DFL), inférence end-to-end sans NMS, optimiseur hybride MuSGD, et jusqu'à 43% d'inférence plus rapide sur CPU par rapport à YOLO11n. Sur le dataset Aigis, la variante **YOLO26s-cls** (~5.4M paramètres) atteint le **meilleur résultat du projet : 94.70% accuracy, AUC macro 0.9942**, avec une vitesse d'inférence de 0.9 ms/image sur GPU P100.

⚠️ **Licence** : YOLO26 est distribué sous licence **AGPL-3.0**, impliquant soit l'open-source du code dérivé, soit l'achat d'une licence commerciale pour un usage propriétaire fermé — point à considérer pour un déploiement commercial futur.

### 2.4 Synthèse comparative — résultats expérimentaux Aigis

| Modèle | Paramètres | Résolution | Accuracy (test) | AUC macro |
|---|---|---|---|---|
| **YOLO26s-cls** 🏆 | ~5.4M | 224×224 | **94.70 %** | **0.9942** |
| ResNet50 (réentraîné) | ~25M | 224×224 | 93.69 % | 0.9908 |
| EfficientNetB2 | ~7.7M | 260×260 | 93.05 % | 0.9909 |
| EfficientNetB0 | ~4M | 224×224 | 92.32 % | 0.9889 |

Ce résultat est cohérent avec la tendance identifiée dans la littérature récente : les architectures de la famille YOLO, initialement conçues pour la détection temps réel, transposent efficacement leurs optimisations d'inférence (backbone léger, entraînement optimisé) à la tâche de classification pure, tout en conservant un nombre de paramètres réduit favorable à l'embarqué.

## 3. Difficultés récurrentes identifiées dans la littérature et confirmées expérimentalement

### 3.1 Confusion entre classes de dommages visuellement proches

Plusieurs études sur la classification de défauts structurels rapportent une confusion récurrente entre les classes de dommages aux frontières visuelles ambiguës (par exemple fissures profondes vs débuts d'écaillage). Ce phénomène est confirmé expérimentalement dans Aigis : sur les **4 modèles indépendants testés** (architectures et frameworks différents), les classes `cracks` et surtout `spalling` restent systématiquement les plus difficiles à classifier (F1-score entre 86.3% et 92.0%), contre plus de 99.8% pour la classe `normal`.

L'analyse du dataset source a révélé des cas concrets d'**étiquetage contradictoire** (images identiques classées différemment entre `spalling` et `cracks`), suggérant que cette difficulté relève davantage de la qualité d'annotation du dataset source que d'une limite architecturale — un point qui rejoint les préoccupations de la littérature sur la fiabilité des datasets d'inspection structurelle publics.

### 3.2 Écart de généralisation entre données de référence et conditions réelles

La littérature souligne un écart de performance fréquent entre les résultats obtenus sur dataset contrôlé et les conditions de terrain (domain shift). Dans les travaux préliminaires du projet Aigis, un article de référence rapportait une chute de 68.01% à 58.64% d'accuracy (AUC de 0.959 à 0.879) entre dataset original et conditions de domain shift — écart qui justifie l'importance accordée dans Aigis au nettoyage rigoureux du dataset (suppression des fuites train/val/test, déduplication) pour obtenir des métriques représentatives de la capacité de généralisation réelle.

## 4. Research gaps identifiés

1. **Manque d'évaluations comparatives standardisées** entre architectures de classification (EfficientNet, ResNet, YOLO-cls) sur un même dataset d'inspection de défauts béton — le protocole expérimental d'Aigis (même dataset, même split, mêmes métriques pour 4 modèles) contribue directement à combler ce manque.
2. **Qualité et cohérence de l'étiquetage** des datasets publics de défauts structurels, particulièrement entre classes visuellement proches (cracks/spalling) — peu documentée dans la littérature existante malgré son impact démontré sur les performances.
3. **Compromis précision/taille/vitesse pour le déploiement embarqué** appliqué spécifiquement à l'inspection structurelle (par opposition aux benchmarks génériques COCO/ImageNet) — axe qu'Aigis commence à documenter via la comparaison YOLO26s-cls vs modèles plus lourds.

## 5. Complément — architectures de détection d'objets (veille prospective)

Cette section recense les architectures de **détection d'objets** (avec localisation par bounding box) identifiées lors d'une veille initiale. Elle ne correspond pas à la tâche actuelle d'Aigis (classification globale) mais constitue une piste pertinente si le projet évolue vers la **localisation précise** des défauts sur l'image.

| Architecture | Points forts | Limites | Référence |
|---|---|---|---|
| YOLOv6-N/S/M/L | Gamme étendue, très haute vitesse (jusqu'à 1187 FPS sur T4 pour le N) | Optimisation embarqué à approfondir | Li et al., 2023, arXiv:2301.05586 |
| YOLO-NAS S/M/L | Recherche d'architecture automatisée (NAS), quantification INT8 optimisée (perte <0.65 point mAP) | Licence des poids pré-entraînés à vérifier ; maintenance arrêtée depuis le rachat de Deci par NVIDIA | Deci AI, GitHub Deci-AI/super-gradients |
| LeYOLO | Ultra-léger, ciblé microcontrôleurs | Encore peu validé en conditions réelles | Hollard et al., 2024-2025, arXiv:2406.14239 |
| YOLO26 (mode détection) | NMS-free, multi-tâches, edge-first | Très récent, recul limité en production | Sapkota et al., 2025, arXiv:2509.25164 |
| YOLOv11n | Meilleur rapport mAP/taille quantifiée (2.8 Mo) | Moins de variantes disponibles | arXiv:2507.18174 |

**Note de cohérence** : YOLOv6-N cité dans cette section rapporte 37.5% AP à 1187 FPS sur T4 — chiffres issus du papier **YOLOv6 v3.0** (arXiv:2301.05586), à ne pas confondre avec le papier YOLOv6 original (arXiv:2209.02976) qui rapportait des valeurs différentes (35.9% AP, 1234 FPS).

## 6. Conclusion

La littérature récente confirme la pertinence des architectures légères de la famille YOLO et EfficientNet pour la classification/détection de défauts structurels dans des contextes à ressources contraintes. Les résultats expérimentaux obtenus dans Aigis corroborent cette tendance : **YOLO26s-cls** surpasse ResNet50 (25M paramètres) avec seulement 5.4M paramètres, confirmant l'intérêt des architectures récentes pour un futur déploiement embarqué (drones/UAV).

Les principales limites identifiées — confusion entre classes visuellement proches et sensibilité à la qualité d'étiquetage — orientent les prochaines étapes du projet vers une révision ciblée du dataset et une extension à des sources externes vérifiées (voir `Suivi_Nouveaux_Datasets.xlsx`).

## 7. Bibliographie

1. Tan, M. & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. *ICML*. arXiv:1905.11946
2. Tan, M. & Le, Q. (2021). EfficientNetV2: Smaller Models and Faster Training. arXiv:2104.00298
3. Sapkota, R., Cheppally, R.H., Sharda, A., Karkee, M. (2025). YOLO26: Key Architectural Enhancements and Performance Benchmarking for Real-Time Object Detection. arXiv:2509.25164
4. Li, C. et al. (2023). YOLOv6 v3.0: A Full-Scale Reloading. arXiv:2301.05586
5. Li, C. et al. (2022). YOLOv6: A Single-Stage Object Detection Framework for Industrial Applications. arXiv:2209.02976
6. Hollard, L. et al. (2024-2025). LeYOLO, New Embedded Architecture for Object Detection. *Proceedings of the Conference on Robots and Vision*. arXiv:2406.14239
7. Deci AI. YOLO-NAS — A Next-Generation Object Detection Model. GitHub: Deci-AI/super-gradients
8. Ultralytics (2026). Ultralytics YOLO26: The New Standard for Edge-First Vision AI. ultralytics.com/blog
9. Guo, P. & Bao, Y. (2025). MDMCS — Multi-Damage Monitoring of Concrete Structures, v2. Mendeley Data. doi:10.17632/6x4dzzrs2h.2
10. Özgenel, Ç.F. (2019). Concrete Crack Images for Classification. Mendeley Data, v2. doi:10.17632/5y9wdsg2zt.2

---
*Document préparé dans le cadre du stage de recherche Aigis — ESPRIT.*
