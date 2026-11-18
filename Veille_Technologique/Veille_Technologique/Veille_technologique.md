# Veille Technologique

## Architectures légères pour la classification de défauts structurels sur plateformes embarquées

**Projet :** Aigis — ESPRIT
**Auteur :** Yacin
**Date :** Août 2026

---

## 1. Contexte et problématique

Le projet Aigis porte sur la sélection et l'évaluation d'architectures de **classification d'image** (défauts structurels en béton : corrosion, cracks, spalling, normal) répondant à un double objectif : haute précision et compatibilité avec des plateformes à ressources limitées (drones/UAV, appareils mobiles, systèmes embarqués).

Le dataset **Aigis** (21 010 images nettoyées, 4 classes, extensible à ~27 000+ via l'intégration de sources externes vérifiées) constitue le socle expérimental. Cette veille identifie les modèles les plus pertinents pour la classification embarquée, issus de recherches récentes disponibles sur Hugging Face, GitHub et Papers With Code, en portant une attention particulière au compromis précision/vitesse/consommation mémoire.

**Note de cohérence avec la recherche bibliographique** : cette veille se concentre sur des architectures de **classification** (comme le reste du projet), avec une section complémentaire sur les architectures de détection pour référence future.

## 2. Modèles évalués expérimentalement dans Aigis

| Modèle | Paramètres | Résolution | Accuracy (test) | AUC macro | Statut |
|---|---|---|---|---|---|
| **YOLO26s-cls** 🏆 | ~5.4M | 224×224 | **94.70 %** | **0.9942** | ✅ Testé |
| ResNet50 (réentraîné) | ~25M | 224×224 | 93.69 % | 0.9908 | ✅ Testé |
| EfficientNetB2 | ~7.7M | 260×260 | 93.05 % | 0.9909 | ✅ Testé |
| EfficientNetB0 | ~4M | 224×224 | 92.32 % | 0.9889 | ✅ Testé |

**YOLO26s-cls constitue à ce jour le meilleur compromis précision/légèreté du projet**, avec 4.6 fois moins de paramètres que ResNet50 tout en le dépassant de 1.01 point d'accuracy et en offrant une inférence de 0.9 ms/image (GPU P100). Licence à noter : **AGPL-3.0** (impact sur un déploiement commercial fermé — nécessiterait une licence Ultralytics payante).

## 3. Architectures identifiées pour extension future

### 3.1 MobileNetV3

CNN conçu spécifiquement pour les applications mobiles, exploitant des *depthwise separable convolutions* et des mécanismes d'attention squeeze-and-excitation. Disponible en versions Large et Small. Ses variantes quantifiées INT8 sont particulièrement adaptées aux NPU des smartphones et drones.

- **Intérêt pour Aigis** : candidat non testé, pertinent pour une comparaison future avec YOLO26s-cls sur le critère spécifique de la latence CPU/NPU embarquée.
- **Point de vigilance** : précision légèrement inférieure à des architectures plus récentes (ConvNeXt, EfficientNetV2), mais rapidité d'inférence supérieure.
- Référence : Howard, A. et al. (2019). Searching for MobileNetV3. arXiv:1905.02244

### 3.2 EfficientNet-Lite0

Version d'EfficientNet optimisée pour les environnements CPU contraints (suppression des opérations peu efficaces sur edge devices comme le squeeze-and-excitation).

- **Intérêt pour Aigis** : alternative plus légère à EfficientNetB0 (déjà testé à 92.32%), pourrait offrir un meilleur compromis vitesse/précision sur cible embarquée.
- **Point de vigilance** : performances généralement en retrait par rapport aux versions standards B0-B7.

### 3.3 EfficientNetV2-S / M

Utilise le *compound scaling* et les blocs Fused-MBConv pour un meilleur compromis vitesse d'entraînement/précision que la V1.

| Modèle | Top-1 accuracy (ImageNet, officiel) | Paramètres |
|---|---|---|
| EfficientNetV2-S | 83.9 % | ~22M |
| EfficientNetV2-M | 85.1 % | ~54M |

- **Intérêt pour Aigis** : successeur direct d'EfficientNetB0/B2 déjà testés, à évaluer sur le dataset Aigis pour vérifier un éventuel gain.
- **Point de vigilance** : taille supérieure à EfficientNetB0/B2 pour la version M, moins adaptée à l'embarqué strict.

### 3.4 DenseNet121 / ConvNeXt-Tiny

DenseNet121 utilise des connexions denses facilitant le flux de gradient ; ConvNeXt-Tiny modernise les CNN en s'inspirant des architectures Transformer (~82.0% top-1 sur ImageNet).

- **Intérêt pour Aigis** : ConvNeXt-Tiny comme référence de précision maximale, non contrainte par la taille.
- **Point de vigilance** : généralement trop lourds pour un déploiement embarqué strict sans distillation ou quantification avancée — moins prioritaires que YOLO26s-cls pour ce projet.

## 4. Complément — architectures de détection d'objets (veille prospective)

Les architectures suivantes concernent la **détection d'objets** (localisation par bounding box), pertinentes uniquement si le projet évolue vers une localisation précise des zones endommagées sur l'image, au-delà de la classification globale actuelle.

| Modèle | mAP (COCO val, T4, 640×640) | FPS/Latence | Paramètres | Positionnement |
|---|---|---|---|---|
| YOLOv6-N | 37.5 % | 1187 FPS | ~4.7M | Ultra-rapide |
| YOLOv6-S | 45.0 % | 484 FPS | ~18.5M | Équilibre |
| YOLOv6-M | 50.0 % | 226 FPS | ~34.9M | Bon compromis |
| YOLOv6-L | 52.8 % | 116 FPS | ~59.6M | Haute précision |
| YOLO-NAS S | 47.5 % | 3.21 ms | ~12M | Hautes perfs/Edge |
| YOLO26 (N à XL, détection) | 40.9 – 57.5 % | 1.7 – 11.8 ms (T4) | Variable | Edge-first, NMS-free |
| MobileNetV3-L (+ tête détection) | ~75.0 % (ImageNet, classif.) | ~600 FPS | ~5.4M | Mobile/drone |

**Correction de référence** : les chiffres YOLOv6 ci-dessus (37.5%/1187 FPS pour le N, etc.) proviennent du papier **YOLOv6 v3.0** (arXiv:2301.05586), et non du papier YOLOv6 original (arXiv:2209.02976) qui rapportait des valeurs différentes.

**Précision sur YOLO26** : annoncé en septembre 2025 (YOLO Vision 2025, Londres), officiellement lancé le **14 janvier 2026**. Licence AGPL-3.0.

## 5. Enjeux et défis pour les systèmes embarqués

### 5.1 Compromis précision-efficacité

Les modèles légers de deep learning opèrent sous des contraintes strictes de puissance de calcul, mémoire, et consommation énergétique. Les résultats d'Aigis illustrent concrètement ce compromis : YOLO26s-cls atteint la meilleure précision du projet avec le deuxième plus petit nombre de paramètres (après EfficientNetB0), validant l'intérêt des architectures récentes orientées edge.

### 5.2 Évaluation sur plateformes réelles

La littérature souligne l'importance de l'évaluation sur dispositifs réels (ex. NVIDIA Jetson Orin NX) plutôt que sur seule simulation GPU de bureau. **Limite actuelle du projet Aigis** : les temps d'inférence mesurés le sont sur GPU P100 (Kaggle), pas sur matériel embarqué représentatif — point à traiter dans les prochaines étapes.

## 6. Gaps de recherche identifiés

1. **Absence de benchmark de classification standardisé** pour l'inspection de défauts structurels en béton, comparable à COCO pour la détection générique — le protocole Aigis (4 modèles, même dataset, mêmes métriques) contribue à combler ce manque à l'échelle du projet.
2. **Manque de validation sur matériel embarqué réel** (Jetson, microcontrôleurs) pour les architectures testées — à date, seule la vitesse d'inférence GPU (P100) est mesurée.
3. **Compromis précision/énergie** insuffisamment exploré dans le contexte spécifique de l'inspection structurelle par drone.

## 7. Conclusion et perspectives pour Aigis

La veille met en évidence une tendance claire vers des architectures de classification/détection optimisées pour les environnements contraints, les approches basées sur YOLO dominant le paysage récent. Cette tendance est **confirmée expérimentalement** dans Aigis : YOLO26s-cls, la plus récente architecture testée, obtient la meilleure précision (94.70%) avec un nombre de paramètres nettement inférieur à ResNet50.

**Prochaines pistes à évaluer expérimentalement**, par priorité :
1. **MobileNetV3-Large** — référence mobile/drone, à comparer à YOLO26s-cls sur la latence embarquée
2. **EfficientNetV2-S** — successeur direct d'EfficientNetB0/B2 déjà testés
3. **EfficientNet-Lite0** — alternative optimisée CPU à EfficientNetB0
4. Mesure de vitesse d'inférence sur matériel embarqué réel (au-delà du GPU P100 actuel) pour les 4 modèles déjà testés

## 8. Bibliographie et ressources

1. Ultralytics Documentation. YOLO26. docs.ultralytics.com/models/yolo26
2. Ultralytics (2026). Ultralytics YOLO26: The New Standard for Edge-First Vision AI. ultralytics.com/blog
3. Sapkota, R. et al. (2025). YOLO26: Key Architectural Enhancements and Performance Benchmarking for Real-Time Object Detection. arXiv:2509.25164
4. Li, C. et al. (2023). YOLOv6 v3.0: A Full-Scale Reloading. arXiv:2301.05586
5. Deci AI. YOLO-NAS Object Detection Documentation. Roboflow Inference Models / GitHub Deci-AI/super-gradients
6. Howard, A. et al. (2019). Searching for MobileNetV3. arXiv:1905.02244
7. Tan, M. & Le, Q. (2021). EfficientNetV2: Smaller Models and Faster Training. arXiv:2104.00298
8. Zhai, H. et al. (2025). Edge Deployment of Deep Networks for Visual Detection: A Review. *IEEE Sensors Journal*, 25, 18662-18683

---
*Document préparé dans le cadre du stage de recherche Aigis — ESPRIT.*
