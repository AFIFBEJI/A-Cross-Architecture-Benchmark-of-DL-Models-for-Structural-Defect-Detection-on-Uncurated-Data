# Rapport d'Évaluation — Modèle EfficientNetB2 (Projet Aigis)

**Date d'évaluation :** 28 juillet 2026
**Auteur :** Yacin — Stage recherche, détection automatique de défauts structurels en béton
**Architecture :** EfficientNetB2 (transfer learning, ImageNet)

---

## 1. Contexte

Ce rapport présente les résultats du modèle EfficientNetB2 entraîné pour la classification automatique de défauts structurels sur infrastructures en béton (4 classes : corrosion, fissures, spalling/écaillage, normal). Les résultats sont évalués sur le **test set** (données jamais vues pendant l'entraînement ni le tuning).

## 2. Configuration du modèle

| Paramètre | Valeur |
|---|---|
| Architecture de base | EfficientNetB2 (poids ImageNet) |
| Taille d'entrée | 260 × 260 × 3 |
| Entraînement | 2 phases (tête gelée → fine-tuning partiel) |
| Hyperparamètres | Sélectionnés via recherche bayésienne (Keras Tuner) |
| Dataset (nettoyé) | 21 010 images, 4 classes, splits 70/15/15 |

## 3. Résultats globaux sur le test set (3 152 images)

| Métrique | Valeur |
|---|---|
| **Accuracy** | **93.05 %** |
| **Loss** | 0.2003 |
| **AUC (Keras, multi-label)** | 0.9894 |
| **AUC macro (sklearn, one-vs-rest)** | 0.9909 |

## 4. Résultats détaillés par classe

| Classe | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Corrosion | 97.28 % | 93.13 % | 95.16 % | 844 |
| Cracks (fissures) | 88.48 % | 91.01 % | 89.72 % | 734 |
| Normal | 99.87 % | 99.87 % | 99.87 % | 771 |
| Spalling (écaillage) | 86.67 % | 88.29 % | 87.48 % | 803 |

**Moyennes globales**
- Macro avg (F1) : 93.06 %
- Weighted avg (F1) : 93.09 %

## 5. Analyse

**Points forts**
- La classe **normal** est quasi parfaitement reconnue (F1 = 99.87 %), ce qui indique que le modèle distingue très bien les surfaces saines des surfaces endommagées.
- La classe **corrosion** est très bien détectée (F1 = 95.16 %), avec une precision élevée (peu de faux positifs).
- L'AUC macro de 0.99 confirme une excellente capacité de discrimination globale entre les 4 classes.

**Point faible identifié : confusion cracks / spalling**
- Ce sont les deux classes avec le F1-score le plus bas (89.72 % et 87.48 %).
- Cette confusion est cohérente avec une observation faite lors du nettoyage du dataset : **2 cas d'images identiques mais étiquetées différemment** (`spalling` vs `cracks`) ont été détectés dans le dataset source (voir `rapport_nettoyage_final.json`). Cela suggère une possible ambiguïté de labellisation à la source, au-delà des seuls doublons trouvés — d'autres images uniques mal étiquetées ne peuvent pas être exclues.
- Recommandation : examiner la matrice de confusion normalisée (`confusion_matrix_normalized_efficientnetb2.png`) pour confirmer si les erreurs se concentrent spécifiquement entre ces deux classes, et envisager une révision manuelle d'un échantillon d'images `cracks`/`spalling` mal classées.

## 6. Comparaison avec les travaux précédents (ResNet50)

| Modèle | Accuracy (référence) |
|---|---|
| ResNet50 (Hamza, dataset 21k images) | ~95.66 % |
| ResNet50 (zero-shot baseline) | ~58.6 % |
| **EfficientNetB2 (ce rapport)** | **93.05 %** |

EfficientNetB2 obtient une performance légèrement inférieure à ResNet50 sur ce dataset, mais reste dans une plage de performance élevée et cohérente, avec un modèle nettement plus léger (utile pour un futur déploiement TFLite/mobile).

## 7. Fichiers associés (dans ce dossier Drive)

| Fichier | Contenu |
|---|---|
| `best_hyperparameters.json` | Hyperparamètres retenus après tuning |
| `best_model_phase1.keras` | Modèle après phase 1 (tête seule) |
| `best_model_phase2.keras` | Modèle après fine-tuning (si applicable) |
| `aigis_efficientnetb2_final.keras` | Modèle final complet |
| `aigis_efficientnetb2.tflite` | Version exportée pour déploiement mobile/embarqué |
| `test_results_efficientnetb2.json` | Résultats bruts (JSON) |
| `confusion_matrix_efficientnetb2.png` | Matrice de confusion (valeurs brutes) |
| `confusion_matrix_normalized_efficientnetb2.png` | Matrice de confusion normalisée (%) |
| `training_log_phase1.csv` / `phase2.csv` | Historique d'entraînement (loss/accuracy par epoch) |
| `rapport_nettoyage_final.json` | Rapport du nettoyage du dataset (doublons, fuites, conversions RGB) |

## 8. Prochaines étapes suggérées

1. Analyser en détail la confusion `cracks` / `spalling` via la matrice normalisée et un échantillon d'erreurs.
2. Vérifier manuellement la cohérence des labels sur un sous-échantillon du dataset source (au-delà des doublons exacts déjà identifiés).
3. Comparer les temps d'inférence EfficientNetB2 (.tflite) vs ResNet50 (.tflite) pour trancher sur le modèle à déployer en production.
4. Discuter ces résultats avec le superviseur lors de la prochaine réunion hebdomadaire.

---
*Rapport généré à partir de l'évaluation sur Kaggle (GPU P100/T4), pipeline Aigis.*
