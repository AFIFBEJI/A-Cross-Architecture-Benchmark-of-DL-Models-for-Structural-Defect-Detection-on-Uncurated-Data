# Notebooks Colab de la v2 (récupérés le 2026-11-22, sur le Colab de Rania)

| Fichier | Contenu | Statut |
|---|---|---|
| `v2_colab_pipelines_rania_original_Untitled1.ipynb` | **Code d'origine** des 4 pipelines v2 : YOLO26n (cellule 5), EfficientNetB0 (6-7), EfficientNetB2 (10), ResNet50 (11-13). Hyperparamètres, seed 42, nettoyage (images corrompues + doublons MD5) et création de `val` (20 % de train si absent). | Code d'origine, **sans sorties d'entraînement** (cellules réexécutées/écrasées). Les 3 modèles PyTorch ont exactement les hyperparamètres de `code/v2/train_pytorch_reference.py` ; YOLO : 100 époques, patience 20, batch 64 (comme `train_yolo26n_reference.py`). |
| `colab_resnet50_keras_repro_rania_Untitled0.ipynb` | Reproduction **Keras ResNet50** sur `Aigis 20k/Model/Dataset.zip` (14 707 / 3 152 / 3 154 = **v1 brut, non nettoyé**), comparée à 0.9566. Entraînement interrompu à l'époque 1. | **Ce n'est pas le ResNet50 v1 rapporté (93,69 %)** : autre jeu de données (brut), autre code (Hamza/`New_Aigis.py`), aucun résultat. |

## Faits établis

- Les 4 pipelines v2 lisent **le même dossier Drive** : `MonProjet/NewDataSet/concrete-damage-v3` (sous-dossiers `train`, `val`, `test`). Au 2026-11-22, `train` contient 4 classes (`corrosion`, `cracks`, `normal`, `spalling`) sur ce Drive.
- Le dossier Drive `Aigis 20k/Model/` de Rania contient `Dataset.zip`, `New_Aigis.py`, `agis_model_folders.py`, `convert_model.py`, `NewTestApp.py`, `benchmark_results/` : c'est la base de Hamza, à inventorier.
- **Mesure du 2026-11-24 (Colab, dossier Drive actuel)** : `MonProjet/NewDataSet/concrete-damage-v3/train` contient **4 classes** : corrosion 5 208, cracks 7 648, normal 7 094, spalling 4 996, soit **24 946 images** (≈ 25 000, la copie « Drive » signalée par le superviseur). Les scores v2 rapportés ne peuvent donc pas venir de ce dossier : R50/B0/B2 « corrosion seule » = 602-603 images, YOLO v2 = train 9 226 / val 2 305 (copie Kaggle, 11 531 images). Le jeu utilisé par les entraînements v2 d'origine reste **non identifié** ; seule une ré-exécution des pipelines de `Untitled1` sur ce dossier (4 classes, ≈ 25 000 images d'entraînement) donnerait des résultats 4 classes traçables, à présenter comme un **nouveau run**.
- Les sorties du notebook sur Colab (T4, Python 3.13) montrent l'environnement actuel, pas celui de l'été.
