# Datasets : sources, licences, nettoyage, classes, partitions

Document unique de référence. Les champs **⚠** sont des informations que le dépôt ne permet pas de vérifier et que l'auteur doit confirmer.

## 1. Versions du dataset

| Nom | Contenu | Effectifs | Où | Utilisé par |
|---|---|---|---|---|
| **v1 brut** | Dataset Kaggle `daikhiyassine/dataset` (héritage du travail Aigis antérieur) | 21 013 images : train 14 707 / val 3 152 / test 3 154 (corrosion 3 936+844+844, spalling 3 747+803+804, normal 3 594+770+771, cracks 3 430+735+735) ; le même dossier contient aussi des dossiers de classes à la racine (5 624 / 5 354 / 5 135 / 4 900) | Kaggle | point de départ de v1 (relevé dans le notebook B2, exécution du 2026-07-28) |
| **v1 nettoyé** | v1 brut après `code/v1/data/02_clean_dataset.py` | **21 010** : train 14 707 / val 3 151 / test 3 152 (classes train : corrosion 3 936, spalling 3 747, normal 3 594, cracks 3 430) | Kaggle `/kaggle/working/dataset_clean_*` (non persistant) | les 4 modèles v1 |
| **extension « v3 »** | v1 nettoyé + crops de MDMCS, RCS-2119, RCS-1841 (`03a`, `03b`) | 35 641 : corrosion 7 443, spalling 7 138, normal 10 135, cracks 10 925 ; split 70/15/15, seed 42 | Kaggle `daikhiyassine/concrete-damage-v3` | voir §4 (v2) |

Provenance du v1 brut : ⚠ sources d'origine des 21 013 images à lister (héritées de l'équipe Aigis précédente) ; l'audit a trouvé 5 753 correspondances MD5 entre le v1 et des images externes ajoutées, ce qui indique des sources qui se recoupent.

## 2. Sources externes et licences

| Source | Usage | Licence (vérifiée sur la page de la source) | Référence |
|---|---|---|---|
| MDMCS (HRCDS), LabelMe polygones → crops | extension v3 | CC BY 4.0 | doi 10.17632/6x4dzzrs2h.2 |
| Reinforced Concrete Structure Segmentation (2119) | extension v3 | CC BY 4.0 | doi 10.17632/2vkm6k4cfg.1 |
| Reinforced Concrete Structure Segmentation (1841) | extension v3 | CC BY 4.0 | doi 10.17632/yfhwfcrfmk.1 |
| Surface Crack Detection (Özgenel), copie Kaggle `arunrk7/surface-crack-detection` | **inclus dans v3** (Positive → cracks, Negative → normal ; plafond 5 000 images par classe) | CC BY 4.0 (source Mendeley ; licence de la copie Kaggle à revérifier) | doi 10.17632/5y9wdsg2zt.2 |
| Autres (Historical_Building_Crack_2019, BFDD, Damage Detection Dataset, SDNET2018, dacl10k, UAV Building Surface Defects, omnicrack30k, …) | évalués, **non utilisés** | voir `reports/Rapport_Global/Suivi_Nouveaux_Datasets.xlsx` | — |

Les datasets sous licence non commerciale ou GPL (dacl10k CC BY-NC, UAV Building Surface Defects CC BY-NC-ND, omnicrack30k GPL v3) n'ont pas été intégrés. Le v1 brut n'a pas de licence établie ⚠ : **ne pas redistribuer** ses images ni le v1 nettoyé tant que ce point n'est pas réglé (d'où l'absence d'archive et de checksum d'archive ci-dessous ; seuls les manifestes MD5 sont publiés).

## 3. Nettoyage et correspondance des classes

**Procédure v1** (`code/v1/data/02_clean_dataset.py`, identique pour les 4 modèles) : (1) empreinte MD5 de chaque fichier ; (2) doublons exacts retirés avec priorité train > val > test (3 doublons, tous des paires spalling/cracks, donc 3 fuites inter-splits) ; (3) conversion RGB (626 images YOLO) ; (4) redimensionnement 256×256, JPEG qualité 85 ; (5) écriture de `rapport_nettoyage_final_*.json`. **Deux variantes ont réellement été exécutées** : (a) B0/YOLO26 (`02_clean_dataset.py`) : redimensionnement 256×256 et JPEG q85 ; (b) EfficientNet-B2 (`02_clean_dataset_b2_fullres.py`, notebook d'origine fourni) : **sans redimensionnement**, taille et format d'origine, seule la conversion RGB (626 images) est appliquée. Les deux variantes donnent les mêmes effectifs finaux (14 707 / 3 151 / 3 152 = 21 010) et retirent les mêmes 3 doublons : `test/spalling/427_yang_jpg…` (doublon de `train/cracks/Image_-192-_jpg…`), `val/spalling/train_train_0318.jpg` (doublon de `train/cracks/train_train_0318.jpg`), `test/cracks/Image_-120-_jpg…` (doublon de `val/spalling/262_yang_jpg…`). Synthèses disponibles : B0 et YOLO26 (`models/*/v1/Dataset/`). **Pas de synthèse pour EfficientNet-B2 et ResNet50 v1** (le script les régénère ; à relancer ⚠).

**Procédure d'extension** (notebook d'origine `notebooks/extension/dataset_extension_v3_kaggle_original.ipynb`, consolidée dans `03a`/`03b`) : deux étapes — (1) fusion de MDMCS, RCS-2119 et Surface Crack avec le v1 brut dans `dataset_v3_final` (seed 42, date 10/08/2026), puis (2) ajout séparé des crops RCS-1841 (seed **43**), plus une déduplication de sécurité. Le notebook sauvegardé n'est pas linéaire (cellules d'essai/erreur conservées, p. ex. `FileNotFoundError` sur RCS-2119) ; sa dernière exécution du 2026-11-20 s'arrête sur cette erreur. `03a`/`03b` sont une consolidation en un seul passage : les partitions peuvent différer de celles du dataset publié, qui fait foi. Détails : polygone LabelMe → boîte englobante, marge 15 %, côté minimum 40 px ; dédoublonnage MD5 contre l'existant puis entre sources ; plafonds 5 000 images pour `cracks` et `normal` ; split 70/15/15, seed 42. Limites connues : split au niveau du crop et non de l'image source (risque de fuite), classe Delamination exclue (choix de l'auteur).

**Classes finales** : `corrosion`, `cracks`, `normal`, `spalling` (ordre alphabétique = indices Keras/YOLO 0–3).

| Source | Étiquette source | → classe |
|---|---|---|
| MDMCS | `spalling`, `corrosion`, `crack` | spalling, corrosion, cracks |
| RCS-2119 | `crack`, `rebar corrosion`, `concrete spalling` | cracks, corrosion, spalling |
| RCS-1841 | `structural crack`, `microcrack`, `rebar corrosion`, `minor spalling`, `moderate spalling` | cracks, cracks, corrosion, spalling, spalling |
| Surface Crack (`arunrk7/surface-crack-detection`) | `Positive`, `Negative` | cracks, normal |

## 4. Données de la version 2 : ce que chaque modèle a utilisé

Les rapports v2 déclarent deux copies différentes, alors que les **quatre `analyse_dataset.csv` d'origine étaient identiques** (même MD5 `c17df336…`, effectifs de la copie Drive : train 24 946 images lues par le CSV, vs « ~25 050 » écrit dans les rapports).

| Modèle v2 | Copie déclarée dans le rapport | Effectifs déclarés | Évaluation | `analyse_dataset.csv` après correction |
|---|---|---|---|---|
| ResNet50 | Google Drive `NewDataSet` | ~25 049 train | `val/` existant (corrosion seule) | copie Drive d'origine conservée (c'était la bonne) |
| EfficientNet-B0 | Google Drive `NewDataSet` | ~25 050 train | idem | idem |
| EfficientNet-B2 | Kaggle `concrete-damage-v3` | 11 531 images (9 226 train / 2 305 val, 80/20 stratifié) | split val | **remplacé** par un fichier reconstitué depuis le rapport (marqué `source=derive_du_rapport`) |
| YOLO26n | Kaggle `concrete-damage-v3` | idem B2 | split val | **remplacé** (idem) |

Incohérence non résolue ⚠ : `concrete-damage-v3` contient 35 641 images (train ≈ 24 950), pas 11 531. La copie Drive (~25 000 train) ressemble à ce train v3, et la « copie Kaggle » de 11 531 images ne correspond à aucun jeu documenté. L'auteur doit indiquer, pour B2 et YOLO v2, le nom exact du dataset Kaggle attaché au notebook, et fournir son manifeste.

## 5. Manifestes de partitionnement

Dossier `manifests/` (voir `manifests/README.md`) : CSV `relative_path, class, split, md5`. Générés par `tools/make_manifest.py`, vérifiés par `tools/verify_manifest.py`. **Les images ne sont pas accessibles dans ce dépôt ni dans l'environnement où il a été préparé : les manifestes doivent être générés par l'auteur sur Kaggle/Colab** (commandes dans `manifests/README.md`). Contrôle v1 attendu : 14 707 / 3 151 / 3 152.

## 6. Checksums

- Archives/datasets nettoyés : non redistribués (licence du v1 brut non établie) → pas de checksum d'archive ; les manifestes MD5 par image tiennent lieu de checksum. Si une redistribution est autorisée, produire `SHA256SUMS.md` avec `tools/hash_files.py`.
- Poids : `docs/WEIGHTS_REGISTRY.md`.

## Mesure du dossier Drive `concrete-damage-v3` (2026-11-24)

Lu depuis Colab sur `MonProjet/NewDataSet/concrete-damage-v3/train` : corrosion 5 208, cracks 7 648, normal 7 094, spalling 4 996 (24 946 images, 4 classes). C'est le dossier lu par les 4 pipelines de `notebooks/v2/v2_colab_pipelines_rania_original_Untitled1.ipynb`. Les résultats v2 rapportés (R50/B0/B2 : 602-603 images, corrosion seule ; YOLO : train 9 226 / val 2 305) ne correspondent pas à ce dossier. Jeu d'origine de ces résultats : non identifié. Voir `EXPERIMENT_INDEX.md` (A1, A3, A16).
