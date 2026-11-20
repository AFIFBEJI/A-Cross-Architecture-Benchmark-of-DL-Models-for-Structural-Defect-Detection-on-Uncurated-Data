# Aigis-ZeroShot-Benchmark

**Evaluating Zero-Shot Generalization: A Cross-Architecture Benchmark of Deep Learning Models for Structural Defect Detection on Uncurated Data**

![Python](https://img.shields.io/badge/Python-3.x-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-used-red) ![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO26-purple) ![EfficientNet](https://img.shields.io/badge/EfficientNet-B0%20%7C%20B2-green) ![Status](https://img.shields.io/badge/Status-Deliverables%20submitted-brightgreen)

A benchmarking study asking a harder question than "which model scores highest on a clean test set": which architecture actually holds up when deployed zero-shot on 21,013 uncontrolled, real-world images it was never tuned for?

## Table of Contents

- [Overview](#overview)
- [Continuity from Prior Work](#continuity-from-prior-work)
- [Project Personnel](#project-personnel)
- [Motivation](#motivation)
- [Research Gap](#research-gap)
- [Objectives](#objectives)
- [Methodology](#methodology)
- [Expected Results](#expected-results)
- [Evaluation Metrics](#evaluation-metrics)
- [Repository Structure](#repository-structure)
- [Data and Model Weights](#data-and-model-weights)
- [Installation](#installation)
- [Risks and Limitations](#risks-and-limitations)
- [References](#references)
- [Status](#status)

## Overview

This project benchmarks pre-trained EfficientNet-B0, EfficientNet-B2, and YOLOv26-Classification against the established Aigis (ResNet50) baseline for structural defect detection, not on a curated benchmark set, but on 21,013 uncurated images reflecting real-world illumination, viewpoint, and acquisition variability. The central question is generalization under domain shift: does performance measured on a clean, curated dataset actually predict performance in the field, and does architectural design (specifically, EfficientNet's compound scaling) offer measurable robustness against that shift?

## Continuity from Prior Work

This project is a direct continuation of **Aigis-Benchmark-GROUP2**, the prior internship benchmark that evaluated the Aigis (ResNet50) baseline against EfficientNet-B0, EfficientNet-B2, and YOLOv8-seg-n on a curated 3,600-image dataset.

Two findings from that prior work directly motivate this project's design:

1. **YOLOv8-seg-n underperformed catastrophically in the curated benchmark.** Rather than persist with a segmentation-head architecture that already failed under favorable, curated conditions, this project switches to YOLOv26-Classification, a more recent, purpose-built classification variant.
2. **The prior benchmark only established performance under curated, favorable conditions.** It did not answer whether that performance holds when the same models face the unfiltered variability of real-world data, which is precisely the gap this project is designed to close.

This project therefore does not repeat the prior benchmark; it extends it along the one axis the prior work left untested: generalization, not just accuracy.

## Project Personnel

**Supervisor:**
- Afif Beji, Eng., M.Sc.
- Role: Assistant Professor & Project Supervisor
- Institution: ESPRIT, School of Engineering

**Interns:**
- Rania Kaabi
- Yacin Daikhi
- Role: Third-Year Engineering Students
- Institution: ESPRIT, School of Engineering

## Motivation

Structural Health Monitoring (SHM) systems are typically validated on datasets curated for balance and image quality, conditions that rarely match field deployment. A model that performs well on a curated 3,600-image benchmark provides limited evidence that it will perform reliably when pointed at an uncontrolled stream of inspection photographs taken under variable lighting, angle, and equipment conditions. This is the standard failure mode that separates benchmark performance from deployed performance in computer vision systems generally, and structural defect detection is not exempt from it.

## Research Gap

| Gap | Prior Work (GROUP2 Benchmark) | This Project |
|---|---|---|
| Domain shift quantification | Not evaluated, single curated dataset only | Explicit before/after comparison: curated (3,600 images) vs. uncurated (21,013 images) performance degradation |
| Architecture choice under known failure | YOLOv8-seg-n included and found catastrophic | Replaced with YOLOv26-Classification, a task-appropriate architecture for this comparison |
| Compound scaling as a robustness factor | EfficientNet-B0/B2 compared on accuracy alone | Statistical analysis of whether compound scaling (B0 to B2) improves robustness to illumination/viewpoint shift, not just raw accuracy |
| Scale of evaluation | 3,600 curated images | 21,013 uncurated images, a scale intended to make degradation patterns statistically detectable rather than anecdotal |

## Objectives

1. Deploy pre-trained EfficientNet-B0, EfficientNet-B2, and YOLOv26-Classification models on an uncurated dataset of 21,013 images.
2. Quantify and compare performance degradation under domain shift against the established Aigis (ResNet50) baseline.
3. Statistically analyze the efficacy of compound scaling in handling real-world illumination and viewpoint variability.

## Methodology

### Phase 1: Uncurated Dataset Assembly and Characterization
Assemble the 21,013-image uncurated dataset and characterize its composition explicitly: distribution of illumination conditions, viewpoint angles, image sources, and any available ground-truth defect labels or proxy annotations. A domain-shift study is only as credible as its description of what, specifically, shifted.

### Phase 2: Zero-Shot Inference Pipeline
Deploy each pre-trained model (Aigis/ResNet50 baseline, EfficientNet-B0, EfficientNet-B2, YOLOv26-Classification) on the full uncurated dataset without fine-tuning, consistent with the zero-shot generalization framing, using batch-optimized inference (batching, mixed precision, checkpointed progress).

### Phase 3: Domain-Shift Quantification
For each model, compute the performance delta between its curated-benchmark performance (from the GROUP2 study, where available) and its zero-shot performance on the uncurated set, reported as an explicit degradation metric per architecture.

### Phase 4: Statistical Analysis of Compound Scaling
Test whether EfficientNet-B2 degrades less under domain shift than EfficientNet-B0, using a paired statistical comparison (e.g., McNemar's test, or a bootstrap confidence interval on the accuracy gap) rather than a simple point-estimate comparison.

### Phase 5: Evaluation and Reporting
Generate normalized confusion matrices and a full evaluation report (accuracy, macro-F1, precision, recall) per model, and per condition subgroup where metadata is available.

### Phase 6: Manuscript Drafting
Draft the Methodology and Results sections of a manuscript targeting an SHM/computer-vision journal, building on the domain-shift and compound-scaling analyses above.

## Expected Results

| Comparison | Hypothesis | Rationale |
|---|---|---|
| Aigis (ResNet50) vs. EfficientNet-B0/B2 under domain shift | EfficientNet variants degrade less due to more efficient feature representations | Compound scaling balances depth/width/resolution rather than over-fitting to a fixed input regime |
| EfficientNet-B0 vs. EfficientNet-B2 | B2 shows smaller degradation than B0 | Larger effective receptive field and resolution may better absorb illumination/viewpoint variability |
| YOLOv26-Classification vs. prior YOLOv8-seg-n (qualitative) | YOLOv26-Classification avoids the prior architecture's catastrophic failure mode | Task-head mismatch (segmentation vs. classification) is a plausible explanation for the prior failure |

These are directional hypotheses to be tested, not pre-supposed results. The possibility that compound scaling shows no measurable robustness benefit would itself be a non-trivial finding.

## Evaluation Metrics

- **Accuracy** (overall and per-subgroup, where condition metadata is available)
- **Macro-F1**, which weights each class equally, appropriate given likely class imbalance in real-world defect data
- **Precision and Recall**, per class and macro-averaged
- **Normalized confusion matrices**, per model
- **Domain-shift degradation delta**, the core metric of this study: the performance gap between curated-benchmark and zero-shot uncurated performance per model

## Repository Structure

```
.
├── EXPERIMENT_INDEX.md          # Index des expériences (modèle x version), définitions v1/v2, anomalies
├── code/                        # Scripts: nettoyage, partition, tuning, entraînement, test (v1) ; références v2
├── notebooks/                   # Exports .ipynb (avec sorties) + EXECUTION_ORDER.md
├── evaluation/                  # Métriques, matrices de confusion, McNemar/bootstrap, courbes
├── inference/                   # Inférence (évaluation seule) d'un checkpoint sur un dossier d'images
├── docs/                        # DATASETS.md, WEIGHTS_REGISTRY.md, ENVIRONMENT.md
├── manifests/                   # Manifestes de partition (CSV MD5) — à générer
├── environments/                # pip freeze / versions par notebook — à générer
├── tools/                       # make_manifest, verify_manifest, hash_files, capture_env
├── models/<modèle>/{v1,v2}/     # Dataset (rapports), Hyperparametres, Resultats, rapports par modèle
├── reports/                     # Rapport global, bibliographie, veille, environnement
├── requirements.txt, requirements-tf.txt, requirements-torch.txt
└── README.md
```

> **Note de cohérence (voir `EXPERIMENT_INDEX.md`)** : le protocole zero-shot décrit plus haut (Phase 2) n'a pas été exécuté. Les résultats du dépôt proviennent d'expériences où chaque modèle est **fine-tuné** sur le train split de son dataset puis évalué sur un split de ce dataset (v1 : jeu nettoyé de 21 010 images ; v2 : jeu étendu, protocole différent). Les sections Objectives/Methodology décrivent le protocole prévu et seront alignées avec le superviseur.

## Data and Model Weights

Règle uniforme (détail : `docs/WEIGHTS_REGISTRY.md`) : les checkpoints d'entraînement (`.keras`, `.pt`) ne sont **pas** dans Git ; ils sont sur Google Drive, identifiés par chemin, taille et SHA-256. Seuls trois exports `.tflite` v1 (convertis, n'ayant produit aucune métrique) sont versionnés. Les images ne sont pas redistribuées ; les partitions sont décrites par des manifestes MD5 (`manifests/`, `docs/DATASETS.md`).

**[Project Google Drive folder](https://drive.google.com/drive/u/0/folders/1F4AUqhfDRvCa-3bZb30VMvE_XZP_OPIN)**

## Installation

```bash
git clone https://github.com/AFIFBEJI/A-Cross-Architecture-Benchmark-of-DL-Models-for-Structural-Defect-Detection-on-Uncurated-Data.git
cd A-Cross-Architecture-Benchmark-of-DL-Models-for-Structural-Defect-Detection-on-Uncurated-Data
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt   # + requirements-tf.txt ou requirements-torch.txt selon l'expérience (docs/ENVIRONMENT.md)
```

Core dependencies: `torch`, `torchvision`, `tensorflow`/`keras` (EfficientNet and ResNet50 `.keras` models), `ultralytics` (YOLOv26-Classification), `scikit-learn`, `scipy`/`statsmodels`, `pandas`, `matplotlib`/`seaborn`.

## Risks and Limitations

- **Ground-truth quality on uncurated data** is the primary methodological risk. "Uncurated" implies less control over label quality than the prior curated benchmark, which should be addressed through a documented annotation protocol or an explicit acknowledgment of label-noise limitations.
- **Dataset licensing and provenance** for the 21,013 images should be documented (source, collection consent, redistribution rights) before any public release.
- **Zero-shot framing must be strictly enforced.** Any fine-tuning or threshold calibration on the uncurated set would contaminate the zero-shot claim.
- **Statistical power** for the compound-scaling claim depends on how the paired test is constructed; a naive accuracy difference without confidence intervals would not support the statistical objective.
- **Comparability to the prior curated-benchmark numbers** assumes the GROUP2 results were computed under comparable conditions (same defect taxonomy, same evaluation protocol), which should be verified.

## References

1. Tan, M., & Le, Q. V. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.* ICML.
2. Ultralytics (2026). *YOLO26: Unified Real-Time End-to-End Vision Models.* docs.ultralytics.com/models/yolo26
3. Aigis-Benchmark-GROUP2: prior internship benchmark that this project extends.

## Status

Travail en cours de mise en conformité reproductibilité (voir `EXPERIMENT_INDEX.md`). Disponible : code v1, index des expériences, documentation datasets/poids/environnement, scripts d'évaluation. **Restent à fournir** : notebooks `.ipynb` exportés, manifestes MD5, SHA-256 et chemins Drive des 8 checkpoints, versions figées des environnements, évaluation 4 classes de la v2. Les résultats v2 ne sont pas des résultats de test à quatre classes (voir index). Le protocole zero-shot n'a pas été exécuté. Rédaction du manuscrit : travail ultérieur.
