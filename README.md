# Aigis-ZeroShot-Benchmark

**Evaluating Zero-Shot Generalization: A Cross-Architecture Benchmark of Deep Learning Models for Structural Defect Detection on Uncurated Data**

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-academic--research-green.svg)](#license)
[![Status](https://img.shields.io/badge/status-in%20development-yellow.svg)]()
[![PyTorch](https://img.shields.io/badge/framework-PyTorch-EE4C2C.svg)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/YOLO-YOLOv26-111111.svg)](https://docs.ultralytics.com/models/yolo26)
[![EfficientNet](https://img.shields.io/badge/CNN-EfficientNet--B0%2FB2-4CAF50.svg)]()
[![Domain Shift](https://img.shields.io/badge/focus-Zero--Shot%20Generalization-orange.svg)]()
[![Reproducibility](https://img.shields.io/badge/reproducible-yes-brightgreen.svg)]()

> A benchmarking study asking a harder question than "which model scores highest on a clean test set": which architecture actually holds up when deployed zero-shot on 21,013 uncontrolled, real-world images it was never tuned for?

---

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
- [Installation](#installation)
- [Risks and Limitations](#risks-and-limitations)
- [References](#references)
- [Status](#status)

---

## Overview

This project benchmarks pre-trained **EfficientNet-B0**, **EfficientNet-B2**, and **YOLOv26-Classification** against the established **Aigis (ResNet50)** baseline for structural defect detection — not on a curated benchmark set, but on **21,013 uncurated images** reflecting real-world illumination, viewpoint, and acquisition variability. The central question is generalization under **domain shift**: does performance measured on a clean, curated dataset actually predict performance in the field, and does architectural design (specifically, EfficientNet's compound scaling) offer measurable robustness against that shift?

## Continuity from Prior Work

This project is a direct continuation of **[Aigis-Benchmark-GROUP2](https://github.com/Aigis-AI-SummerInternship2025/Aigis-Benchmark-GROUP2)**, the prior internship benchmark that evaluated the Aigis (ResNet50) baseline against EfficientNet-B0, EfficientNet-B2, and YOLOv8-seg-n on a **curated 3,600-image dataset**.

Two findings from that prior work directly motivate this project's design:

1. **YOLOv8-seg-n underperformed catastrophically** in the curated benchmark. Rather than persist with a segmentation-head architecture that already failed under favorable, curated conditions, this project switches to **YOLOv26-Classification** — a more recent, purpose-built classification variant, avoiding a known point of failure rather than re-testing it at greater scale and expecting a different outcome.
2. **The prior benchmark only established performance under curated, favorable conditions.** It did not answer whether that performance holds when the same models face the unfiltered variability of real-world data — which is precisely the gap this project is designed to close.

This project therefore does not repeat the prior benchmark; it extends it along the one axis the prior work left untested: **generalization, not just accuracy**.

## Project Personnel

**Supervisor:**

- **Afif Beji, Eng., M.Sc.**
  * *Role:* Assistant Professor & Project Supervisor
  * *Institution:* ESPRIT, School of Engineering
- **Interns:**
  - **Rania Kaabi**
  - **Yacin Daikhi**
      * *Role:* Third-Year Engineering Students
      * *Institution:* ESPRIT, School of Engineering

## Motivation

Structural Health Monitoring (SHM) systems are typically validated on datasets curated for balance and image quality — conditions that rarely match field deployment. A model that performs well on a curated 3,600-image benchmark provides limited evidence that it will perform reliably when pointed at an uncontrolled stream of inspection photographs taken under variable lighting, angle, and equipment conditions. This is not a hypothetical concern: it is the standard failure mode that separates benchmark performance from deployed performance in computer vision systems generally, and structural defect detection is not exempt from it.

## Research Gap

| Gap | Prior Work (GROUP2 Benchmark) | This Project |
|---|---|---|
| **Domain shift quantification** | Not evaluated — single curated dataset only | Explicit before/after comparison: curated (3,600 images) vs. uncurated (21,013 images) performance degradation |
| **Architecture choice under known failure** | YOLOv8-seg-n included and found catastrophic | Replaced with YOLOv26-Classification, a task-appropriate architecture for this comparison |
| **Compound scaling as a robustness factor** | EfficientNet-B0/B2 compared on accuracy alone | Statistical analysis of whether compound scaling (B0 → B2) specifically improves robustness to illumination/viewpoint shift, not just raw accuracy |
| **Scale of evaluation** | 3,600 curated images | 21,013 uncurated images — a scale intended to make degradation patterns statistically detectable rather than anecdotal |

## Objectives

- Deploy pre-trained **EfficientNet-B0**, **EfficientNet-B2**, and **YOLOv26-Classification** models on an uncurated dataset of 21,013 images.
- Quantify and compare performance degradation under domain shift against the established **Aigis (ResNet50)** baseline.
- Statistically analyze the efficacy of compound scaling in handling real-world illumination and viewpoint variability.

## Methodology

### Phase 1 — Uncurated Dataset Assembly and Characterization
Assemble the 21,013-image uncurated dataset, and — critically, given the "uncurated" framing — characterize its composition explicitly: distribution of illumination conditions, viewpoint angles, image sources, and any available ground-truth defect labels or proxy annotations. This characterization step is treated as a first-class methodological output, not a preliminary formality, since a domain-shift study is only as credible as its description of what, specifically, shifted.

### Phase 2 — Zero-Shot Inference Pipeline
Deploy each pre-trained model (Aigis/ResNet50 baseline, EfficientNet-B0, EfficientNet-B2, YOLOv26-Classification) on the full uncurated dataset **without fine-tuning**, consistent with the zero-shot generalization framing. Build batch-optimized inference scripts capable of processing the full 21,013-image set efficiently (batching, mixed-precision inference, and checkpointed progress to tolerate interruption on a dataset this size).

### Phase 3 — Domain-Shift Quantification
For each model, compute the performance delta between its originally reported/curated-benchmark performance (from the GROUP2 study, where available) and its zero-shot performance on the uncurated set. Report this as an explicit **degradation metric** per architecture, not just two independent accuracy numbers to be compared informally.

### Phase 4 — Statistical Analysis of Compound Scaling
Directly test whether EfficientNet-B2 (deeper/wider/higher-resolution than B0 via compound scaling) degrades less under domain shift than EfficientNet-B0. This requires a paired statistical comparison (e.g., McNemar's test on prediction agreement/disagreement with ground truth, or a bootstrap confidence interval on the accuracy gap) rather than a simple point-estimate comparison, since the claim being tested is specifically about robustness, not just which model scores higher.

### Phase 5 — Evaluation and Reporting
Generate normalized confusion matrices and a full evaluation report (accuracy, macro-F1, precision, recall) per model, per condition subgroup (e.g., by illumination bucket or viewpoint category, where such metadata is available), to support both the aggregate finding and a more granular failure analysis.

### Phase 6 — Manuscript Drafting
Draft the Methodology and Results sections of a manuscript targeting a prestigious SHM/computer-vision journal, building directly on the quantitative domain-shift and compound-scaling analyses above.

## Expected Results

| Comparison | Hypothesis | Rationale |
|---|---|---|
| Aigis (ResNet50) vs. EfficientNet-B0/B2 under domain shift | EfficientNet variants degrade less due to more efficient feature representations | Compound scaling is designed to balance depth/width/resolution rather than over-fit to a fixed input regime |
| EfficientNet-B0 vs. EfficientNet-B2 | B2 shows smaller degradation than B0 | Larger effective receptive field and resolution may better absorb illumination/viewpoint variability |
| YOLOv26-Classification vs. prior YOLOv8-seg-n (qualitative, not re-tested) | YOLOv26-Classification avoids the prior architecture's catastrophic failure mode | Task-head mismatch (segmentation vs. classification) is a plausible explanation for the prior failure, addressed directly by this substitution |

These are **directional hypotheses to be tested**, not pre-supposed results — the value of this study lies precisely in whether the data supports or contradicts them, including the possibility that compound scaling shows no measurable robustness benefit, which would itself be a publishable, non-trivial finding.

## Evaluation Metrics

- **Accuracy** (overall and per-subgroup, where condition metadata is available)
- **Macro-F1** — weights each class equally, appropriate given likely class imbalance in real-world defect data
- **Precision and Recall** — per class and macro-averaged
- **Normalized confusion matrices** — per model, to visualize specific failure/confusion patterns rather than a single scalar score
- **Domain-shift degradation delta** — the core novel metric of this study, computed as the performance gap between curated-benchmark and zero-shot uncurated performance per model

## Repository Structure

```
aigis-zeroshot-benchmark/
├── data/
│   ├── uncurated/                   # 21,013-image dataset (licensing permitting; see Risks)
│   └── metadata/                    # Illumination/viewpoint/source characterization
├── models/
│   ├── aigis_resnet50/              # Baseline model checkpoint/wrapper
│   ├── efficientnet_b0/
│   ├── efficientnet_b2/
│   └── yolov26_cls/
├── inference/
│   ├── batch_inference.py           # Batch-optimized, checkpointed inference pipeline
│   └── run_all_models.py            # Orchestrates inference across all four architectures
├── evaluation/
│   ├── confusion_matrices.py        # Normalized confusion matrix generation
│   ├── metrics_report.py            # Accuracy, macro-F1, precision, recall per model
│   ├── domain_shift_delta.py        # Curated vs. uncurated degradation computation
│   └── compound_scaling_stats.py    # Paired statistical test: B0 vs. B2 robustness
├── reports/
│   ├── evaluation_report.md
│   ├── manuscript_draft/            # Methodology and Results sections
│   └── figures/
├── requirements.txt
├── README.md
└── LICENSE
```

## Installation

```bash
git clone https://github.com/AFIFBEJI/aigis-zeroshot-benchmark.git
cd aigis-zeroshot-benchmark
uv venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

Alternatively, skip activation and let `uv` manage the environment per command:

```bash
uv run python inference/run_all_models.py
```

**Core dependencies:** `torch`, `torchvision` (EfficientNet-B0/B2 backbones), `ultralytics` (YOLOv26-Classification), `scikit-learn` (metrics, confusion matrices), `scipy`/`statsmodels`/`pingouin` (paired statistical testing), `pandas`, `matplotlib`/`seaborn`.

## Risks and Limitations

- **Ground-truth quality on uncurated data is the primary methodological risk.** "Uncurated" by definition implies less control over label quality than the prior curated benchmark. This needs to be addressed explicitly in the manuscript — either through a documented annotation protocol for the 21,013 images, inter-annotator agreement reporting, or an honest acknowledgment of label-noise limitations.
- **Dataset licensing and provenance** for the 21,013 uncurated images should be documented (source, collection consent, redistribution rights) before public release, consistent with standard practice for any image dataset shared alongside a publication.
- **Zero-shot framing must be strictly enforced.** Any inadvertent fine-tuning or threshold calibration on the uncurated set would contaminate the "zero-shot" claim central to this study's novelty — the inference pipeline should be architected to make this mistake structurally difficult, not just procedurally discouraged.
- **Statistical power for the compound-scaling claim** depends on how the paired test is constructed; a naive accuracy-difference comparison without confidence intervals or significance testing would not support the "statistically analyze" objective as stated.
- **Comparability to the prior curated-benchmark numbers** assumes those original GROUP2 results are still accessible and computed under comparable conditions (same defect taxonomy, same evaluation protocol) — this should be explicitly verified rather than assumed, since any inconsistency would undermine the domain-shift delta calculation that is this project's central contribution.

## References

- Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. *ICML*.
- Ultralytics (2026). YOLO26: Unified Real-Time End-to-End Vision Models. [docs.ultralytics.com/models/yolo26](https://docs.ultralytics.com/models/yolo26)
- [Aigis-Benchmark-GROUP2](https://github.com/Aigis-AI-SummerInternship2025/Aigis-Benchmark-GROUP2) — prior internship benchmark this project directly extends.

## Status

This repository is in active development, continuing directly from the GROUP2 benchmark. Uncurated dataset characterization and the zero-shot inference pipeline (Phases 1–2) are the current focus; domain-shift quantification, compound-scaling statistical analysis, and manuscript drafting will follow per the methodology above.

---

<sub>Continuation of the Aigis structural defect detection benchmarking initiative. Supervised by Afif Beji (ESPRIT). Contributions, issues, and methodological critique are welcome via GitHub Issues.</sub>
