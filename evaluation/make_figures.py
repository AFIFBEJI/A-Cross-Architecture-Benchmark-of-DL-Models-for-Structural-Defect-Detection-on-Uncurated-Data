"""
evaluation/make_figures.py
Courbes d'apprentissage à partir des CSV d'entraînement (Keras phase1+phase2, ou Ultralytics results.csv).

Usage : python evaluation/make_figures.py --keras models/efficientnet_b0/v1/Resultats/training_log_b0_phase1.csv models/efficientnet_b0/v1/Resultats/training_log_b0_phase2.csv --out curves_b0.png
        python evaluation/make_figures.py --yolo models/yolov26_cls/v1/Resultats/results.csv --out curves_yolo.png
"""
import argparse
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--keras", nargs=2); ap.add_argument("--yolo"); ap.add_argument("--out", required=True)
a = ap.parse_args()
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
if a.keras:
    d = pd.concat([pd.read_csv(f) for f in a.keras], ignore_index=True)
    ax[0].plot(d["loss"], label="train"); ax[0].plot(d["val_loss"], label="val")
    ax[1].plot(d["accuracy"], label="train"); ax[1].plot(d["val_accuracy"], label="val")
    n1 = len(pd.read_csv(a.keras[0]))
    for x in ax: x.axvline(n1 - 0.5, ls="--", c="gray")  # frontière phase 1 / phase 2
else:
    d = pd.read_csv(a.yolo); d.columns = [c.strip() for c in d.columns]
    ax[0].plot(d["train/loss"], label="train"); ax[0].plot(d["val/loss"], label="val")
    ax[1].plot(d["metrics/accuracy_top1"], label="val top1")
ax[0].set_title("Loss"); ax[1].set_title("Accuracy"); [x.legend() or x.set_xlabel("epoch") for x in ax]
plt.tight_layout(); plt.savefig(a.out, dpi=150)
