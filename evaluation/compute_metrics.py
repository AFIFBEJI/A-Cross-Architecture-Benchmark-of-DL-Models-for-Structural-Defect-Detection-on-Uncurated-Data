"""
evaluation/compute_metrics.py
Calcule accuracy, P/R/F1 par classe, macro-F1, AUC macro (one-vs-rest) et matrices de
confusion (brute + normalisée) à partir d'un CSV produit par inference/predict_folder.py.

Usage : python evaluation/compute_metrics.py predictions_b0.csv --tag b0 --outdir evaluation/out
"""
import argparse, json, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

CLASSES = ["corrosion", "cracks", "normal", "spalling"]

ap = argparse.ArgumentParser()
ap.add_argument("csv"); ap.add_argument("--tag", required=True); ap.add_argument("--outdir", default="evaluation/out")
a = ap.parse_args()
os.makedirs(a.outdir, exist_ok=True)
df = pd.read_csv(a.csv)
y, p = df["true"], df["pred"]
present = sorted(set(y))
rep = classification_report(y, p, labels=CLASSES, output_dict=True, zero_division=0)
res = {"n": int(len(df)), "accuracy": float((y == p).mean()), "classes_presentes_dans_la_verite_terrain": present,
       "classification_report": rep}
if len(present) == len(CLASSES):  # AUC seulement si les 4 classes existent
    res["auc_macro_ovr"] = float(roc_auc_score(y, df[[f"p_{c}" for c in CLASSES]].values, multi_class="ovr", labels=CLASSES))
else:
    res["avertissement"] = f"Seules {present} sont présentes : accuracy non représentative, AUC non calculée."
json.dump(res, open(f"{a.outdir}/metrics_{a.tag}.json", "w"), indent=2, ensure_ascii=False)
cm = confusion_matrix(y, p, labels=CLASSES)
for norm, name in [(False, ""), (True, "_normalized")]:
    m = cm / np.maximum(cm.sum(1, keepdims=True), 1) if norm else cm
    plt.figure(figsize=(6, 5)); sns.heatmap(m, annot=True, fmt=".2f" if norm else "d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel("Prédit"); plt.ylabel("Réel"); plt.title(f"Matrice de confusion {a.tag}")
    plt.savefig(f"{a.outdir}/confusion_matrix{name}_{a.tag}.png", dpi=150, bbox_inches="tight"); plt.close()
print(json.dumps({k: v for k, v in res.items() if k != "classification_report"}, indent=2, ensure_ascii=False))
