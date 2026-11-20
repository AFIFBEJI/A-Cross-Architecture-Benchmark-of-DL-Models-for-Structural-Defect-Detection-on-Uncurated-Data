"""
evaluation/mcnemar_test.py
Test de McNemar apparié + IC bootstrap 95 % sur l'écart d'accuracy entre deux modèles évalués
sur LES MÊMES images (deux CSV de predict_folder.py, même ordre de fichiers).

Usage : python evaluation/mcnemar_test.py predictions_b0.csv predictions_b2.csv
"""
import sys
import numpy as np, pandas as pd
from scipy.stats import binomtest

a, b = pd.read_csv(sys.argv[1]), pd.read_csv(sys.argv[2])
assert (a["path"].values == b["path"].values).all(), "Les deux CSV doivent couvrir les mêmes images dans le même ordre"
ca, cb = (a["true"] == a["pred"]).values, (b["true"] == b["pred"]).values
n01, n10 = int((~ca & cb).sum()), int((ca & ~cb).sum())  # A faux/B juste ; A juste/B faux
p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0  # McNemar exact
rng = np.random.default_rng(42)
diffs = [(cb[i].mean() - ca[i].mean()) for i in (rng.integers(0, len(ca), len(ca)) for _ in range(2000))]
print(f"acc A={ca.mean():.4f}  acc B={cb.mean():.4f}  n01={n01} n10={n10}  McNemar exact p={p:.4g}")
print(f"écart acc (B-A) IC95% bootstrap = [{np.percentile(diffs,2.5):.4f}, {np.percentile(diffs,97.5):.4f}]  (seed 42, 2000 rééchantillons)")
