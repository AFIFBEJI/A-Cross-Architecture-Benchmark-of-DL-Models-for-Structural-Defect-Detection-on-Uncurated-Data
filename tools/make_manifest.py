"""
tools/make_manifest.py — manifeste de partitionnement (chemin relatif, classe, partition, MD5).

Usage : python tools/make_manifest.py <racine_dataset> manifests/<nom>.csv
  <racine_dataset> contient train|val|test / <classe> / images
Affiche les effectifs par partition pour comparaison avec les chiffres rapportés
(v1 nettoyé : 14 707 / 3 151 / 3 152).
"""
import csv, hashlib, os, sys
from collections import Counter

root, out = sys.argv[1], sys.argv[2]
EXT = (".jpg", ".jpeg", ".png", ".bmp")
rows = []
for split in ("train", "val", "test"):
    for cls in sorted(os.listdir(os.path.join(root, split))) if os.path.isdir(os.path.join(root, split)) else []:
        d = os.path.join(root, split, cls)
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(EXT):
                p = os.path.join(d, f)
                rows.append((os.path.relpath(p, root).replace(os.sep, "/"), cls, split, hashlib.md5(open(p, "rb").read()).hexdigest()))
with open(out, "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["relative_path", "class", "split", "md5"]); w.writerows(rows)
c = Counter(r[2] for r in rows)
print(f"{len(rows)} images -> {out} | train={c['train']} val={c['val']} test={c['test']}")
print("par classe/partition:", dict(Counter((r[2], r[1]) for r in rows)))
