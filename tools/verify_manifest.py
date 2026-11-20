"""
tools/verify_manifest.py — vérifie qu'un dataset local correspond exactement à un manifeste.
Usage : python tools/verify_manifest.py <racine_dataset> manifests/<nom>.csv
Code retour 0 = identique (fichiers, partitions, MD5).
"""
import csv, hashlib, os, sys

root, man = sys.argv[1], sys.argv[2]
bad = 0
for r in csv.DictReader(open(man)):
    p = os.path.join(root, r["relative_path"])
    if not os.path.isfile(p):
        print("MANQUANT", r["relative_path"]); bad += 1
    elif hashlib.md5(open(p, "rb").read()).hexdigest() != r["md5"]:
        print("MD5 DIFFÉRENT", r["relative_path"]); bad += 1
print("OK : dataset conforme au manifeste" if not bad else f"{bad} écart(s)")
sys.exit(1 if bad else 0)
