"""
tools/hash_files.py — taille + SHA-256 de fichiers (poids, archives de datasets) au format Markdown.
Usage : python tools/hash_files.py fichier1 fichier2 ... > lignes.md
Pour un fichier sur Drive : le télécharger (ou le monter dans Colab) puis lancer ce script ;
relever aussi le chemin Drive exact. Pour un dossier d'archives : python tools/hash_files.py archives/* > SHA256SUMS.md
"""
import hashlib, os, sys

for p in sys.argv[1:]:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    print(f"| `{os.path.basename(p)}` | {os.path.getsize(p)} | `{h.hexdigest()}` |")
