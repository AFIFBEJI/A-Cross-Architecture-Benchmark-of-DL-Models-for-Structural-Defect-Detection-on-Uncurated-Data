"""
02_clean_dataset_b2_fullres.py

Nettoyage v1 tel que réellement exécuté pour EfficientNet-B2 (notebook d'origine :
notebooks/v1/v1_efficientnet_b2_cleaning_kaggle_original.ipynb, cellule « 393ee103 », exécuté le 2026-07-28).
Différence avec 02_clean_dataset.py (variante B0/YOLO) : ici PAS de redimensionnement ni de recompression :
les images gardent leur taille et leur format d'origine (seule la conversion RGB est appliquée).
Le code est celui de la cellule, plus un seul ajout : CLEAN_PATH surchargeable par variable d'environnement.
"""
import os
import shutil
import hashlib
from PIL import Image
import json

DATASET_PATH = "/kaggle/input/datasets/daikhiyassine/dataset"
import os as _os
CLEAN_PATH = _os.environ.get("CLEAN_PATH", "/kaggle/working/dataset_clean")
SPLITS = ["train", "val", "test"]
CLASSES = ["corrosion", "spalling", "normal", "cracks"]

# Ordre de priorité si un doublon existe entre splits : on garde train, on retire de val/test
SPLIT_PRIORITY = {"train": 0, "val": 1, "test": 2}

# ============================================================
# 1️⃣ Recalculer les hashs pour identifier précisément quoi retirer
# ============================================================
print("Étape 1/3 : identification des doublons à retirer...")

def get_file_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

from collections import defaultdict
hash_map = defaultdict(list)

for split in SPLITS:
    for cls in CLASSES:
        cls_path = os.path.join(DATASET_PATH, split, cls)
        if not os.path.isdir(cls_path):
            continue
        for img_name in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img_name)
            h = get_file_hash(img_path)
            hash_map[h].append((split, cls, img_name, img_path))

# Pour chaque groupe de doublons, on garde 1 seule copie (priorité train > val > test)
files_to_skip = set()  # chemins à NE PAS copier

for h, entries in hash_map.items():
    if len(entries) > 1:
        # trie par priorité de split, garde le premier, skip les autres
        entries_sorted = sorted(entries, key=lambda e: SPLIT_PRIORITY[e[0]])
        keep = entries_sorted[0]
        remove = entries_sorted[1:]
        for split, cls, img_name, img_path in remove:
            files_to_skip.add(img_path)
            print(f"   ❌ Retiré : {split}/{cls}/{img_name} (doublon de {keep[0]}/{keep[1]}/{keep[2]})")

print(f"\nTotal fichiers à exclure : {len(files_to_skip)}")

# ============================================================
# 2️⃣ Copier le dataset en excluant doublons + convertir non-RGB
# ============================================================
print("\nÉtape 2/3 : copie + conversion RGB...")

converted_count = 0
copied_count = 0

for split in SPLITS:
    for cls in CLASSES:
        src_dir = os.path.join(DATASET_PATH, split, cls)
        dst_dir = os.path.join(CLEAN_PATH, split, cls)
        os.makedirs(dst_dir, exist_ok=True)

        if not os.path.isdir(src_dir):
            continue

        for img_name in os.listdir(src_dir):
            src_path = os.path.join(src_dir, img_name)

            if src_path in files_to_skip:
                continue  # doublon exclu

            dst_path = os.path.join(dst_dir, img_name)

            try:
                with Image.open(src_path) as img:
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                        converted_count += 1
                    img.save(dst_path)
                    copied_count += 1
            except Exception as e:
                print(f"   ⚠️ Erreur sur {src_path} : {e}")

print(f"\nImages copiées : {copied_count}")
print(f"Images converties en RGB : {converted_count}")

# ============================================================
# 3️⃣ Vérification finale
# ============================================================
print("\nÉtape 3/3 : vérification finale...")

final_counts = {}
for split in SPLITS:
    final_counts[split] = {}
    for cls in CLASSES:
        cls_path = os.path.join(CLEAN_PATH, split, cls)
        n = len(os.listdir(cls_path)) if os.path.isdir(cls_path) else 0
        final_counts[split][cls] = n

print("\nComptage final (dataset nettoyé) :")
for split in SPLITS:
    total = sum(final_counts[split].values())
    print(f"\n{split.upper()} (total: {total})")
    for cls in CLASSES:
        print(f"   {cls}: {final_counts[split][cls]}")

# Sauvegarde du rapport de nettoyage
rapport_clean = {
    "fichiers_exclus_doublons": list(files_to_skip),
    "images_converties_rgb": converted_count,
    "comptage_final": final_counts
}
with open('/kaggle/working/rapport_nettoyage_final.json', 'w') as f:
    json.dump(rapport_clean, f, indent=2, ensure_ascii=False)

print(f"\n✅ Dataset nettoyé disponible dans : {CLEAN_PATH}")
print("✅ Rapport sauvegardé dans /kaggle/working/rapport_nettoyage_final.json")
