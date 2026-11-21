"""
02_clean_dataset.py

Nettoyage du dataset Aigis : détection de doublons (hash MD5), vérification
des fuites entre splits train/val/test, conversion RGB, resize et compression.

Usage (dans un notebook Kaggle) :
    DATASET_PATH = "/kaggle/input/datasets/daikhiyassine/dataset"
    CLEAN_PATH = "/kaggle/working/dataset_clean"  (variable d'environnement CLEAN_PATH pour changer)
"""

import os
import hashlib
import json
from collections import defaultdict
from PIL import Image

DATASET_PATH = "/kaggle/input/datasets/daikhiyassine/dataset"
CLEAN_PATH = os.environ.get("CLEAN_PATH", "/kaggle/working/dataset_clean")  # ex. CLEAN_PATH=/kaggle/working/dataset_clean_b0
SPLITS = ["train", "val", "test"]
CLASSES = ["corrosion", "spalling", "normal", "cracks"]
SPLIT_PRIORITY = {"train": 0, "val": 1, "test": 2}

TARGET_SIZE = 256
JPEG_QUALITY = 85


def get_file_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def find_duplicates(dataset_path):
    """Indexe tous les fichiers par hash et identifie doublons + fuites train/val/test."""
    hash_map = defaultdict(list)
    for split in SPLITS:
        for cls in CLASSES:
            cls_path = os.path.join(dataset_path, split, cls)
            if not os.path.isdir(cls_path):
                continue
            for img_name in os.listdir(cls_path):
                img_path = os.path.join(cls_path, img_name)
                h = get_file_hash(img_path)
                hash_map[h].append((split, cls, img_name, img_path))

    files_to_skip = set()
    leak_count = 0
    duplicates = {h: v for h, v in hash_map.items() if len(v) > 1}

    for h, entries in duplicates.items():
        entries_sorted = sorted(entries, key=lambda e: SPLIT_PRIORITY[e[0]])
        remove = entries_sorted[1:]
        for split, cls, img_name, img_path in remove:
            files_to_skip.add(img_path)
        if len(set(e[0] for e in entries)) > 1:
            leak_count += 1

    return files_to_skip, leak_count, len(duplicates)


def clean_and_resize(dataset_path, clean_path, files_to_skip):
    """Copie le dataset en excluant les doublons, convertit en RGB, resize + compresse."""
    converted_count = 0
    copied_count = 0

    for split in SPLITS:
        for cls in CLASSES:
            src_dir = os.path.join(dataset_path, split, cls)
            dst_dir = os.path.join(clean_path, split, cls)
            os.makedirs(dst_dir, exist_ok=True)
            if not os.path.isdir(src_dir):
                continue
            for img_name in os.listdir(src_dir):
                src_path = os.path.join(src_dir, img_name)
                if src_path in files_to_skip:
                    continue
                base_name = os.path.splitext(img_name)[0]
                dst_path = os.path.join(dst_dir, base_name + ".jpg")
                try:
                    with Image.open(src_path) as img:
                        if img.mode != "RGB":
                            img = img.convert("RGB")
                            converted_count += 1
                        img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
                        img.save(dst_path, "JPEG", quality=JPEG_QUALITY, optimize=True)
                        copied_count += 1
                except Exception as e:
                    print(f"   ⚠️ Erreur sur {src_path} : {e}")

    return copied_count, converted_count


def main():
    print("Étape 1/3 : identification des doublons...")
    files_to_skip, leak_count, n_duplicate_groups = find_duplicates(DATASET_PATH)
    print(f"Doublons à exclure : {len(files_to_skip)} | Fuites entre splits : {leak_count}")

    print(f"\nÉtape 2/3 : resize à {TARGET_SIZE}x{TARGET_SIZE} + compression JPEG q={JPEG_QUALITY}...")
    copied, converted = clean_and_resize(DATASET_PATH, CLEAN_PATH, files_to_skip)
    print(f"Images traitées : {copied} | Converties RGB : {converted}")

    print("\nÉtape 3/3 : rapport final...")
    final_counts = {}
    for split in SPLITS:
        final_counts[split] = {
            cls: len(os.listdir(os.path.join(CLEAN_PATH, split, cls)))
            for cls in CLASSES
            if os.path.isdir(os.path.join(CLEAN_PATH, split, cls))
        }
        print(f"{split}: {final_counts[split]}")

    rapport = {
        "target_size": TARGET_SIZE,
        "jpeg_quality": JPEG_QUALITY,
        "doublons_exclus": len(files_to_skip),
        "fuites_entre_splits": leak_count,
        "images_converties_rgb": converted,
        "comptage_final": final_counts,
    }
    with open(os.path.join(CLEAN_PATH, "rapport_nettoyage_final.json"), "w") as f:
        json.dump(rapport, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Dataset nettoyé : {CLEAN_PATH}")


if __name__ == "__main__":
    main()
