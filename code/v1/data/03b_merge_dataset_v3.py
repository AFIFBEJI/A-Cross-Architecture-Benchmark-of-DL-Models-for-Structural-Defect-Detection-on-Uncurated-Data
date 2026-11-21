"""
03b_merge_dataset_v3.py

Fusion finale du dataset Aigis (v1, ~21 010 images) avec les sources externes
vérifiées (MDMCS, RCS 2119, RCS 1841 via crops LabelMe — voir
03a_extract_crops_labelme.py — et Surface Crack Detection), sans doublon,
en conservant le split train/val/test 70/15/15.

Résultat obtenu lors de l'exécution documentée : 35 641 images
(cracks: 10 925, normal: 10 135, corrosion: 7 443, spalling: 7 138).

Prérequis : avoir lancé 03a_extract_crops_labelme.py au préalable, qui
alimente /kaggle/working/all_crops_staging/{corrosion,spalling,cracks}/.
"""

import os
import json
import random
import hashlib
import shutil
from collections import defaultdict
from PIL import Image

EXISTING_DATASET_PATH = "/kaggle/input/datasets/daikhiyassine/dataset"
SFC_PATH = "/kaggle/input/datasets/arunrk7/surface-crack-detection"
CROPS_STAGING = "/kaggle/working/all_crops_staging"
MERGED_OUTPUT_PATH = "/kaggle/working/dataset_v3_final"

SPLITS = ["train", "val", "test"]
CLASSES = ["corrosion", "spalling", "normal", "cracks"]
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

# Pas de plafond pour corrosion/spalling (classes prioritaires à enrichir) ;
# plafond raisonnable pour cracks/normal (déjà abondamment fournis)
CAPS = {"corrosion": None, "spalling": None, "cracks": 5000, "normal": 5000}

TARGET_SIZE = 256
JPEG_QUALITY = 85
random.seed(42)


def get_file_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def index_existing_dataset(dataset_path):
    hashes = set()
    counts = {split: {cls: 0 for cls in CLASSES} for split in SPLITS}
    for split in SPLITS:
        for cls in CLASSES:
            cls_path = os.path.join(dataset_path, split, cls)
            if not os.path.isdir(cls_path):
                continue
            for img_name in os.listdir(cls_path):
                img_path = os.path.join(cls_path, img_name)
                try:
                    hashes.add(get_file_hash(img_path))
                    counts[split][cls] += 1
                except Exception:
                    continue
    return hashes, counts


def collect_new_images(crops_staging, sfc_path):
    """Rassemble les crops (MDMCS/RCS) et Surface Crack Detection, avec tag de source."""
    new_images = defaultdict(list)

    for cls in ["corrosion", "spalling", "cracks"]:
        cls_dir = os.path.join(crops_staging, cls)
        if os.path.isdir(cls_dir):
            for img_name in os.listdir(cls_dir):
                source_tag = img_name.split("_")[0]
                new_images[cls].append((os.path.join(cls_dir, img_name), source_tag))

    sfc_mapping = {"Positive": "cracks", "Negative": "normal"}
    for source_folder, target_class in sfc_mapping.items():
        folder_path = os.path.join(sfc_path, source_folder)
        if not os.path.isdir(folder_path):
            continue
        for img_name in os.listdir(folder_path):
            if img_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                new_images[target_class].append((os.path.join(folder_path, img_name), "sfc"))

    return new_images


def deduplicate(new_images, existing_hashes):
    unique = defaultdict(list)
    stats = {"doublons_avec_existant": 0, "doublons_internes": 0}
    seen = set()

    for cls, entries in new_images.items():
        shuffled = entries.copy()
        random.shuffle(shuffled)
        for img_path, tag in shuffled:
            try:
                h = get_file_hash(img_path)
            except Exception:
                continue
            if h in existing_hashes:
                stats["doublons_avec_existant"] += 1
            elif h in seen:
                stats["doublons_internes"] += 1
            else:
                seen.add(h)
                unique[cls].append((img_path, tag))
    return unique, stats


def split_and_sample(unique_by_class, caps):
    assignment = defaultdict(lambda: defaultdict(list))
    sampling_stats = {}

    for cls, entries in unique_by_class.items():
        n_available = len(entries)
        cap = caps.get(cls)
        n_target = min(cap, n_available) if cap else n_available
        sampled = random.sample(entries, n_target)

        source_breakdown = defaultdict(int)
        for _, tag in sampled:
            source_breakdown[tag] += 1
        sampling_stats[cls] = {
            "disponibles": n_available,
            "retenues": n_target,
            "sources": dict(source_breakdown),
        }

        n_train = int(n_target * SPLIT_RATIOS["train"])
        n_val = int(n_target * SPLIT_RATIOS["val"])
        assignment["train"][cls] = sampled[:n_train]
        assignment["val"][cls] = sampled[n_train:n_train + n_val]
        assignment["test"][cls] = sampled[n_train + n_val:]

    return assignment, sampling_stats


def process_and_save(src_path, dst_path):
    with Image.open(src_path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
        img.save(dst_path, "JPEG", quality=JPEG_QUALITY, optimize=True)


def build_merged_dataset(existing_path, split_assignment, output_path):
    copied_v1 = 0
    for split in SPLITS:
        for cls in CLASSES:
            src_dir = os.path.join(existing_path, split, cls)
            dst_dir = os.path.join(output_path, split, cls)
            os.makedirs(dst_dir, exist_ok=True)
            if not os.path.isdir(src_dir):
                continue
            for img_name in os.listdir(src_dir):
                try:
                    shutil.copy2(os.path.join(src_dir, img_name), os.path.join(dst_dir, img_name))
                    copied_v1 += 1
                except Exception as e:
                    print(f"⚠️ {e}")

    added = 0
    for split in SPLITS:
        for cls in CLASSES:
            dst_dir = os.path.join(output_path, split, cls)
            os.makedirs(dst_dir, exist_ok=True)
            for img_path, tag in split_assignment[split].get(cls, []):
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                dst_path = os.path.join(dst_dir, f"{base_name}.jpg")
                try:
                    process_and_save(img_path, dst_path)
                    added += 1
                except Exception as e:
                    print(f"⚠️ {e}")

    return copied_v1, added


def final_dedup_pass(output_path):
    """Sécurité : re-hash le dataset final assemblé et retire les doublons résiduels."""
    hash_map = defaultdict(list)
    for split in SPLITS:
        for cls in CLASSES:
            cls_path = os.path.join(output_path, split, cls)
            if not os.path.isdir(cls_path):
                continue
            for img_name in os.listdir(cls_path):
                img_path = os.path.join(cls_path, img_name)
                h = get_file_hash(img_path)
                hash_map[h].append(img_path)

    removed = 0
    for h, paths in hash_map.items():
        if len(paths) > 1:
            for p in paths[1:]:
                os.remove(p)
                removed += 1
    return removed


def main():
    print("Étape 1/6 : indexation du dataset existant...")
    existing_hashes, existing_counts = index_existing_dataset(EXISTING_DATASET_PATH)
    print(f"✅ {len(existing_hashes)} images indexées")

    print("\nÉtape 2/6 : collecte des nouvelles images...")
    new_images = collect_new_images(CROPS_STAGING, SFC_PATH)
    for cls, imgs in new_images.items():
        print(f"   {cls}: {len(imgs)} images (toutes sources)")

    print("\nÉtape 3/6 : déduplication...")
    unique_new, dedup_stats = deduplicate(new_images, existing_hashes)
    print(f"   Doublons avec l'existant : {dedup_stats['doublons_avec_existant']}")
    print(f"   Doublons internes : {dedup_stats['doublons_internes']}")

    print("\nÉtape 4/6 : split train/val/test...")
    split_assignment, sampling_stats = split_and_sample(unique_new, CAPS)
    for cls, s in sampling_stats.items():
        print(f"   {cls}: {s['retenues']}/{s['disponibles']} — sources: {s['sources']}")

    print(f"\nÉtape 5/6 : construction du dataset final dans {MERGED_OUTPUT_PATH}...")
    copied_v1, added = build_merged_dataset(EXISTING_DATASET_PATH, split_assignment, MERGED_OUTPUT_PATH)
    print(f"   {copied_v1} images v1 copiées, {added} nouvelles images ajoutées")

    print("\nÉtape 6/6 : déduplication finale de sécurité + documentation...")
    removed = final_dedup_pass(MERGED_OUTPUT_PATH)
    print(f"   {removed} doublons résiduels supprimés")

    final_counts = {}
    for split in SPLITS:
        final_counts[split] = {
            cls: len(os.listdir(os.path.join(MERGED_OUTPUT_PATH, split, cls)))
            for cls in CLASSES
            if os.path.isdir(os.path.join(MERGED_OUTPUT_PATH, split, cls))
        }

    rapport = {
        "sources": [
            {"nom": "MDMCS v2", "licence": "CC BY 4.0", "doi": "10.17632/6x4dzzrs2h.2"},
            {"nom": "Reinforced Concrete Structure Segmentation (2119)", "licence": "CC BY 4.0", "doi": "10.17632/2vkm6k4cfg.1"},
            {"nom": "Reinforced Concrete Structure Segmentation (1841)", "licence": "CC BY 4.0", "doi": "10.17632/yfhwfcrfmk.1"},
            {"nom": "Surface Crack Detection (Özgenel)", "licence": "CC BY 4.0", "doi": "10.17632/5y9wdsg2zt.2"},
        ],
        "stats_echantillonnage": sampling_stats,
        "stats_deduplication": dedup_stats,
        "doublons_residuels_supprimes": removed,
        "images_v1_original": copied_v1,
        "images_nouvelles_nettes": added - removed,
        "images_total_final": copied_v1 + added - removed,
        "comparaison_v1_vs_final": {"v1": existing_counts, "final": final_counts},
    }
    with open(os.path.join(MERGED_OUTPUT_PATH, "rapport_fusion_v3.json"), "w") as f:
        json.dump(rapport, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Dataset final : {MERGED_OUTPUT_PATH}")
    print(f"✅ Total : {copied_v1 + added - removed} images")


if __name__ == "__main__":
    main()
