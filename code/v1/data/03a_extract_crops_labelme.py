"""
03a_extract_crops_labelme.py

Extraction de crops de classification depuis des datasets annotés au format
LabelMe (shapes/points/label), utilisé pour MDMCS et les datasets
"Reinforced Concrete Structure Segmentation" (2119 et 1841).

Une image source peut contenir plusieurs zones annotées de classes
différentes (dataset multi-dommages) : chaque zone donne un crop distinct.
"""

import os
import json
from PIL import Image

PADDING_RATIO = 0.15   # marge de contexte autour de chaque polygone
MIN_CROP_SIZE = 40     # ignore les zones annotées trop petites (bruit)


def extract_crops_labelme(img_dir, json_dir, mapping, source_tag, output_base):
    """
    Parcourt un dossier d'annotations LabelMe et extrait un crop JPEG par
    shape annotée, dans le dossier de la classe cible correspondante.

    mapping : dict {label_source (minuscules) -> classe_cible}
              les labels absents du mapping sont exclus (comptés dans 'exclus')
    """
    stats = {cls: 0 for cls in set(mapping.values())}
    stats["exclus"] = 0
    skipped_too_small = 0
    missing_image = 0
    skipped_invalid_box = 0

    # Index des images par nom de base, insensible à l'extension exacte
    img_index = {}
    for fname in os.listdir(img_dir):
        base = os.path.splitext(fname)[0]
        img_index[base] = fname

    for json_name in os.listdir(json_dir):
        if not json_name.endswith(".json"):
            continue
        base_name = os.path.splitext(json_name)[0]

        if base_name not in img_index:
            missing_image += 1
            continue

        img_path = os.path.join(img_dir, img_index[base_name])
        json_path = os.path.join(json_dir, json_name)

        with open(json_path) as f:
            try:
                annotation = json.load(f)
            except Exception:
                continue

        shapes = annotation.get("shapes", [])

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        img_w, img_h = img.size

        for shape_idx, shape in enumerate(shapes):
            label = shape.get("label", "").lower().strip()
            target_class = mapping.get(label)
            if target_class is None:
                stats["exclus"] += 1
                continue

            points = shape.get("points", [])
            if len(points) < 2:
                continue

            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            box_w, box_h = x_max - x_min, y_max - y_min

            if box_w < MIN_CROP_SIZE or box_h < MIN_CROP_SIZE:
                skipped_too_small += 1
                continue

            pad_x, pad_y = box_w * PADDING_RATIO, box_h * PADDING_RATIO
            crop_x_min = max(0, x_min - pad_x)
            crop_y_min = max(0, y_min - pad_y)
            crop_x_max = min(img_w, x_max + pad_x)
            crop_y_max = min(img_h, y_max + pad_y)

            if crop_x_max <= crop_x_min or crop_y_max <= crop_y_min:
                skipped_invalid_box += 1
                continue

            try:
                crop = img.crop((crop_x_min, crop_y_min, crop_x_max, crop_y_max))
            except Exception:
                skipped_invalid_box += 1
                continue

            out_dir = os.path.join(output_base, target_class)
            os.makedirs(out_dir, exist_ok=True)
            out_name = f"{source_tag}_{base_name}_shape{shape_idx}.jpg"
            crop.save(os.path.join(out_dir, out_name), "JPEG", quality=90)
            stats[target_class] += 1

    stats["images_manquantes"] = missing_image
    stats["crops_trop_petits"] = skipped_too_small
    stats["boites_invalides"] = skipped_invalid_box
    return stats


# ============================================================
# Mappings label -> classe cible, spécifiques à chaque dataset source
# ============================================================

MDMCS_MAPPING = {
    "spalling": "spalling",
    "corrosion": "corrosion",
    "crack": "cracks",
    # "exposed rebar" volontairement exclu (pas de classe correspondante)
}

RCS_2119_MAPPING = {
    "crack": "cracks",
    "rebar corrosion": "corrosion",
    "concrete spalling": "spalling",
    # "rebar exposure", "concrete crushing" exclus
}

RCS_1841_MAPPING = {
    "structural crack": "cracks",
    "microcrack": "cracks",
    "rebar corrosion": "corrosion",
    "minor spalling": "spalling",
    "moderate spalling": "spalling",
    # "delamination", "structural deformation", "concrete crushing" exclus
    # (delamination jugé conceptuellement distinct du spalling)
}


def main():
    CROPS_OUTPUT = "/kaggle/working/all_crops_staging"

    # --- MDMCS (a un sous-split train/val/test propre) ---
    MDMCS_BASE = (
        "/kaggle/input/datasets/daikhiyassine/mdmcs-dataset/"
        "MDMCS A Benchmark Dataset for Multi-Damage Monitor/HRCDS/HRCDS"
    )
    print("=== Extraction MDMCS ===")
    for split in ["train", "val", "test"]:
        img_dir = os.path.join(MDMCS_BASE, f"{split}_image")
        ann_dir = os.path.join(MDMCS_BASE, f"{split}_annotations")
        if not os.path.isdir(ann_dir):
            continue
        stats = extract_crops_labelme(img_dir, ann_dir, MDMCS_MAPPING, f"mdmcs{split}", CROPS_OUTPUT)
        print(f"  {split}:", stats)

    # --- RCS 2119 ---
    RCS_2119_BASE = "/kaggle/input/datasets/daikhiyassine/reinforced-concrete-structure-segmentation-2119"
    print("\n=== Extraction RCS 2119 ===")
    stats_2119 = extract_crops_labelme(
        os.path.join(RCS_2119_BASE, "imagedata (1)"),
        os.path.join(RCS_2119_BASE, "jsondata"),
        RCS_2119_MAPPING, "rcs2119", CROPS_OUTPUT,
    )
    print(stats_2119)

    # --- RCS 1841 ---
    RCS_1841_BASE = "/kaggle/input/datasets/daikhiyassine/reinforced-concrete-structure-segmentation-1841"
    print("\n=== Extraction RCS 1841 ===")
    stats_1841 = extract_crops_labelme(
        os.path.join(RCS_1841_BASE, "imagedata"),
        os.path.join(RCS_1841_BASE, "labeldata"),
        RCS_1841_MAPPING, "rcs1841", CROPS_OUTPUT,
    )
    print(stats_1841)

    print(f"\n✅ Tous les crops sauvegardés dans : {CROPS_OUTPUT}")


if __name__ == "__main__":
    main()
