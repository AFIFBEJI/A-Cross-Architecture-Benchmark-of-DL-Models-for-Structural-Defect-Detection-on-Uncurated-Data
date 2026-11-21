"""
01_visualize_dataset.py

Visualisation du dataset Aigis : distribution des classes par split,
échantillons visuels, analyse des dimensions d'images.
"""

import os
import random
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

DATASET_PATH = "/kaggle/input/datasets/daikhiyassine/dataset"
OUTPUT_DIR = "/kaggle/working"
SPLITS = ["train", "val", "test"]
CLASSES = ["corrosion", "spalling", "normal", "cracks"]


def count_images(dataset_path):
    counts = {split: {} for split in SPLITS}
    for split in SPLITS:
        for cls in CLASSES:
            cls_path = os.path.join(dataset_path, split, cls)
            counts[split][cls] = len(os.listdir(cls_path)) if os.path.isdir(cls_path) else 0
    return counts


def plot_distribution(counts, output_path):
    x = np.arange(len(CLASSES))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, split in enumerate(SPLITS):
        values = [counts[split][cls] for cls in CLASSES]
        bars = ax.bar(x + i * width, values, width, label=split)
        ax.bar_label(bars, fontsize=8)
    ax.set_xlabel("Classe")
    ax.set_ylabel("Nombre d'images")
    ax.set_title("Distribution des classes par split - Dataset Aigis", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(CLASSES)
    ax.legend(title="Split")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_samples(dataset_path, output_path, n_samples=4):
    fig, axes = plt.subplots(len(CLASSES), n_samples, figsize=(n_samples * 3, len(CLASSES) * 3))
    for i, cls in enumerate(CLASSES):
        cls_path = os.path.join(dataset_path, "train", cls)
        images = os.listdir(cls_path)
        sample_imgs = random.sample(images, min(n_samples, len(images)))
        for j, img_name in enumerate(sample_imgs):
            img = Image.open(os.path.join(cls_path, img_name))
            ax = axes[i, j]
            ax.imshow(img)
            ax.axis("off")
            if j == 0:
                ax.set_ylabel(cls, fontsize=12, fontweight="bold", rotation=90)
            ax.set_title(f"{img.size[0]}x{img.size[1]}", fontsize=9)
    plt.suptitle("Échantillons du dataset (train) par classe", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()


def main():
    counts = count_images(DATASET_PATH)
    print("Résumé :")
    for split in SPLITS:
        total = sum(counts[split].values())
        print(f"\n{split.upper()} (total: {total})")
        for cls in CLASSES:
            print(f"   {cls}: {counts[split][cls]}")

    plot_distribution(counts, os.path.join(OUTPUT_DIR, "distribution_classes.png"))
    plot_samples(DATASET_PATH, os.path.join(OUTPUT_DIR, "echantillons_dataset.png"))


if __name__ == "__main__":
    main()
