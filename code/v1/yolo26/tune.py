"""
models/yolo26/tune.py

Recherche d'hyperparamètres pour YOLO26 (classification) via l'algorithme
génétique natif d'Ultralytics.

⚠️ Point de vigilance environnement (voir docs/environnement_technique.md) :
sur GPU Tesla P100 (architecture Pascal, sm_60), l'image Kaggle par défaut
installe PyTorch 2.10+ qui ne supporte plus sm_60. Il faut downgrader
PyTorch AVANT d'installer ultralytics :

    pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 \
        --index-url https://download.pytorch.org/whl/cu121
    pip install ultralytics

...puis redémarrer le kernel avant de lancer ce script.
Sur GPU T4, aucun downgrade n'est nécessaire.
"""

import os
import json
import yaml
from ultralytics import YOLO

DATASET_PATH = "/kaggle/working/dataset_clean_yolo26"
IMG_SIZE = 224
EPOCHS_SEARCH = 10
ITERATIONS = 15


def main():
    model = YOLO("yolo26n-cls.pt")

    print("Lancement de la recherche d'hyperparamètres YOLO26...")
    model.tune(
        data=DATASET_PATH,
        epochs=EPOCHS_SEARCH,
        iterations=ITERATIONS,
        imgsz=IMG_SIZE,
        batch=32,
        optimizer="AdamW",
        plots=False,
        save=False,
        val=True,
        project="/kaggle/working/yolo26_tune",
        name="aigis_yolo26_search",
        exist_ok=True,
    )

    best_hp_path = "/kaggle/working/yolo26_tune/aigis_yolo26_search/best_hyperparameters.yaml"
    if os.path.exists(best_hp_path):
        with open(best_hp_path) as f:
            best_hp = yaml.safe_load(f)
        print("\nMEILLEURS HYPERPARAMÈTRES - YOLO26")
        for k, v in best_hp.items():
            print(f"   {k}: {v}")
        with open("/kaggle/working/best_hyperparameters_yolo26.json", "w") as f:
            json.dump(best_hp, f, indent=2)
        print("\n✅ Sauvegardé : /kaggle/working/best_hyperparameters_yolo26.json")
    else:
        print("⚠️ Fichier de résultats non trouvé — vérifie /kaggle/working/yolo26_tune/")


if __name__ == "__main__":
    main()
