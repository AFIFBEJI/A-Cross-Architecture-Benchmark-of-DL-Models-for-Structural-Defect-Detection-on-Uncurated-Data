"""
models/yolo26/train.py

Entraînement final YOLO26s-cls avec les meilleurs hyperparamètres issus
de tune.py. Meilleur modèle du projet Aigis (94.70% accuracy, AUC 0.9942).
"""

import json
from ultralytics import YOLO

DATASET_PATH = "/kaggle/working/dataset_clean_yolo26"
IMG_SIZE = 224
EPOCHS = 50
BATCH_SIZE = 32


def main():
    with open("/kaggle/working/best_hyperparameters_yolo26.json") as f:
        hp = json.load(f)
    print("Hyperparamètres utilisés :", hp)

    model = YOLO("yolo26s-cls.pt")

    model.train(
        data=DATASET_PATH,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        optimizer="AdamW",
        patience=10,
        project="/kaggle/working/yolo26_train",
        name="aigis_yolo26_final",
        exist_ok=True,
        plots=True,
        save=True,
        val=True,
        lr0=hp["lr0"], lrf=hp["lrf"], momentum=hp["momentum"],
        weight_decay=hp["weight_decay"], warmup_epochs=hp["warmup_epochs"],
        warmup_momentum=hp["warmup_momentum"], box=hp["box"], cls=hp["cls"],
        dfl=hp["dfl"], hsv_h=hp["hsv_h"], hsv_s=hp["hsv_s"], hsv_v=hp["hsv_v"],
        degrees=hp["degrees"], translate=hp["translate"], scale=hp["scale"],
        shear=hp["shear"], perspective=hp["perspective"], flipud=hp["flipud"],
        fliplr=hp["fliplr"], mosaic=hp["mosaic"], mixup=hp["mixup"],
        close_mosaic=int(hp["close_mosaic"]),
    )

    best_model_path = "/kaggle/working/yolo26_train/aigis_yolo26_final/weights/best.pt"
    print(f"\n✅ Meilleur modèle : {best_model_path}")

    # Export ONNX (le TFLite échoue avec le downgrade PyTorch requis par le GPU P100,
    # voir docs/environnement_technique.md) — refaire l'export TFLite sur T4 si besoin.
    model_best = YOLO(best_model_path)
    model_best.export(format="onnx", imgsz=IMG_SIZE)
    print("✅ Modèle exporté en ONNX")


if __name__ == "__main__":
    main()
