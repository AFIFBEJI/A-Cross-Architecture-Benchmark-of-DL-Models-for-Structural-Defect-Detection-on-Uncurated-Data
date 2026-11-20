"""
models/yolo26/test.py

Évaluation de YOLO26s-cls sur le test set : classification report,
matrices de confusion, AUC macro.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

DATASET_PATH = "/kaggle/working/dataset_clean_yolo26"
MODEL_PATH = "/kaggle/working/yolo26_train/aigis_yolo26_final/weights/best.pt"
IMG_SIZE = 224


def main():
    model = YOLO(MODEL_PATH)

    print("Évaluation native Ultralytics sur le split test...")
    metrics = model.val(
        data=DATASET_PATH, split="test", imgsz=IMG_SIZE,
        project="/kaggle/working/yolo26_test", name="aigis_yolo26_test",
        exist_ok=True, plots=True,
    )
    print(f"Top-1 Accuracy : {metrics.top1:.4f} | Top-5 Accuracy : {metrics.top5:.4f}")

    test_dir = os.path.join(DATASET_PATH, "test")
    class_names = sorted(os.listdir(test_dir))

    y_true, y_pred, y_pred_probs = [], [], []
    for cls_idx, cls_name in enumerate(class_names):
        cls_dir = os.path.join(test_dir, cls_name)
        for img_name in os.listdir(cls_dir):
            result = model.predict(os.path.join(cls_dir, img_name), imgsz=IMG_SIZE, verbose=False)[0]
            probs = result.probs.data.cpu().numpy()
            y_true.append(cls_idx)
            y_pred.append(int(np.argmax(probs)))
            y_pred_probs.append(probs)

    y_true, y_pred, y_pred_probs = np.array(y_true), np.array(y_pred), np.array(y_pred_probs)

    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    y_true_onehot = np.eye(len(class_names))[y_true]
    auc_macro = roc_auc_score(y_true_onehot, y_pred_probs, average="macro", multi_class="ovr")
    print(f"AUC macro : {auc_macro:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Prédiction"); plt.ylabel("Vraie classe")
    plt.title("Matrice de confusion - YOLO26 (test set)")
    plt.tight_layout()
    plt.savefig("/kaggle/working/confusion_matrix_yolo26.png", dpi=150, bbox_inches="tight")
    plt.show()

    results = {
        "top1_accuracy": float(metrics.top1), "top5_accuracy": float(metrics.top5),
        "auc_macro_sklearn": float(auc_macro),
        "classification_report": classification_report(y_true, y_pred, target_names=class_names, output_dict=True),
        "confusion_matrix": cm.tolist(),
    }
    with open("/kaggle/working/test_results_yolo26.json", "w") as f:
        json.dump(results, f, indent=2)
    print("✅ Résultats sauvegardés")


if __name__ == "__main__":
    main()
