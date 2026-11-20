"""
models/efficientnetb2/test.py

Évaluation d'EfficientNetB2 sur le test set : classification report,
matrices de confusion, AUC macro.
"""

import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

DATASET_PATH = "/kaggle/working/dataset_clean_b2"
MODEL_PATH = "/kaggle/working/aigis_efficientnetb2_final.keras"
IMG_SIZE = 260
BATCH_SIZE = 32


def main():
    model = tf.keras.models.load_model(MODEL_PATH)

    test_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATASET_PATH, "test"), image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, label_mode="categorical", shuffle=False,
    )
    class_names = test_ds.class_names

    def preprocess(images, labels):
        return preprocess_input(images), labels

    test_ds_processed = test_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)

    y_true, y_pred_probs = [], []
    for images, labels in test_ds_processed:
        preds = model.predict(images, verbose=0)
        y_pred_probs.extend(preds)
        y_true.extend(labels.numpy())

    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_true_classes = np.argmax(y_true, axis=1)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)

    print(classification_report(y_true_classes, y_pred_classes, target_names=class_names, digits=4))

    test_loss, test_acc, test_auc = model.evaluate(test_ds_processed, verbose=0)
    auc_macro = roc_auc_score(y_true, y_pred_probs, average="macro", multi_class="ovr")
    print(f"Test Accuracy : {test_acc:.4f} | AUC macro : {auc_macro:.4f}")

    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Prédiction"); plt.ylabel("Vraie classe")
    plt.title("Matrice de confusion - EfficientNetB2 (test set)")
    plt.tight_layout()
    plt.savefig("/kaggle/working/confusion_matrix_efficientnetb2.png", dpi=150, bbox_inches="tight")
    plt.show()

    results = {
        "test_loss": float(test_loss), "test_accuracy": float(test_acc),
        "test_auc": float(test_auc), "auc_macro_sklearn": float(auc_macro),
        "classification_report": classification_report(y_true_classes, y_pred_classes, target_names=class_names, output_dict=True),
        "confusion_matrix": cm.tolist(),
    }
    with open("/kaggle/working/test_results_efficientnetb2.json", "w") as f:
        json.dump(results, f, indent=2)
    print("✅ Résultats sauvegardés")


if __name__ == "__main__":
    main()
