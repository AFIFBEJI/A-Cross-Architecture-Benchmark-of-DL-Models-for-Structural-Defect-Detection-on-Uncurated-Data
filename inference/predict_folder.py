"""
inference/predict_folder.py

Inférence (évaluation seule, SANS entraînement) d'un checkpoint déjà entraîné
sur un dossier d'images organisé en <racine>/<classe>/*.jpg.
Écrit un CSV de prédictions (une ligne par image) consommé par evaluation/.

Usage :
  python inference/predict_folder.py --framework keras --model aigis_efficientnetb0_final.keras \
      --arch efficientnet_b0 --data /chemin/test --out predictions_b0.csv
  python inference/predict_folder.py --framework yolo --model best.pt --data /chemin/test --out predictions_yolo.csv

Le dossier --data peut être le test split (benchmark nettoyé) ou un jeu non curé
(si les sous-dossiers portent les étiquettes) : c'est le mode "évaluation seule".
"""
import argparse, os, csv
import numpy as np

CLASSES = ["corrosion", "cracks", "normal", "spalling"]  # ordre alphabétique = ordre Keras/YOLO
EXT = (".jpg", ".jpeg", ".png", ".bmp")


def list_images(root):
    items = []
    for c in sorted(os.listdir(root)):
        d = os.path.join(root, c)
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.lower().endswith(EXT):
                    items.append((os.path.join(d, f), c))
    return items


def run_keras(model_path, arch, items, bs):
    import tensorflow as tf
    if arch == "resnet50":
        from tensorflow.keras.applications.resnet50 import preprocess_input
        size = 224
    elif arch == "efficientnet_b2":
        from tensorflow.keras.applications.efficientnet import preprocess_input
        size = 260
    else:
        from tensorflow.keras.applications.efficientnet import preprocess_input
        size = 224
    model = tf.keras.models.load_model(model_path)
    probs = []
    for i in range(0, len(items), bs):
        batch = []
        for p, _ in items[i:i + bs]:
            img = tf.keras.utils.load_img(p, target_size=(size, size))
            batch.append(preprocess_input(tf.keras.utils.img_to_array(img)))
        probs.append(model.predict(np.stack(batch), verbose=0))
    return np.concatenate(probs)


def run_yolo(model_path, items, bs):
    from ultralytics import YOLO
    model = YOLO(model_path)
    names = model.names
    order = [k for k, _ in sorted(names.items(), key=lambda kv: kv[1])]
    out = []
    for i in range(0, len(items), bs):
        res = model.predict([p for p, _ in items[i:i + bs]], imgsz=224, verbose=False)
        for r in res:
            pr = r.probs.data.cpu().numpy()
            out.append([pr[order.index(k)] for k in sorted(names, key=lambda k: names[k])])
    # remet dans l'ordre alphabétique CLASSES
    idx = [sorted(names.values()).index(c) for c in CLASSES]
    return np.array(out)[:, idx]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--framework", choices=["keras", "yolo"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--arch", default="efficientnet_b0", choices=["efficientnet_b0", "efficientnet_b2", "resnet50"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    items = list_images(a.data)
    probs = run_keras(a.model, a.arch, items, a.batch) if a.framework == "keras" else run_yolo(a.model, items, a.batch)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "true", "pred"] + [f"p_{c}" for c in CLASSES])
        for (p, t), pr in zip(items, probs):
            w.writerow([p, t, CLASSES[int(np.argmax(pr))]] + [f"{x:.6f}" for x in pr])
    print(f"{len(items)} prédictions -> {a.out}")
