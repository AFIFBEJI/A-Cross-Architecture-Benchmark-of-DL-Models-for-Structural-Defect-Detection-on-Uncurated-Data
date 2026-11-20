"""
models/efficientnetb0/train.py

Entraînement EfficientNetB0 en 2 phases (tête gelée puis fine-tuning),
avec les hyperparamètres issus de tune.py.
"""

import os
import json
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import layers, models, optimizers, callbacks

DATASET_PATH = "/kaggle/working/dataset_clean_b0"
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 4
EPOCHS_PHASE1 = 15
EPOCHS_PHASE2 = 20


def load_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATASET_PATH, "train"), image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, label_mode="categorical", shuffle=True, seed=42,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATASET_PATH, "val"), image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, label_mode="categorical", shuffle=False,
    )

    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ])

    def prepare_train(images, labels):
        return preprocess_input(data_augmentation(images)), labels

    def prepare_val(images, labels):
        return preprocess_input(images), labels

    train_ds = train_ds.map(prepare_train).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(prepare_val).prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds


def build_model(hp):
    base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = False

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(hp["dense_units"], activation="relu")(x)
    if hp["use_batchnorm"]:
        x = layers.BatchNormalization()(x)
    x = layers.Dropout(hp["dropout_rate"])(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = models.Model(inputs=base_model.input, outputs=outputs)
    return model, base_model


def get_optimizer(name, lr):
    return optimizers.Adam(learning_rate=lr) if name == "adam" else optimizers.RMSprop(learning_rate=lr)


def main():
    with open("/kaggle/working/best_hyperparameters_b0.json") as f:
        hp = json.load(f)
    print("Hyperparamètres utilisés :", hp)

    train_ds, val_ds = load_datasets()
    model, base_model = build_model(hp)

    model.compile(
        optimizer=get_optimizer(hp["optimizer"], hp["learning_rate"]),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc", multi_label=True)],
    )
    model.summary()

    callbacks_list = [
        callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7),
        callbacks.ModelCheckpoint("/kaggle/working/best_model_b0_phase1.keras", monitor="val_accuracy", save_best_only=True),
        callbacks.CSVLogger("/kaggle/working/training_log_b0_phase1.csv"),
    ]

    print("\n=== PHASE 1 : tête seule (base gelée) ===")
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_PHASE1, callbacks=callbacks_list)

    if hp["unfreeze_layers"] > 0:
        print(f"\n=== PHASE 2 : fine-tuning ({hp['unfreeze_layers']} dernières couches) ===")
        base_model.trainable = True
        for layer in base_model.layers[:-hp["unfreeze_layers"]]:
            layer.trainable = False

        model.compile(
            optimizer=get_optimizer(hp["optimizer"], hp["learning_rate"] / 10),
            loss="categorical_crossentropy",
            metrics=["accuracy", tf.keras.metrics.AUC(name="auc", multi_label=True)],
        )

        callbacks_phase2 = [
            callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-8),
            callbacks.ModelCheckpoint("/kaggle/working/best_model_b0_phase2.keras", monitor="val_accuracy", save_best_only=True),
            callbacks.CSVLogger("/kaggle/working/training_log_b0_phase2.csv"),
        ]
        model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_PHASE2, callbacks=callbacks_phase2)

    model.save("/kaggle/working/aigis_efficientnetb0_final.keras")
    print("\n✅ Modèle final sauvegardé")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open("/kaggle/working/aigis_efficientnetb0.tflite", "wb") as f:
        f.write(tflite_model)
    print("✅ Modèle TFLite exporté")


if __name__ == "__main__":
    main()
