"""
models/efficientnetb2/tune.py

Recherche d'hyperparamètres (Keras Tuner, optimisation bayésienne) pour
EfficientNetB2 sur le dataset Aigis.
"""

import os
import json
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB2
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import layers, models, optimizers
import keras_tuner as kt

DATASET_PATH = "/kaggle/working/dataset_clean_b2"
IMG_SIZE = 260
BATCH_SIZE = 32
NUM_CLASSES = 4
SEARCH_EPOCHS = 5
MAX_TRIALS = 10


def build_model(hp):
    dense_units = hp.Choice("dense_units", values=[256, 512, 1024])
    dropout_rate = hp.Float("dropout_rate", min_value=0.3, max_value=0.7, step=0.1)
    learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4, 1e-5])
    unfreeze_layers = hp.Choice("unfreeze_layers", values=[0, 20, 50])
    use_batchnorm = hp.Boolean("use_batchnorm")
    optimizer_choice = hp.Choice("optimizer", values=["adam", "rmsprop"])

    base_model = EfficientNetB2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = unfreeze_layers > 0
    if unfreeze_layers > 0:
        for layer in base_model.layers[:-unfreeze_layers]:
            layer.trainable = False

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    if use_batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = models.Model(inputs=base_model.input, outputs=outputs)
    opt = optimizers.Adam(learning_rate=learning_rate) if optimizer_choice == "adam" else optimizers.RMSprop(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss="categorical_crossentropy",
                  metrics=["accuracy", tf.keras.metrics.AUC(name="auc", multi_label=True)])
    return model


def main():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATASET_PATH, "train"), image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, label_mode="categorical", shuffle=True, seed=42,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATASET_PATH, "val"), image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, label_mode="categorical", shuffle=False,
    )

    def preprocess(images, labels):
        return preprocess_input(images), labels

    train_ds = train_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)
    train_ds_search = train_ds.take(int(len(train_ds) * 0.4))

    tuner = kt.BayesianOptimization(
        build_model, objective=kt.Objective("val_accuracy", direction="max"),
        max_trials=MAX_TRIALS, directory="/kaggle/working/tuner_results_b2",
        project_name="aigis_b2_search", overwrite=True,
    )

    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
    tuner.search(train_ds_search, validation_data=val_ds, epochs=SEARCH_EPOCHS, callbacks=[early_stop], verbose=1)

    best_hps = tuner.get_best_hyperparameters(num_trials=3)
    print("\nTOP 3 - EfficientNetB2")
    for i, hp in enumerate(best_hps):
        print(f"Rang {i+1}: {hp.values}")

    with open("/kaggle/working/best_hyperparameters_b2.json", "w") as f:
        json.dump(best_hps[0].values, f, indent=2)
    print("✅ Sauvegardé : /kaggle/working/best_hyperparameters_b2.json")


if __name__ == "__main__":
    main()
