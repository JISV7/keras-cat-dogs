"""Entrenamiento desde cero según el ejemplo oficial de Keras, con 50 épocas.

Sin pesos preentrenados: arquitectura mini-Xception, augmentación y compile
idénticos a https://keras.io/examples/vision/image_classification_from_scratch/
Se ejecuta en una GPU T4 de Google Colab.
"""

import os
import time

import numpy as np
import keras
from keras import layers
from tensorflow import data as tf_data

DATA_DIR = "/content/PetImages"
MODEL_PATH = "/content/cats_dogs_model_scratch.keras"
LOG_PATH = "/content/cats_dogs_history.csv"
INFO_PATH = "/content/cats_dogs_scratch_info.txt"
image_size = (180, 180)
batch_size = 128
epochs = 50


def log(*args):
    print(*args, flush=True)


train_ds, val_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="both",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)
log("Clases:", train_ds.class_names)

data_augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
]


def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images


train_ds = train_ds.map(
    lambda img, label: (data_augmentation(img), label),
    num_parallel_calls=tf_data.AUTOTUNE,
)
train_ds = train_ds.prefetch(tf_data.AUTOTUNE)
val_ds = val_ds.prefetch(tf_data.AUTOTUNE)


def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    units = 1 if num_classes == 2 else num_classes

    x = layers.Dropout(0.25)(x)
    # activation=None para devolver logits
    outputs = layers.Dense(units, activation=None)(x)
    return keras.Model(inputs, outputs)


model = make_model(input_shape=image_size + (3,), num_classes=2)
model.summary(print_fn=lambda line: log(line))

callbacks = [
    keras.callbacks.ModelCheckpoint(
        MODEL_PATH, monitor="val_acc", mode="max", save_best_only=True
    ),
    keras.callbacks.CSVLogger(LOG_PATH),
]
model.compile(
    optimizer=keras.optimizers.Adam(3e-4),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy(name="acc")],
)

log(f"Iniciando entrenamiento desde cero: {epochs} épocas en GPU T4")
start = time.time()
history = model.fit(
    train_ds,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=val_ds,
    verbose=1,
)
log(f"Tiempo total: {time.time() - start:.1f} s")

best_val_acc = max(history.history["val_acc"])
log(f"Mejor accuracy en validación: {best_val_acc:.4f}")

clf_model = keras.models.load_model(MODEL_PATH)


def classify(image_path):
    """Devuelve (etiqueta, confianza%, p_gato%, p_perro%)."""
    img = keras.utils.load_img(image_path, target_size=image_size)
    img_array = keras.utils.img_to_array(img)
    img_array = keras.ops.expand_dims(img_array, 0)
    logit = clf_model.predict(img_array, verbose=0)[0][0]
    p_perro = float(keras.ops.sigmoid(logit))
    p_gato = 1.0 - p_perro
    if p_perro >= 0.5:
        return "perro", p_perro * 100, p_gato * 100, p_perro * 100
    return "gato", p_gato * 100, p_gato * 100, p_perro * 100


with open(INFO_PATH, "w", encoding="utf-8") as fh:
    fh.write("Modelo entrenado desde cero (sin pesos preentrenados)\n")
    fh.write("Arquitectura: mini-Xception del ejemplo oficial de Keras\n")
    fh.write(f"Epocas: {epochs}\n")
    fh.write(f"Mejor accuracy validacion: {best_val_acc:.4f}\n")
    fh.write("0 = gato (Cat)\n1 = perro (Dog)\n")

for folder, esperado in (("Cat", "gato"), ("Dog", "perro")):
    files = sorted(f for f in os.listdir(os.path.join(DATA_DIR, folder)))
    sample = os.path.join(DATA_DIR, folder, files[0])
    etiqueta, conf, pg, pp = classify(sample)
    log(
        f"Prueba ({esperado}): {etiqueta} - confianza {conf:.2f}% "
        f"[gato {pg:.2f}% / perro {pp:.2f}%]"
    )

log("LISTO")
