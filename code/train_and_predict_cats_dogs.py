"""Entrena el clasificador perro/gato en la sesión CPU y define classify().

Modelo tomado de https://keras.io/examples/vision/image_classification_from_scratch/
Adaptado: 5000 imágenes, batch 32, 5 épocas (pensado para CPU).
"""

import os
import time

import keras
from keras import layers
from tensorflow import data as tf_data

MODEL_PATH = "/content/cats_dogs_model.keras"
SUBSET_DIR = "/content/cats_dogs_subset"
image_size = (180, 180)


def log(*args):
    print(*args, flush=True)


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
log("Arquitectura:")
model.summary(print_fn=lambda line: log(line))

callbacks = [
    keras.callbacks.ModelCheckpoint(
        MODEL_PATH, monitor="val_acc", mode="max", save_best_only=True
    )
]
model.compile(
    optimizer=keras.optimizers.Adam(3e-4),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy(name="acc")],
)

log("Iniciando entrenamiento (5 epocas, aprox. varios minutos en CPU)...")
start = time.time()
model.fit(
    train_ds,
    epochs=5,
    callbacks=callbacks,
    validation_data=val_ds,
    verbose=1,
)
log(f"Entrenamiento terminado en {time.time() - start:.1f} s")

# Cargamos el mejor checkpoint para las predicciones.
clf_model = keras.models.load_model(MODEL_PATH)
log(f"Modelo guardado en {MODEL_PATH}")

# Orden de clases de image_dataset_from_directory: 0 = Cat, 1 = Dog.
CLASS_NAMES = {0: "gato", 1: "perro"}


def classify(image_path):
    """Devuelve la clase y probabilidades para una imagen en /content."""
    img = keras.utils.load_img(image_path, target_size=image_size)
    img_array = keras.utils.img_to_array(img)
    img_array = keras.ops.expand_dims(img_array, 0)  # eje de batch
    predictions = clf_model.predict(img_array, verbose=0)
    score = float(keras.ops.sigmoid(predictions[0][0]))  # P(perro)
    return {
        "imagen": image_path,
        "clase": "perro" if score >= 0.5 else "gato",
        "probabilidad_gato": round(1 - score, 4),
        "probabilidad_perro": round(score, 4),
    }


with open("/content/cats_dogs_class_names.txt", "w", encoding="utf-8") as fh:
    fh.write("0 = gato (Cat)\n1 = perro (Dog)\n")

# Prueba rápida con una imagen de cada clase del subconjunto.
for folder, label in (("Cat", "gato"), ("Dog", "perro")):
    files = sorted(f for f in os.listdir(os.path.join(SUBSET_DIR, folder)))
    sample = os.path.join(SUBSET_DIR, folder, files[0])
    result = classify(sample)
    log(
        f"Prueba ({label}): {result['clase']} "
        f"[{result['probabilidad_gato']:.3f} gato / "
        f"{result['probabilidad_perro']:.3f} perro]"
    )

log("Listo. Usa classify('/content/tu_imagen.jpg') para clasificar una imagen.")
