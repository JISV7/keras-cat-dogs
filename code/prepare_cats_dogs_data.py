"""Prepara un subconjunto pequeño de Kaggle Cats vs Dogs en la sesión de Colab.

Basado en https://keras.io/examples/vision/image_classification_from_scratch/
Adaptado para entrenar en CPU: se usa un subconjunto de 2500 imágenes por clase.
"""

import os
import random
import shutil
import urllib.request
import zipfile

import keras

DATA_URL = (
    "https://download.microsoft.com/download/3/E/1/"
    "3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"
)
ZIP_PATH = "/content/kagglecatsanddogs_5340.zip"
DATA_DIR = "/content/PetImages"
SUBSET_DIR = "/content/cats_dogs_subset"
TRAIN_PER_CLASS = 2500


def log(*args):
    print(*args, flush=True)


if not os.path.exists(ZIP_PATH):
    progress = [10]

    def report(block_num, block_size, total_size):
        pct = min(100.0, block_num * block_size * 100.0 / total_size)
        if pct >= progress[0]:
            log(f"Descargando datos... {pct:.0f}%")
            progress[0] += 10

    log("Descargando el ZIP de Kaggle Cats vs Dogs (786 MB)...")
    urllib.request.urlretrieve(DATA_URL, ZIP_PATH, report)
else:
    log("ZIP ya descargado, se reutiliza.")

if not os.path.exists(DATA_DIR):
    log("Extrayendo imágenes...")
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall("/content")
else:
    log("PetImages ya existe, se reutiliza.")

# Filtrar imágenes corruptas (mismo criterio que el ejemplo de Keras).
num_skipped = 0
for folder_name in ("Cat", "Dog"):
    folder_path = os.path.join(DATA_DIR, folder_name)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        try:
            with open(fpath, "rb") as fobj:
                is_jfif = b"JFIF" in fobj.read(10)
        except OSError:
            is_jfif = False
        if not is_jfif:
            num_skipped += 1
            os.remove(fpath)
log(f"Imágenes corruptas eliminadas: {num_skipped}")

# Crear el subconjunto para entrenar rápido en CPU.
shutil.rmtree(SUBSET_DIR, ignore_errors=True)
rng = random.Random(1337)
for folder_name in ("Cat", "Dog"):
    src = os.path.join(DATA_DIR, folder_name)
    dst = os.path.join(SUBSET_DIR, folder_name)
    os.makedirs(dst, exist_ok=True)
    files = sorted(
        f for f in os.listdir(src) if f.lower().endswith((".jpg", ".jpeg"))
    )
    rng.shuffle(files)
    chosen = files[:TRAIN_PER_CLASS]
    for fname in chosen:
        shutil.copy2(os.path.join(src, fname), os.path.join(dst, fname))
    log(f"{folder_name}: {len(chosen)} imágenes en el subconjunto")

image_size = (180, 180)
batch_size = 32

train_ds, val_ds = keras.utils.image_dataset_from_directory(
    SUBSET_DIR,
    validation_split=0.2,
    subset="both",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)
log("Clases (orden de etiquetas):", train_ds.class_names)
log("Subconjunto listo en", SUBSET_DIR)
