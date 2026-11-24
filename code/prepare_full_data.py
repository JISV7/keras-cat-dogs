"""Descarga y limpia el dataset completo de Kaggle Cats vs Dogs en Colab.

Sin subconjuntos: reproduce exactamente el ejemplo oficial de Keras.
"""

import os
import urllib.request
import zipfile

DATA_URL = (
    "https://download.microsoft.com/download/3/E/1/"
    "3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"
)
ZIP_PATH = "/content/kagglecatsanddogs_5340.zip"
DATA_DIR = "/content/PetImages"


def log(*args):
    print(*args, flush=True)


if not os.path.exists(ZIP_PATH):
    progress = [10]

    def report(block_num, block_size, total_size):
        pct = min(100.0, block_num * block_size * 100.0 / total_size)
        if pct >= progress[0]:
            log(f"Descargando... {pct:.0f}%")
            progress[0] += 10

    log("Descargando ZIP completo (786 MB)...")
    urllib.request.urlretrieve(DATA_URL, ZIP_PATH, report)

if not os.path.exists(DATA_DIR):
    log("Extrayendo...")
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall("/content")

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

for folder_name in ("Cat", "Dog"):
    folder_path = os.path.join(DATA_DIR, folder_name)
    log(f"{folder_name}: {len(os.listdir(folder_path))} imágenes válidas")
log(f"Eliminadas por corruptas: {num_skipped}")
