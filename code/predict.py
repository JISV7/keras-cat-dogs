"""Clasifica imágenes como perro o gato mostrando el porcentaje de confianza.

Uso:
    python predict.py ruta/a/imagen.jpg [otra_imagen.jpg ...]
    python predict.py ruta/a/imagen.jpg --save-plot probabilidades.png

Desde Colab:
    from predict import classify
    etiqueta, confianza, p_gato, p_perro = classify("/content/mi_imagen.jpg")
"""

import argparse
import os

import keras
import matplotlib.pyplot as plt

IMAGE_SIZE = (180, 180)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cats_dogs_model.keras")


def find_model_path():
    """Busca el modelo en la carpeta del script o en ubicaciones típicas de Colab."""
    candidates = [
        MODEL_PATH,
        os.path.join(os.getcwd(), "cats_dogs_model.keras"),
        os.path.join(os.getcwd(), "code", "cats_dogs_model.keras"),
        "/content/cats_dogs_model.keras",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "No se encontró cats_dogs_model.keras. Colócalo junto a predict.py "
        "o en /content/cats_dogs_model.keras."
    )


_model = None


def get_model():
    global _model
    if _model is None:
        _model = keras.models.load_model(find_model_path())
    return _model


def classify(image_path):
    """Devuelve (etiqueta, confianza_en_porcentaje, p_gato, p_perro)."""
    img = keras.utils.load_img(image_path, target_size=IMAGE_SIZE)
    img_array = keras.utils.img_to_array(img)
    img_array = keras.ops.expand_dims(img_array, 0)  # eje de batch
    prediction = get_model().predict(img_array, verbose=0)
    p_perro = float(keras.ops.sigmoid(prediction[0][0]))
    p_gato = 1.0 - p_perro
    if p_perro >= 0.5:
        return "perro", p_perro * 100.0, p_gato, p_perro
    return "gato", p_gato * 100.0, p_gato, p_perro


def plot_probabilities(p_gato, p_perro, save_path=None):
    """Dibuja con Matplotlib las probabilidades de gato y perro."""
    etiquetas = ["Gato", "Perro"]
    valores = [p_gato * 100.0, p_perro * 100.0]
    colores = ["#4C78A8", "#F58518"]

    fig, ax = plt.subplots(figsize=(6, 4))
    barras = ax.bar(etiquetas, valores, color=colores)
    ax.set_ylabel("Probabilidad (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Clasificación perro / gato")
    ax.grid(axis="y", alpha=0.3)
    for barra, valor in zip(barras, valores):
        ax.text(
            barra.get_x() + barra.get_width() / 2,
            valor + 1.5,
            f"{valor:.2f}%",
            ha="center",
        )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Gráfico guardado: {save_path}")
    plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Clasificador perro/gato con Keras")
    parser.add_argument("imagenes", nargs="+", help="rutas de las imágenes a clasificar")
    parser.add_argument(
        "--save-plot",
        metavar="PNG",
        help="guarda el gráfico de probabilidades en un archivo PNG",
    )
    args = parser.parse_args()

    for indice, path in enumerate(args.imagenes, start=1):
        if not os.path.exists(path):
            print(f"{path}: archivo no encontrado")
            continue
        etiqueta, confianza, p_gato, p_perro = classify(path)
        print(
            f"{path} -> {etiqueta} (confianza: {confianza:.2f}%) "
            f"| gato: {p_gato * 100:.2f}% | perro: {p_perro * 100:.2f}%"
        )

        destino = None
        if args.save_plot:
            base, ext = os.path.splitext(args.save_plot)
            destino = (
                args.save_plot
                if len(args.imagenes) == 1
                else f"{base}_{indice}{ext or '.png'}"
            )
        plot_probabilities(p_gato, p_perro, destino)


if __name__ == "__main__":
    main()
