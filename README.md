https://github.com/JISV7/keras-cat-dogs

30.958.324 José Sánchez
32.590.258 Verónica Blázquez

# Clasificador perro / gato con Keras (Google Colab)

Modelo entrenado **desde cero** en una máquina virtual de **Google Colab (GPU T4)**
que clasifica una imagen como **perro** o **gato**, indica el **porcentaje de confianza**
y dibuja las probabilidades con **Matplotlib**.

## Estructura

```
keras-cat-dogs/
├── .gitignore                             # ignora ZIPs y temporales locales
├── code/                                  # código, modelo y notebook
│   ├── image_classification_from_scratch.py      # ejemplo original de GitHub
│   ├── image_classification_from_scratch.ipynb   # ejemplo original de Colab
│   ├── prepare_cats_dogs_data.py                # descarga del dataset (subconjunto)
│   ├── prepare_full_data.py                     # descarga del dataset completo
│   ├── train_and_predict_cats_dogs.py           # entrenamiento desde cero (referencia)
│   ├── train_full_from_scratch.py               # entrenamiento final usado en Colab
│   ├── predecir_perro_gato.ipynb                # notebook de inferencia + gráfico
│   ├── predict.py                               # CLI: imagen -> clase, confianza y gráfico
│   ├── cats_dogs_model.keras                    # modelo final entrenado desde cero
│   ├── cats_dogs_history.csv                    # métricas de las 50 épocas
│   └── requirements.txt
└── dataset/                               # solo enlace de descarga, sin imágenes
    └── README.md
```

El dataset no se versiona porque el ZIP oficial pesa 786 MB (límite de GitHub: 100 MB por
archivo). Las instrucciones de descarga y extracción están en `dataset/README.md`.

## Entrenamiento

Basado en el ejemplo oficial de Keras
[Image classification from scratch](https://keras.io/examples/vision/image_classification_from_scratch/)
([GitHub](https://github.com/keras-team/keras-io/blob/master/examples/vision/image_classification_from_scratch.py),
[Colab](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/image_classification_from_scratch.ipynb)).

**No se usaron pesos preentrenados.** Se entrenó la arquitectura mini-Xception del
ejemplo directamente sobre el dataset completo de Kaggle Cats vs Dogs:

- 23 198 imágenes válidas (11 626 gatos y 11 572 perros); se descartaron 1 804 corruptas.
- 180×180 píxeles, lote 128, `validation_split=0.2`, semilla 1337.
- Augmentación: `RandomFlip("horizontal")` y `RandomRotation(0.1)`.
- Optimizador Adam(3e-4), `BinaryCrossentropy(from_logits=True)`, 50 épocas en T4.
- `ModelCheckpoint(save_best_only=True, monitor="val_acc")` y `CSVLogger`.
- **Mejor accuracy en validación: 95.99%** (época 39 contando desde 0).
- Pruebas con imágenes del dataset: gato 87.48%, perro 95.93% de confianza.

El historial completo está en `code/cats_dogs_history.csv`.

Para reproducirlo en Colab:

```python
exec(open('/content/code/prepare_full_data.py').read())
exec(open('/content/code/train_full_from_scratch.py').read())
```

## Clasificar una imagen con porcentaje de confianza

```bash
python code/predict.py /content/mi_imagen.jpg
python code/predict.py /content/mi_imagen.jpg --save-plot probabilidades.png
```

Ejemplo de salida:

```
/content/mi_imagen.jpg -> perro (confianza: 95.93%) | gato: 4.07% | perro: 95.93%
```

Además de la salida por terminal se dibuja una barra con `Gato %` y `Perro %` usando
Matplotlib. En Colab abre `code/predecir_perro_gato.ipynb`: sube la imagen, ejecuta las
celdas y obtendrás la clase, la confianza y el gráfico.

El modelo devuelve un logit; `sigmoid(logit)` es la probabilidad de **perro**.
Etiquetas: `0 = gato (Cat)`, `1 = perro (Dog)`.
