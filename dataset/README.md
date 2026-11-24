# Dataset: Cats vs Dogs

El repositorio **no incluye las imágenes ni el ZIP**: GitHub rechaza archivos de más de 100 MB
y el dataset completo ocupa 786 MB. Descárgalo desde la fuente oficial y descomprímelo en Colab.

- Origen oficial: [Kaggle Cats and Dogs Dataset (Microsoft)](https://www.microsoft.com/en-us/download/details.aspx?id=54765)
- Archivo: `kagglecatsanddogs_5340.zip` (786 MB)
- Contenido tras la limpieza: **23 198 imágenes válidas** (11 626 gatos y 11 572 perros).
- Se eliminan las imágenes corruptas (cabecera sin `JFIF`), como en el ejemplo de Keras.
- Orden de clases de `keras.utils.image_dataset_from_directory`: `0 = Cat (gato)`, `1 = Dog (perro)`.

## Preparación en Google Colab

Sube el ZIP a `/content/` o descárgalo desde el enlace y ejecuta:

```python
exec(open('/content/code/prepare_full_data.py').read())
```

El script descarga/extrae el ZIP a `/content/PetImages`, elimina las imágenes corruptas
y deja los directorios `PetImages/Cat` y `PetImages/Dog` listos para el entrenamiento.

Para entrenar el modelo final (50 épocas, desde cero):

```python
exec(open('/content/code/train_full_from_scratch.py').read())
```

Para clasificar una imagen con porcentaje de confianza y gráfico de probabilidades:

```python
from predict import classify, plot_probabilities
etiqueta, confianza, p_gato, p_perro = classify('/content/mi_imagen.jpg')
print(f'{etiqueta}: {confianza:.2f}%')
plot_probabilities(p_gato, p_perro)
```
