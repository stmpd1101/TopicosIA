# ============================================================
# CLASIFICADOR DE IMÁGENES CON CIFAR-100
# 3 clases seleccionadas:
# - apple
# - orange
# - pear
#
# El programa:
# 1. Carga CIFAR-100
# 2. Filtra solo 3 clases
# 3. Entrena una red neuronal convolucional
# 4. Evalúa el modelo
# 5. Genera matriz de confusión
# 6. Muestra imágenes con etiqueta real y predicha
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


# ============================================================
# 1. CARGAR DATASET CIFAR-100
# ============================================================

# CIFAR-100 tiene 100 clases.
# label_mode='fine' significa que usaremos las etiquetas específicas.
(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

print("Forma de x_train:", x_train.shape)
print("Forma de y_train:", y_train.shape)
print("Forma de x_test:", x_test.shape)
print("Forma de y_test:", y_test.shape)


# ============================================================
# 2. DEFINIR NOMBRES DE LAS CLASES DE CIFAR-100
# ============================================================

# Lista oficial de las 100 clases de CIFAR-100 en orden.
fine_labels = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver',
    'bed', 'bee', 'beetle', 'bicycle', 'bottle',
    'bowl', 'boy', 'bridge', 'bus', 'butterfly',
    'camel', 'can', 'castle', 'caterpillar', 'cattle',
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach',
    'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox',
    'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
    'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard',
    'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
    'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid',
    'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
    'plain', 'plate', 'poppy', 'porcupine', 'possum',
    'rabbit', 'raccoon', 'ray', 'road', 'rocket',
    'rose', 'sea', 'seal', 'shark', 'shrew',
    'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor',
    'train', 'trout', 'tulip', 'turtle', 'wardrobe',
    'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]


# ============================================================
# 3. SELECCIONAR 3 CLASES
# ============================================================

# Puedes cambiar estas clases si quieres.
clases_seleccionadas = ['apple', 'orange', 'pear']

# Obtener el número original de cada clase dentro de CIFAR-100.
indices_clases = [fine_labels.index(clase) for clase in clases_seleccionadas]

print("Clases seleccionadas:", clases_seleccionadas)
print("Índices originales:", indices_clases)


# ============================================================
# 4. FILTRAR EL DATASET PARA QUEDARNOS SOLO CON ESAS 3 CLASES
# ============================================================

def filtrar_clases(x, y, indices_clases):
    """
    Esta función recibe las imágenes y etiquetas originales,
    y devuelve únicamente las imágenes que pertenecen a las clases seleccionadas.
    """

    # Convertimos y de matriz a vector simple.
    y = y.flatten()

    # Creamos una máscara booleana.
    # True para las imágenes cuya etiqueta esté en indices_clases.
    mascara = np.isin(y, indices_clases)

    # Filtramos imágenes y etiquetas.
    x_filtrado = x[mascara]
    y_filtrado = y[mascara]

    # Convertimos las etiquetas originales a etiquetas nuevas:
    # apple  -> 0
    # orange -> 1
    # pear   -> 2
    nuevo_y = np.zeros_like(y_filtrado)

    for nueva_etiqueta, etiqueta_original in enumerate(indices_clases):
        nuevo_y[y_filtrado == etiqueta_original] = nueva_etiqueta

    return x_filtrado, nuevo_y


x_train_filtrado, y_train_filtrado = filtrar_clases(x_train, y_train, indices_clases)
x_test_filtrado, y_test_filtrado = filtrar_clases(x_test, y_test, indices_clases)

print("Imágenes de entrenamiento filtradas:", x_train_filtrado.shape)
print("Etiquetas de entrenamiento filtradas:", y_train_filtrado.shape)
print("Imágenes de prueba filtradas:", x_test_filtrado.shape)
print("Etiquetas de prueba filtradas:", y_test_filtrado.shape)


# ============================================================
# 5. NORMALIZAR IMÁGENES
# ============================================================

# Las imágenes originalmente tienen valores de 0 a 255.
# Se dividen entre 255 para dejarlas entre 0 y 1.
x_train_filtrado = x_train_filtrado.astype("float32") / 255.0
x_test_filtrado = x_test_filtrado.astype("float32") / 255.0


# ============================================================
# 6. CONVERTIR ETIQUETAS A FORMATO CATEGÓRICO
# ============================================================

# Como tenemos 3 clases, las etiquetas se convierten así:
# 0 -> [1, 0, 0]
# 1 -> [0, 1, 0]
# 2 -> [0, 0, 1]

y_train_cat = to_categorical(y_train_filtrado, num_classes=3)
y_test_cat = to_categorical(y_test_filtrado, num_classes=3)


# ============================================================
# 7. CREAR MODELO CNN
# ============================================================

modelo = Sequential()

# Primera capa convolucional.
# Detecta patrones básicos como bordes, líneas y texturas.
modelo.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
modelo.add(MaxPooling2D((2, 2)))

# Segunda capa convolucional.
# Detecta patrones un poco más complejos.
modelo.add(Conv2D(64, (3, 3), activation='relu'))
modelo.add(MaxPooling2D((2, 2)))

# Tercera capa convolucional.
# Detecta características más abstractas.
modelo.add(Conv2D(128, (3, 3), activation='relu'))

# Aplanamos la información para pasarla a capas densas.
modelo.add(Flatten())

# Capa densa para clasificación.
modelo.add(Dense(128, activation='relu'))

# Dropout ayuda a reducir sobreajuste.
modelo.add(Dropout(0.5))

# Capa final.
# Tiene 3 neuronas porque hay 3 clases.
# softmax devuelve probabilidades.
modelo.add(Dense(3, activation='softmax'))


# ============================================================
# 8. COMPILAR MODELO
# ============================================================

modelo.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

modelo.summary()


# ============================================================
# 9. ENTRENAR MODELO
# ============================================================

historial = modelo.fit(
    x_train_filtrado,
    y_train_cat,
    epochs=15,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)


# ============================================================
# 10. EVALUAR MODELO
# ============================================================

loss, accuracy = modelo.evaluate(x_test_filtrado, y_test_cat, verbose=0)

print("\nPérdida en prueba:", loss)
print("Precisión en prueba:", accuracy)


# ============================================================
# 11. GRAFICAR PRECISIÓN Y PÉRDIDA
# ============================================================

plt.figure(figsize=(8, 5))
plt.plot(historial.history['accuracy'], label='Precisión entrenamiento')
plt.plot(historial.history['val_accuracy'], label='Precisión validación')
plt.title('Precisión durante el entrenamiento')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(8, 5))
plt.plot(historial.history['loss'], label='Pérdida entrenamiento')
plt.plot(historial.history['val_loss'], label='Pérdida validación')
plt.title('Pérdida durante el entrenamiento')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.grid(True)
plt.show()


# ============================================================
# 12. REALIZAR PREDICCIONES
# ============================================================

predicciones = modelo.predict(x_test_filtrado)

# Convertimos las probabilidades en clases.
# Por ejemplo:
# [0.80, 0.15, 0.05] -> clase 0
y_pred = np.argmax(predicciones, axis=1)

# Etiquetas reales.
y_real = y_test_filtrado


# ============================================================
# 13. MATRIZ DE CONFUSIÓN
# ============================================================

matriz = confusion_matrix(y_real, y_pred)

plt.figure(figsize=(7, 5))
sns.heatmap(
    matriz,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=clases_seleccionadas,
    yticklabels=clases_seleccionadas
)

plt.title('Matriz de confusión')
plt.xlabel('Etiqueta predicha')
plt.ylabel('Etiqueta real')
plt.show()


# ============================================================
# 14. REPORTE DE CLASIFICACIÓN
# ============================================================

print("\nReporte de clasificación:")
print(classification_report(
    y_real,
    y_pred,
    target_names=clases_seleccionadas
))


# ============================================================
# 15. MOSTRAR IMÁGENES CON REAL VS PREDICHO
# ============================================================

def mostrar_predicciones(x, y_real, y_pred, clases, cantidad=12):
    """
    Muestra imágenes del conjunto de prueba con:
    - etiqueta real
    - etiqueta predicha
    """

    plt.figure(figsize=(12, 8))

    for i in range(cantidad):
        plt.subplot(3, 4, i + 1)
        plt.imshow(x[i])

        real = clases[y_real[i]]
        predicho = clases[y_pred[i]]

        if real == predicho:
            titulo = f"Real: {real}\nPred: {predicho}\nCorrecto"
        else:
            titulo = f"Real: {real}\nPred: {predicho}\nIncorrecto"

        plt.title(titulo, fontsize=9)
        plt.axis('off')

    plt.tight_layout()
    plt.show()


mostrar_predicciones(
    x_test_filtrado,
    y_real,
    y_pred,
    clases_seleccionadas,
    cantidad=12
)


# ============================================================
# 16. MOSTRAR SOLO PREDICCIONES INCORRECTAS
# ============================================================

def mostrar_errores(x, y_real, y_pred, clases, cantidad=9):
    """
    Muestra únicamente imágenes donde el modelo se equivocó.
    Esto sirve para analizar los errores del clasificador.
    """

    indices_errores = np.where(y_real != y_pred)[0]

    if len(indices_errores) == 0:
        print("No hubo errores de clasificación.")
        return

    plt.figure(figsize=(10, 8))

    for i, indice in enumerate(indices_errores[:cantidad]):
        plt.subplot(3, 3, i + 1)
        plt.imshow(x[indice])

        real = clases[y_real[indice]]
        predicho = clases[y_pred[indice]]

        plt.title(f"Real: {real}\nPred: {predicho}", fontsize=9)
        plt.axis('off')

    plt.tight_layout()
    plt.show()


mostrar_errores(
    x_test_filtrado,
    y_real,
    y_pred,
    clases_seleccionadas,
    cantidad=9
)