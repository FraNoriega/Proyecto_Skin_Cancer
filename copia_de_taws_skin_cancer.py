import pandas as pd
import numpy as np
from glob import glob
import tensorflow as tf
from tensorflow import keras
from  tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from sklearn.preprocessing import LabelEncoder
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from PIL import Image
from tqdm import tqdm
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import plotly.graph_objects as go
import gc
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam, Adamax
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/drive')

metadata_dir = "/content/drive/MyDrive/ProyectoTaws/archive/HAM10000_metadata.csv"
data = pd.read_csv(metadata_dir)
base_skin_dir = os.path.join('..', '/content/drive/MyDrive/ProyectoTaws/archive')

# Hacemos un diccionario, donde las claves sean los nombres de las imagenes y los valores los directorios
imageid_path_dict = {
    os.path.splitext(os.path.basename(x))[0]: x
    for x in glob(os.path.join(base_skin_dir, '*', '*.jpg'))
}

data['path'] = data['image_id'].map(imageid_path_dict.get) #Hacemos una columna donde agregamos los directorios al df
data['dx_code'] = pd.Categorical(data['dx']).codes #Hacemos una columna donde agregamos las etiquetas de forma numerica al df

print(data.isnull().sum()) #Verificamos si existen elementos nulos

data['age'].fillna((data['age'].mean()), inplace=True) #Reeplazamos los elementos nulos

image_paths = list(data['path'])
print(image_paths)
data['image'] = [np.asarray(Image.open(path).resize((100, 100)), dtype=np.float32)/255.0 for path in tqdm(image_paths)] #Hacemos una columna que representa las imagenes como matrices de pixeles

img= data["image"]
label=data["dx_code"]

train_img, test_img, train_label, test_label = train_test_split(img, label, test_size=0.20, random_state=42) #Separamos en train y test

train_label = to_categorical(train_label)
test_label = to_categorical(test_label)

# Realizamos aumento de datos para evitar el sobreajuste
datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range = 0.1,
        width_shift_range=0.1,
        height_shift_range=0.1)
datagen.fit(train_img)

# Funcion para nuestra matriz de confusion
def plot_confusion_matrix(cm, classes, normalize=False):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        colorscale = 'Plasma'
    else:
        colorscale = 'Magma_r'  # Reversed 'Magma' colorscale for non-normalized matrix

    annotations = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            annotations.append(
                {
                    'x': classes[j],
                    'y': classes[i],
                    'text': str(cm[i, j]),
                    'showarrow': False,
                    'font': {'color': 'red' if cm[i, j] > 0.5 else 'black'}
                }
            )

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=list(classes),
        y=list(classes),
        colorscale=colorscale,
        colorbar=dict(title='Normalized' if normalize else 'Count'),
        showscale=True,
        hoverinfo='z'
    ))

    fig.update_layout(
        title='Confusion Matrix',
        xaxis=dict(title='Predicted label'),
        yaxis=dict(title='True label'),
        annotations=annotations
    )

    if normalize:
        fig.update_layout(title_text='Normalized Confusion Matrix')
    else:
        fig.update_layout(title_text='Confusion Matrix (Counts)')

    fig.show()

from keras.callbacks import ReduceLROnPlateau

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy'
                                            , patience = 2
                                            , verbose=1
                                            ,factor=0.5
                                            , min_lr=0.00001)

model = keras.models.Sequential()

# Create Model Structure
model.add(keras.layers.Input(shape=[100, 100, 3]))
model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(keras.layers.MaxPooling2D())
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(keras.layers.MaxPooling2D())
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(keras.layers.MaxPooling2D())
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(keras.layers.MaxPooling2D())

model.add(keras.layers.Flatten())
model.add(keras.layers.Dropout(rate=0.5))
model.add(keras.layers.Dense(units=256, activation='relu', kernel_initializer='he_normal'))
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Dense(units=128, activation='relu', kernel_initializer='he_normal'))
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Dense(units=64, activation='relu', kernel_initializer='he_normal'))
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Dense(units=32, activation='relu', kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.L1L2()))
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Dense(units=7, activation='softmax', kernel_initializer='glorot_uniform', name='classifier'))
model.compile(Adamax(learning_rate= 0.001), loss= 'categorical_crossentropy', metrics= ['accuracy'])

# model.summary()
# keras.utils.plot_model(model, show_shapes=True)

early_stopping = EarlyStopping(monitor='loss', patience=3)
board = TensorBoard(log_dir='./board')
epochs = 100
batch_size = 128
history = model.fit(datagen.flow(train_img, train_label, batch_size=batch_size),
                              epochs=epochs,
                              verbose=1,
                              callbacks=[board,early_stopping,learning_rate_reduction], validation_data=(test_img, test_label))

def plot_training(hist):
    tr_acc = hist.history['accuracy']
    tr_loss = hist.history['loss']
    val_acc = hist.history['val_accuracy']
    val_loss = hist.history['val_loss']
    index_loss = np.argmin(val_loss)
    val_lowest = val_loss[index_loss]
    index_acc = np.argmax(val_acc)
    acc_highest = val_acc[index_acc]

    plt.figure(figsize= (20, 8))
    plt.style.use('fivethirtyeight')
    Epochs = [i+1 for i in range(len(tr_acc))]
    loss_label = f'best epoch= {str(index_loss + 1)}'
    acc_label = f'best epoch= {str(index_acc + 1)}'

    plt.subplot(1, 2, 1)
    plt.plot(Epochs, tr_loss, 'r', label= 'Training loss')
    plt.plot(Epochs, val_loss, 'g', label= 'Validation loss')
    plt.scatter(index_loss + 1, val_lowest, s= 150, c= 'blue', label= loss_label)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(Epochs, tr_acc, 'r', label= 'Training Accuracy')
    plt.plot(Epochs, val_acc, 'g', label= 'Validation Accuracy')
    plt.scatter(index_acc + 1 , acc_highest, s= 150, c= 'blue', label= acc_label)
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout
    plt.show()

plot_training(history)

from sklearn.metrics import confusion_matrix, classification_report

classes = range(7)

# Y_true (true labels) and Y_pred_classes (predicted labels)
Y_pred = model.predict(test_img)
Y_pred_classes = np.argmax(Y_pred, axis=1)
Y_true = np.argmax(test_label, axis=1)

# Compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)

# Plot the confusion matrix with the new colorscale
plot_confusion_matrix(confusion_mtx, classes=classes, normalize=False)

report = classification_report(Y_true, Y_pred_classes)
print(f"Classification Report for <<DenseNet121>> : ")
print(report)

model.save('modelo.h5')

!pip install tensorflowjs

!mkdir carpeta_salida

!tensorflowjs_converter --input_format keras modelo.h5 carpeta_salida