import numpy as np
import struct
import tensorflow as tf
from tensorflow.keras import layers, models

# IDX formatındaki dosyaları okuyan fonksiyonlar
def load_images(filename):
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28, 28)
    return images

def load_labels(filename):
    with open(filename, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

# Veriyi yükleme
x_train = load_images('train-images.idx3-ubyte') / 255.0
y_train = load_labels('train-labels.idx1-ubyte')

x_test = load_images('t10k-images.idx3-ubyte') / 255.0
y_test = load_labels('t10k-labels.idx1-ubyte')

# CNN için giriş boyutunu ayarla (28, 28, 1)
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

# CNN modeli tanımı
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Modeli derle
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Modeli eğit
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Eğitilen modeli kaydet
model.save('mnist_cnn_model.h5')
