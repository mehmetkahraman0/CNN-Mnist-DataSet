import numpy as np
import tensorflow as tf
import struct
import matplotlib.pyplot as plt

# IDX dosyalarını okuyacak yardımcı fonksiyonlar
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

# Test verisini yükle
x_test = load_images('t10k-images.idx3-ubyte') / 255.0
y_test = load_labels('t10k-labels.idx1-ubyte')
x_test = x_test[..., np.newaxis]

# Eğitilmiş modeli yükle
model = tf.keras.models.load_model('mnist_cnn_model.h5')

# İlk 5 test örneğini tahmin et
predictions = model.predict(x_test[:5])

# Sonuçları yazdır ve görselleştir
for i in range(5):
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"Gerçek: {y_test[i]} - Tahmin: {np.argmax(predictions[i])}")
    plt.axis('off')
    plt.show()
