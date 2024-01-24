import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# label mapping
labels = '''airplane automobile bird cat deer dog frog horse ship truck'''.split()

# Load the saved model
model = tf.keras.models.load_model('Models/model.h5')

# Load in the image
image_path = 'Images/0000.jpg'
image = Image.open(image_path)

# Reduce pixel values
n = np.array(image) / 255.0

# reshape it
p = n.reshape(1, 32, 32, 3)

# load the original label
original_label = 'airplane'
print(f"\nOriginal label is {original_label}")

predicted_label = labels[model.predict(p).argmax()]
print(f"\nPredicted label is {predicted_label}")