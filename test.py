import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# label mapping
labels = '''airplane automobile bird cat deer dog frog horse ship truck'''.split()

# Load the saved model
model = tf.keras.models.load_model('Models/model.h5')

# Load in the data
cifar10 = tf.keras.datasets.cifar10
# Distribute it to test set
(x_test, y_test), _ = cifar10.load_data()

# Reduce pixel values
x_test = x_test / 255.0

# select the image from our test dataset
image_number = 0

# display the image
plt.imshow(x_test[image_number])

# load the image in an array
n = np.array(x_test[image_number])

# reshape it
p = n.reshape(1, 32, 32, 3)

# pass in the network for prediction and 
# save the predicted label
predicted_label = labels[model.predict(p).argmax()]

# load the original label
original_label = labels[int(y_test[image_number])]

# display the result
print(f"\nOriginal label is {original_label} and predicted label is {predicted_label}")
