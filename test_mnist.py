import tensorflow as tf
import numpy as np
import csv
import random

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(train_images, _), (_, _) = mnist.load_data()

# Select a random image
random_index = random.randint(0, len(train_images) - 1)
random_image = train_images[random_index]

# Flatten the 28x28 image into a 1D array
flat_image = np.reshape(random_image, (28 * 28))

# Save the pixel values to a CSV file
csv_file_path = 'Data/random_mnist_image.csv'
with open(csv_file_path, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(flat_image)

print(f"CSV file saved at: {csv_file_path}")
