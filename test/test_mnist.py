import tensorflow as tf
import numpy as np
import csv
import os
import random

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(train_images, _), (_, _) = mnist.load_data()

# Define the number of random images to create
num_images = 25

for i in range(num_images):
    # Select a random image
    random_index = random.randint(0, len(train_images) - 1)
    random_image = train_images[random_index]

    # Normalize the image values to 0-1 range if needed
    random_image_normalized = random_image / 255.0

    # Decide which version of the image to write (0-255 range or 0-1 range)
    # Use `random_image` for 0-255 range, and `random_image_normalized` for 0-1 range
    image_to_write = random.choice([random_image, random_image_normalized])

    # Save the pixel values to a CSV file
    csv_file_path = f'Data/random_image_{i + 1}.csv'
    with open(csv_file_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, lineterminator='\n')
        # Write each row of the image to the CSV file
        for row in image_to_write:
            csvwriter.writerow(row)

    print(f"CSV file saved at: {csv_file_path}")
