import tensorflow as tf
import pandas as pd
import numpy as np
import base64
import json
import io

# Read the CSV file as bytes and encode it with base64
csv_file_path = 'Data/test_image.csv'
df = pd.read_csv(csv_file_path, header=None)
print(f'df:\n{df}')

local_model_path = 'Models/transformer_model.h5'
model = tf.keras.models.load_model(local_model_path)

# Preprocess the image data
image_data = df.values.reshape(1, 28, 28)
image_data = image_data / 255.0

# Use the model to classify the image
prediction = model.predict(image_data)
predicted_class = int(np.argmax(prediction))
print(f'predicted_class: {predicted_class}')