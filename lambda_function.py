import tensorflow as tf
from PIL import Image
import pandas as pd
import numpy as np
import logging
import base64
import keras
import boto3
import json
import csv
import ast
import io

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Load the saved model
s3 = boto3.client('s3')
bucket_name = 'opti-tf-test-lambda'
cnn_model_key = 'Models/cnn_model.h5'
transformer_model_key = 'Models/transformer_model.h5'
cnn_model_path = '/tmp/cnn_model.h5'
transformer_model_path = '/tmp/transformer_model.h5'

# Download the model file from S3 to a local temporary file
s3.download_file(bucket_name, cnn_model_key, cnn_model_path)
s3.download_file(bucket_name, transformer_model_key, transformer_model_path)

class ClassToken(tf.keras.layers.Layer):
    def _init_(self):
        super()._init_()

    def build(self, input_shape):
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value = w_init(shape=(1, 1, input_shape[-1]), dtype=tf.float32),
            trainable = True
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        hidden_dim = self.w.shape[-1]

        cls = tf.broadcast_to(self.w, [batch_size, 1, hidden_dim])
        cls = tf.cast(cls, dtype=inputs.dtype)
        return cls

# Load the model from the local file
cnn_model = tf.keras.models.load_model(cnn_model_path)
with keras.utils.custom_object_scope({'ClassToken': ClassToken}):
    transformer_model = keras.models.load_model(transformer_model_path)

def predict(csv_data_array):
    print(f'predict')

    # Use below to classify the image
    n = 4
    block_size = 7
    x_test_ravel = np.zeros((1,n**2,block_size**2))
    for img in range(1):
        ind = 0
        for row in range(n):
            for col in range(n):
                x_test_ravel[img, ind, :] = csv_data_array[(row * block_size):((row + 1) * block_size), (col * block_size):((col + 1) * block_size)].ravel()
                ind += 1

    pos_feed = np.array([list(range(n**2))]*1)
    transformer_predicted_output = transformer_model.predict([x_test_ravel,pos_feed])
    transformer_predicted_class = np.argmax(transformer_predicted_output)
    print(f'transformer_predicted_class: {transformer_predicted_class}')
    
    # Use the cnn_model to classify the image
    csv_data = csv_data_array.reshape(1,28, 28, 1)
    cnn_prediction = cnn_model.predict(csv_data)
    cnn_predicted_class = int(np.argmax(cnn_prediction))
    print(f'cnn_predicted_class: {cnn_predicted_class}')

    return cnn_predicted_class, trasformer_predicted_class

def to_image(csv_data):
    print(f'to_image')

    image_array = (csv_data * 255).reshape(28, 28).astype(np.uint8)
    image = Image.fromarray(image_array)
    
    # Convert the image to base64 to send back
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    print(f'img_str\n{img_str}')

    return img_str

def handler(event, context):
    try:
        logger.info(f"event: {event}")

        # Parse the event body JSON
        body = json.loads(event['body'])
        
        # Retrieve CSV data from the event
        csv_data = body.get('csv_data', '')
        logger.info(f"csv_data: {csv_data}")

        # Convert the preprocessed data to a NumPy array
        # Normalize pixel values to be in the range [0.0, 1.0] if they're in the 0-255 range
        csv_data = np.array(csv_data).astype(float)
        if csv_data.max() > 1.0:
            csv_data = csv_data / 255.0
        csv_data_array = np.array(csv_data)

        function = body.get('function', '')
        print(f'function: {function}')

        if function == 'predict':
            cnn_predicted_class, transformer_predicted_class = predict(csv_data_array)
            return {
                'cnn_predicted_label': cnn_predicted_class,
                'transformer_predicted_label': transformer_predicted_class
            }
        elif function == 'to_image':
            base64_img = to_image(csv_data_array)
            return {
                'base64_img': base64_img
            }
        else:
            return {"error": "Invalid function specified"}

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return {"error": str(e)}
