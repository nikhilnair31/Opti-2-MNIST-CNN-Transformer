import tensorflow as tf
from PIL import Image
import pandas as pd
import numpy as np
import logging
import base64
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
trasformer_model_key = 'Models/cnn_model.h5'

# Download the model file from S3 to a local temporary file
cnn_model_path = '/tmp/cnn_model.h5'
s3.download_file(bucket_name, cnn_model_key, cnn_model_path)
trasformer_model_path = '/tmp/trasformer_model.h5'
s3.download_file(bucket_name, trasformer_model_key, trasformer_model_path)

# Load the model from the local file
cnn_model = tf.keras.models.load_model(cnn_model_path)
trasformer_model = tf.keras.models.load_model(trasformer_model_path)

def predict(image_data):
    print(f'predict')

    # Use the cnn_model to classify the image
    cnn_prediction = cnn_model.predict(image_data)
    cnn_predicted_class = int(np.argmax(cnn_prediction))
    # Use the cnn_model to classify the image
    trasformer_prediction = trasformer_model.predict(image_data)
    trasformer_predicted_class = int(np.argmax(trasformer_prediction))
    print(f'cnn_predicted_class: {cnn_predicted_class}\ntrasformer_predicted_class: {trasformer_predicted_class}')

    return cnn_predicted_class, trasformer_predicted_class

def to_image(image_data):
    print(f'to_image')

    # Convert array to 8-bit unsigned integer
    uint8_array = image_data.astype(np.uint8)

    # Create PIL Image
    image = Image.fromarray(uint8_array)

    # Convert PIL Image to bytes
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='PNG')
    img_bytes.seek(0)

    # Encode the image bytes to base64. The result is still in bytes.
    base64_encoded_image = base64.b64encode(img_bytes.getvalue())
    
    # Decode the base64 bytes object to a string to make it easy to work with.
    base64_string = base64_encoded_image.decode('utf-8')
    print(f'base64_string\n{base64_string}')

    return base64_string

def handler(event, context):
    try:
        logger.info(f"event: {event}")

        # Parse the event body JSON
        body = json.loads(event['body'])

        # Retrieve CSV data from the event
        csv_data_base64 = body.get('csv_data', '')
        csv_data = base64.b64decode(csv_data_base64).decode('utf-8')
        print(f'csv_data\n{csv_data}')

        # Parse string to list of lists
        list_of_lists = ast.literal_eval(csv_data)
        list_of_lists = [[float(entry) for entry in row] for row in list_of_lists]
        print(f'list_of_lists\n{list_of_lists}')

        # Convert the preprocessed data to a NumPy array
        # Normalize pixel values to be in the range [0.0, 1.0] if they're in the 0-255 range
        image_data = np.array(list_of_lists)
        if image_data.max() > 1.0:
            image_data = image_data / 255.0

        function = body.get('function', '')
        print(f'function: {function}')

        if function == 'predict':
            cnn_predicted_class, transformer_predicted_class = predict(image_data)
            return {
                'cnn_predicted_label': cnn_predicted_class,
                'transformer_predicted_label': transformer_predicted_class
            }
        elif function == 'to_image':
            base64_img = to_image(image_data)
            return {
                'base64_img': base64_img
            }
        else:
            return {"error": "Invalid function specified"}

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return {"error": str(e)}
