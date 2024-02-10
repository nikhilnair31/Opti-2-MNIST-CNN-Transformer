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

    image_array = (image_data * 255).reshape(28, 28).astype(np.uint8)
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
        image_data = np.array(csv_data).astype(float)
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
