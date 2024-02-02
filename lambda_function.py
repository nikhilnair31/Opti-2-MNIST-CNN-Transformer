import tensorflow as tf
from PIL import Image
import pandas as pd
import numpy as np
import logging
import base64
import boto3
import json
import io

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Load the saved model
s3 = boto3.client('s3')
bucket_name = 'opti-tf-test-lambda'
model_key = 'Models/model.h5'

# Download the model file from S3 to a local temporary file
cnn_model_path = '/tmp/cnn_model.h5'
s3.download_file(bucket_name, model_key, cnn_model_path)
trasformer_model_path = '/tmp/trasformer_model.h5'
s3.download_file(bucket_name, model_key, trasformer_model_path)

# Load the model from the local file
cnn_model = tf.keras.models.load_model(cnn_model_path)
trasformer_model = tf.keras.models.load_model(trasformer_model_path)

def handler(event, context):
    try:
        logger.info(f"event: {event}")

        # Parse the event body JSON
        body = json.loads(event['body'])

        # Retrieve CSV data from the event
        csv_data_base64 = body.get('csv_data', '')
        csv_data = base64.b64decode(csv_data_base64).decode('utf-8')

        # Load CSV data into a DataFrame
        data = pd.read_csv(io.StringIO(csv_data), header=None)

        # Preprocess the image data
        image_data = data.values.reshape(1, 28, 28)
        image_data = image_data / 255.0

        # Use the cnn_model to classify the image
        cnn_prediction = cnn_model.predict(image_data)
        cnn_predicted_class = int(np.argmax(cnn_prediction))
        # Use the cnn_model to classify the image
        trasformer_prediction = trasformer_model.predict(image_data)
        trasformer_predicted_class = int(np.argmax(trasformer_prediction))

        # Return the result
        return {
            'cnn_predicted_label': cnn_predicted_class,
            'trasformer_predicted_label': trasformer_predicted_class
        }

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return {"error": str(e)}
