import tensorflow as tf
import numpy as np
import logging
import base64
import boto3
import json
import io

# label mapping
labels = '''airplane automobile bird cat deer dog frog horse ship truck'''.split()

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Load the saved model
s3 = boto3.client('s3')
bucket_name = 'opti-tf-test-lambda'
model_key = 'Models/model.h5'

# Download the model file from S3 to a local temporary file
local_model_path = '/tmp/model.h5'
s3.download_file(bucket_name, model_key, local_model_path)

# Load the model from the local file
model = tf.keras.models.load_model(local_model_path)

def handler(event, context):
    try:
        logger.info(f"event: {event}")

        # Parse the event body JSON
        body = json.loads(event['body'])

        # Load the input image from the event
        image_data = base64.b64decode(body['image'])
        image = Image.open(io.BytesIO(image_data))
        image = image.resize((32, 32))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # Pass the image through the network and get the predicted label
        prediction = model.predict(image_array)
        predicted_label_index = np.argmax(prediction)
        predicted_label = labels[predicted_label_index]

        # Load the original label
        original_label_index = body['label']
        original_label = labels[original_label_index]

        # Return the result
        return {
            'original_label': original_label,
            'predicted_label': predicted_label
        }

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return {"error": str(e)}
