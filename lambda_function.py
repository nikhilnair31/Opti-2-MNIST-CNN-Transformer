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
    # Use the cnn_model to classify the image
    cnn_prediction = cnn_model.predict(image_data)
    cnn_predicted_class = int(np.argmax(cnn_prediction))
    # Use the cnn_model to classify the image
    trasformer_prediction = trasformer_model.predict(image_data)
    trasformer_predicted_class = int(np.argmax(trasformer_prediction))

    return cnn_predicted_class, trasformer_predicted_class

def to_image(image_data):
    # Convert the image data to bytes
    img = Image.fromarray((image_data * 255).astype(np.uint8))
    img_byte_array = io.BytesIO()
    img.save(img_byte_array, format='PNG')
    img_byte_array = img_byte_array.getvalue()

    # Encode the image data in base64
    base64_img = base64.b64encode(img_byte_array).decode('utf-8')

    return base64_img

def handler(event, context):
    try:
        logger.info(f"event: {event}")

        # Parse the event body JSON
        body = json.loads(event['body'])

        # Retrieve CSV data from the event
        csv_data_base64 = body.get('csv_data', '')
        csv_data = base64.b64decode(csv_data_base64).decode('utf-8')
        print(f'csv_data\n{csv_data}')
        
        # Load CSV data into a DataFrame
        csv_df = pd.read_csv(io.StringIO(csv_data), header=None)
        print(f'csv_df\n{csv_df}')
        
        # Convert the DataFrame to a NumPy array
        image_data = csv_df.to_numpy()
        image_data = np.array(image_data, dtype=np.float64)
        print(f'image_data\n{image_data}')

        # Check if normalization is needed (assuming the values are either in 0-255 or 0-1 range)
        if image_data.max() > 1.0:
            # Normalize pixel values to be in the range [0.0, 1.0] if they're in the 0-255 range
            image_data = image_data / 255.0

        function = body.get('function', '')
        if function == 'predict':
            cnn_predicted_class, transformer_predicted_class = predict(image_data)
            return {
                'cnn_predicted_label': cnn_predicted_class,
                'transformer_predicted_label': transformer_predicted_class
            }
        elif function == 'to_image':
            base64_img = to_image(image_data)
            return {
                'base64_img': base64.b64encode(base64_img).decode('utf-8')
            }
        else:
            return {"error": "Invalid function specified"}

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return {"error": str(e)}
