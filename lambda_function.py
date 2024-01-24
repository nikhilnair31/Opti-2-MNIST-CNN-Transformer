import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import boto3
import io

# label mapping
labels = '''airplane automobile bird cat deer dog frog horse ship truck'''.split()

# Load the saved model
s3 = boto3.client('s3')
bucket_name = 'opti-tf-test-lambda'
model_key = 'Models/model.h5'
model_stream = s3.get_object(Bucket=bucket_name, Key=model_key)['Body']
model = tf.keras.models.load_model(io.BytesIO(model_stream.read()))

def lambda_handler(event, context):
    # Load the input image from the event
    image_data = base64.b64decode(event['image'])
    image = Image.open(io.BytesIO(image_data))
    image = image.resize((32, 32))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Pass the image through the network and get the predicted label
    prediction = model.predict(image_array)
    predicted_label_index = np.argmax(prediction)
    predicted_label = labels[predicted_label_index]

    # Load the original label
    original_label_index = event['label']
    original_label = labels[original_label_index]

    # Return the result
    return {
        'original_label': original_label,
        'predicted_label': predicted_label
    }
