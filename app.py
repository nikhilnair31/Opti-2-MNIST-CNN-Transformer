import streamlit as st
import pandas as pd
import numpy as np
import base64
import requests
import io

lambda_function_url = 'https://b6xp6og5g5s2bjmcabdi73odw40zimzl.lambda-url.ap-south-1.on.aws/'

# Function to convert CSV data to image representation
def csv_to_image(csv_data):
    # Use pandas to read the CSV file from the uploaded file-like object
    df = pd.read_csv(io.StringIO(csv_data.decode('utf-8')), header=None)
    
    # Check if the DataFrame has 28 rows and 28 columns
    if df.shape != (28, 28):
        st.error("CSV data should contain exactly 28x28 pixel values.")
        return None

    # Convert the DataFrame to a NumPy array
    image = df.to_numpy()

    # Check if normalization is needed (assuming the values are either in 0-255 or 0-1 range)
    if image.max() > 1.0:
        # Normalize pixel values to be in the range [0.0, 1.0] if they're in the 0-255 range
        image = image / 255.0

    return image

# Function to call an example API
def call_api(uploaded_file):
    csv_data = base64.b64encode(uploaded_file.getvalue()).decode('utf-8')

    # Create the event object
    event = {
        "csv_data": csv_data
    }

    # Call the Lambda function
    response = requests.post(lambda_function_url, json=event)
    print(f'Response:\n{response}\n')

    # Check the response status code
    if response.status_code != 200:
        print(f'Error: {response.status_code} {response.reason}\n')
        return "API Error"

    # Check the response content type
    content_type = response.headers.get('Content-Type')
    if not content_type or not content_type.startswith('application/json'):
        print('Error: Invalid content type')
        return "API Error"

    # Parse the response as JSON
    response_json = response.json()

    cnn_response_text = response_json['cnn_predicted_label']
    transformer_response_text = response_json['transformer_predicted_label']

    # Return the response
    return cnn_response_text, transformer_response_text

# Streamlit app
def main():
    st.title("Project 1 - Image Classification")

    # File upload
    st.subheader("File Upload")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        # Read the CSV file as bytes and encode it with base64
        csv_data = base64.b64encode(uploaded_file.getvalue()).decode('utf-8')

        # Convert CSV to image
        st.subheader("Image Representation")
        image = csv_to_image(uploaded_file.getvalue())
        st.image(image, caption="CSV to Image", width=100, use_column_width=True)

        # Button to call API
        if st.button("Predict Number"):
            with st.spinner("Running Inference..."):
                # Call API with the image representation
                cnn_response_text, transformer_response_text = call_api(uploaded_file)
                st.subheader(f"Predicted Number")
                st.text(f"CNN: {cnn_response_text}")
                st.text(f"Transformer: {transformer_response_text}")

if __name__ == "__main__":
    main()
