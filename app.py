import streamlit as st
import pandas as pd
import numpy as np
import base64
import requests
import io

lambda_function_url = 'https://b6xp6og5g5s2bjmcabdi73odw40zimzl.lambda-url.ap-south-1.on.aws/'

# Function to convert CSV data to image representation
def csv_to_image(csv_data):
    # Convert CSV data to a list of values
    values = list(map(float, csv_data.decode('utf-8').split(',')))

    # Ensure that the list has exactly 784 values
    if len(values) != 28 * 28:
        st.error("CSV data should contain exactly 28x28 pixel values.")
        return None

    # Convert the list to a NumPy array and reshape it into a 28x28 image
    image = np.array(values).reshape((28, 28))

    # Normalize pixel values to be in the range [0.0, 1.0]
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

    # Return the response
    return response.text

# Streamlit app
def main():
    st.title("CSV to Image and API Call App")

    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        # Read the CSV file as bytes and encode it with base64
        csv_data = base64.b64encode(uploaded_file.getvalue()).decode('utf-8')

        # Convert CSV to image
        st.subheader("Image Representation:")
        image = csv_to_image(uploaded_file.getvalue())
        st.image(image, caption="CSV to Image", use_column_width=True)

        # Button to call API
        if st.button("Call API"):
            # Call API with the image representation
            response_text = call_api(uploaded_file)
            st.subheader("API Response:")
            st.write(response_text)

if __name__ == "__main__":
    main()
