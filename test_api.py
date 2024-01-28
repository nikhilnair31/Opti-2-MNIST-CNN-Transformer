import pandas as pd
import requests
import base64

# Replace with your Lambda function URL
lambda_function_url = 'https://b6xp6og5g5s2bjmcabdi73odw40zimzl.lambda-url.ap-south-1.on.aws/'

# Read the CSV file as bytes and encode it with base64
csv_file_path = 'Data/test_image.csv'
df = pd.read_csv(csv_file_path, header=None)
print(f'df:\n{df}')

csv_data_as_bytes = df.to_csv(index=False).encode('utf-8')
base64_encoded = base64.b64encode(csv_data_as_bytes).decode('utf-8')
print(f'base64_encoded:\n{base64_encoded}')

# Create the event object
event = {
    "csv_data": base64_encoded
}
print(f'event:\n{event}')

# Call the Lambda function
response = requests.post(lambda_function_url, json=event)
print(f'Response:\n{response}\n')

# Check the response status code
if response.status_code != 200:
    print(f'Error: {response.status_code} {response.reason}\n')
    exit(1)

# Check the response content type
content_type = response.headers.get('Content-Type')
if not content_type or not content_type.startswith('application/json'):
    print('Error: Invalid content type')
    exit(1)

# Parse the response as JSON
response_json = response.json()

# Print the response
print(response_json)