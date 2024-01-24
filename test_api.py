import requests

# Replace with your Lambda function URL
lambda_function_url = 'https://b6xp6og5g5s2bjmcabdi73odw40zimzl.lambda-url.ap-south-1.on.aws/'

# Create the event object
event = {
    'image_file_path': "Images/0000.jpg",
    'label_data': "airplane"
}

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