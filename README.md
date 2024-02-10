<div style="text-align: center;">
    <img src="./other/Opti II MNIST.png" width="200" />
</div>
</br>


# TensorFlow Image Classification Lambda Function

This repository contains code for an AWS Lambda function that performs image classification using TensorFlow models. The function is designed to be triggered via an API Gateway endpoint, accepting CSV data representing images, and returning predictions or image data.

## Requirements

- Python 3.x
- TensorFlow
- Pillow (PIL)
- pandas
- numpy
- boto3

## Usage

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/tensorflow-image-classification-lambda.git
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure AWS Credentials:**
   Ensure you have AWS credentials set up either through AWS CLI or environment variables.

4. **Upload Models to S3:**
   Upload your TensorFlow models to an S3 bucket. The Lambda function will download these models during execution.

5. **Deploy Lambda Function:**
   Deploy the Lambda function using the provided code. You can use AWS CLI or AWS Management Console for this.

6. **Invoke Lambda Function:**
   Use API Gateway to create an endpoint that triggers the Lambda function. Pass CSV data representing images in the request body along with the desired function (`predict` or `to_image`).

## Code Explanation

- `predict`: This function takes image data as input, reshapes it, and performs classification using two pre-trained TensorFlow models: a CNN model and a transformer model. It returns the predicted classes for both models.

- `to_image`: This function takes image data, converts it to a PIL Image, and then converts the image to base64 format for transmission.

- `handler`: This is the main Lambda function handler. It receives the event and context objects, extracts CSV data representing images from the request body, and invokes either `predict` or `to_image` based on the specified function. It returns the results or any errors encountered during execution.

## Example

Here's an example of how to invoke the Lambda function via API Gateway:

```json
{
  "csv_data": [0.2, 0.4, 0.1, ..., 0.9], // CSV data representing image pixels
  "function": "predict" // or "to_image"
}
```

Replace `csv_data` with your actual image data and `function` with either `predict` or `to_image`.