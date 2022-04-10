# LAMBDA FUNCTION FOR EMAIL SPAM DETECTION

This lambda function receive S3 PUT event and extract email from it. The email is fed to Amazon Sagemaker Endpoint for spam labeling. Finally a report will be sent through Amazon SES.

## Prerequisite
- Python >= 3.8
- Pip

## How to Build
- Prepare the necessary layer for required packages:
    - Sagemaker (without numpy and pandas dependency): follow instruction under layer directory
    - Numpy and Pandas: installing numpy and pandas in lambda is tricky, I advise to use existing layer for it e.g. arn:aws:lambda:us-east-1:770693421928:layer:Klayers-p39-pandas:2
- Use `lambda_function.py` as the main lambda handler.