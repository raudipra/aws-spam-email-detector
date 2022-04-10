# SAGEMAKER LAYER

## Prerequisite
- Python >= 3.8
- Pip

## How to Build
- `mkdir python`
- `pip3 install --no-deps -t python/ -r requirements.txt`
- `zip -r sagemaker.zip python`
- Upload to S3 `aws s3 mv sagemaker.zip <BUCKET_URL>`
- Create layer from console using S3 object above