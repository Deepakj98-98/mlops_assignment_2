#!/bin/bash
set -eo pipefail
env=$1
aws s3 mb s3://mlops-23-09-2024-bits-$env
ARTIFACT_BUCKET=mlops-23-09-2024-bits-$env
aws cloudformation package --template-file template.yml --s3-bucket $ARTIFACT_BUCKET --output-template-file out.yml --region us-east-1
aws cloudformation deploy --template-file out.yml --stack-name mlops-lambda-$env --capabilities CAPABILITY_NAMED_IAM --region us-east-1 --parameter-overrides file://aws_cloudformation_params/$env.json
