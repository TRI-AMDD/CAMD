#!/bin/bash

mkdir -p ~/matr.io
aws s3 sync s3://matr.io/bin ~/matr.io/bin
aws s3 sync s3://matr.io/env ~/matr.io/env

chmod +x ~/matr.io/bin/*

$(aws ecr get-login --no-include-email --region us-west-2)
docker pull 251589461219.dkr.ecr.us-west-2.amazonaws.com/camd-worker:internal-campaigns-latest
