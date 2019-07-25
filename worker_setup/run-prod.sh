#!/bin/bash

$(aws ecr get-login --no-include-email --region us-west-2)
docker pull 251589461219.dkr.ecr.us-west-2.amazonaws.com/camd-worker:internal-master-latest

# This is the worst
docker run -it \
	-e TRI_PATH=/home/camd/matr.io -e CAMD_S3_BUCKET=camd-runs -e TRI_BUCKET=matr.io \
	-v /home/ec2-user/matr.io:/home/camd/matr.io/ \
	251589461219.dkr.ecr.us-west-2.amazonaws.com/camd-worker:internal-master-latest \
	bash -c 'export PATH=$PATH:$TRI_PATH/bin; python -m camd.campaigns.worker'
