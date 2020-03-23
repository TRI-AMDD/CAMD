#!/bin/bash

export DOCKER_TAG=camd-public

docker build -t $DOCKER_TAG .
docker tag $DOCKER_TAG:latest $DOCKER_URI:latest
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $DOCKER_URI
docker push $DOCKER_URI:latest
