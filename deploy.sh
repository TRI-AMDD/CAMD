#!/bin/bash
set -e

export DOCKER_TAG=camd-public

if [[ "$TRAVIS_BRANCH" =~ ^(master|dev)$ ]]; then
    docker build -t $DOCKER_TAG .
    docker tag $DOCKER_TAG:latest $DOCKER_URI:$TRAVIS_BRANCH-latest
    aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $DOCKER_URI
    docker push $DOCKER_URI:$TRAVIS_BRANCH-latest
else
    echo "TRAVIS_BRANCH $TRAVIS_BRANCH not in push list"
fi
