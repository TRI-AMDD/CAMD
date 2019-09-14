#!/usr/bin/env bash

# This script is run as the default procedure for testing
# in the docker container

set -e

# Set TQDM to be off in tests
# export TQDM_OFF=1

# TODO: set up postgresql testing
# Start postgres in the background
# service postgresql start

# Run nosetests
nosetests --with-xunit --all-modules --traverse-namespace \
    --with-coverage --cover-package=camd --cover-inclusive

# Generate coverage
python -m coverage xml --include=camd*

# Do linting
pylint -f parseable -d I0011,R0801 camd | tee pylint.out

