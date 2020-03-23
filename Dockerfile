FROM continuumio/miniconda3

# Activate shell
SHELL ["/bin/bash", "-c"]
ENV PATH="/opt/conda/bin/:$PATH"

RUN mkdir -p /home/camd && \
    conda create -n camd python=3.6 && \
    apt-get update && \

WORKDIR /home/camd

# Create camd env
ENV PATH="/opt/conda/envs/camd/bin:$PATH"

COPY . /home/camd

# Install package
RUN source /opt/conda/bin/activate camd && \
    pip install -r requirements.txt && \
    python setup.py develop
