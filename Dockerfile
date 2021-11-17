FROM continuumio/miniconda3

# Activate shell
SHELL ["/bin/bash", "-c"]
ENV PATH="/opt/conda/bin/:$PATH"

RUN mkdir -p /home/camd && \
    conda create -n camd python=3.7 && \
    apt-get update && \
    apt-get -y install gcc g++

WORKDIR /home/camd

# Create camd env
ENV PATH="/opt/conda/envs/camd/bin:$PATH"

COPY setup.py requirements.txt /home/camd/

# Install package + awscli for Mc1
RUN source /opt/conda/bin/activate camd && \
    pip install `grep numpy requirements.txt` && \
    pip install -r requirements.txt && \
    pip install awscli

COPY camd /home/camd/camd
RUN pip install -e .
