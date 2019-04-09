FROM continuumio/miniconda

# Activate shell
SHELL ["/bin/bash", "-c"]

WORKDIR /home
RUN mkdir -p /home/camd
WORKDIR /home/camd

# Create BEEP_EP env
RUN conda create -n camd python=2.7
ENV PATH="/opt/conda/envs/camd/bin:$PATH"
# ENV SHOW_TQDM=false

COPY . /home/camd

# Install camd
RUN source /opt/conda/bin/activate camd

# Goofiness for MySQL-python
# RUN apt-get install libmysqlclient-dev python-dev
RUN apt-get update
RUN apt install -y default-libmysqlclient-dev gcc
ENV PATH=$PATH:/usr/local/mysql/bin/

# Goofy numpy pre-install
RUN pip install numpy

# Install package
RUN pip install -e .
RUN pip install nose
RUN pip install coverage
RUN pip install pylint
