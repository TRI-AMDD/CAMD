FROM continuumio/miniconda3

# Activate shell
SHELL ["/bin/bash", "-c"]

WORKDIR /home
RUN mkdir -p /home/camd
WORKDIR /home/camd

# Create BEEP_EP env
RUN conda create -n camd python=3.6
ENV PATH="/opt/conda/envs/camd/bin:$PATH"
# ENV SHOW_TQDM=false

COPY . /home/camd

# Install camd
RUN source /opt/conda/bin/activate camd

# Update mysql
RUN apt-get update
RUN apt-get install -y gcc default-libmysqlclient-dev

# Set TQDM to be off in tests
ENV TQDM_OFF=1

# Goofy numpy pre-install
RUN pip install numpy
RUN pip install Django

# Install package
RUN python setup.py develop
RUN pip install nose
RUN pip install coverage
RUN pip install pylint
