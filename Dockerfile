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

# Set TQDM to be off in tests
ENV TQDM_OFF=1

# Goofy numpy pre-install
RUN pip install numpy

# Install package
RUN pip install -e .
RUN pip install nose
RUN pip install coverage
RUN pip install pylint
