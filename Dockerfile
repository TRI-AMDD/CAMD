FROM continuumio/miniconda3

# Activate shell
SHELL ["/bin/bash", "-c"]
ENV PATH="/opt/conda/bin/:$PATH"

RUN mkdir -p /home/camd && \
    conda create -n camd python=3.6 && \
    apt-get update && \
    apt-get install -y gcc g++ default-libmysqlclient-dev libpq-dev postgresql

WORKDIR /home/camd

# Create camd env
ENV PATH="/opt/conda/envs/camd/bin:$PATH"

COPY . /home/camd

# Install package
RUN source /opt/conda/bin/activate camd && \
    pip install gpflow==1.5.0 tensorflow==1.15.0 && \
    cd bulk_enumerator && python setup.py install && cd .. && \
    cd protosearch && python setup.py install && cd .. && \
    python setup.py develop && \
    pip install nose && \
    pip install coverage && \
    pip install pylint && \
    chmod +x dockertest.sh

CMD ["./dockertest.sh"]
