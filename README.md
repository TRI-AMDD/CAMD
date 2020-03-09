# Computational Autonomy for Materials Discovery (CAMD)


## Installation

Note that, since qmpy is currently only python 2.7 compatible, CAMD python 3 
compatibility depends on a custom fork of qmpy [here](https://github.com/JosephMontoya-TRI/qmpy_py3), which is installed using
the `setup.py` procedure.

We recommend using Anaconda python, and creating a
fresh conda environment for the install (e. g. `conda create -n MY_ENV_NAME`).

### Linux

Update packages via apt and set pathing for MySQL dependency:

```angular2
apt-get update
apt install -y default-libmysqlclient-dev gcc
export PATH=$PATH:/usr/local/mysql/bin
```

Install numpy/Django via pip first, since the build depends on this and numpy has some difficulty recognizing
its own install:

```angular2
pip install numpy
pip install Django
```

Then use the included setup.py procedure, from the cloned directory.

```angular2
python setup.py develop
```

### Mac OSX

First dependencies via [homebrew](https://brew.sh/). Thanks to the contributors to this 
[stack exchange thread](https://stackoverflow.com/questions/12218229/my-config-h-file-not-found-when-intall-mysql-python-on-osx-10-8).

```angular2
$ brew install mysql
$ brew install postgresql
$ brew install gcc
```

Install numpy/Django via pip first, since the build depends on these and numpy has some difficulty recognizing
its own install:

```angular2
pip install numpy
pip install Django
```

Then use the included setup.py procedure, from the cloned directory.

```angular2
python setup.py develop
```

## Data download

Datasets for featurized OQMD entries for after-the-fact testing can be 
downloaded at [data.matr.io/3](https://data.matr.io/3/)
