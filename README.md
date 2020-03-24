# Computational Autonomy for Materials Discovery (CAMD)
[![Build Status](https://travis-ci.com/TRI-AMDD/CAMD.svg?branch=master)](https://travis-ci.com/TRI-AMDD/CAMD)
[![Coverage Status](https://coveralls.io/repos/github/TRI-AMDD/CAMD/badge.svg?branch=master)](https://coveralls.io/github/TRI-AMDD/CAMD?branch=master)


## Installation

Note that, since qmpy is currently only python 2.7 compatible, CAMD python 3 
compatibility depends on a custom fork of qmpy [here](https://github.com/JosephMontoya-TRI/qmpy_py3), which is installed using
the `setup.py` procedure.

We recommend using Anaconda python, and creating a
fresh conda environment for the install (e. g. `conda create -n MY_ENV_NAME`).

### Linux

Install numpy via pip first, since the build depends on this and numpy has some difficulty recognizing
its own install.  Then install requirements and use setup.py.

```
pip install numpy
pip install -r requirements.txt
python setup.py develop
```

### Mac OSX

First dependencies via [homebrew](https://brew.sh/). Thanks to the contributors to this 
[stack exchange thread](https://stackoverflow.com/questions/12218229/my-config-h-file-not-found-when-intall-mysql-python-on-osx-10-8).

```
brew install gcc
```

Install numpy via pip first, since the build depends on this and numpy has some difficulty recognizing
its own install.  Then install requirements and use setup.py.

```
pip install numpy
pip install -r requirements.txt
python setup.py develop
```

## Data download

Datasets for featurized OQMD entries for after-the-fact testing can be 
downloaded at [data.matr.io/3](https://data.matr.io/3/).  These are done automatically
in the code and stored in the camd/_cache directory.
