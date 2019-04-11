# CAMD
Computational Autonomy for Materials Discovery

Dataset for featurized OQMD entries for after-the-fact testing can be downloaded from a dedicated S3 bucket at: https://s3.console.aws.amazon.com/s3/buckets/ml-dash-datastore/

CAMD documents and flow diagram for stable material discovery is available on [Google Drive](https://drive.google.com/open?id=1wvPy4qOzY_-AD5xar4SeUQ4GlcDrzF77).


## Installation

Note that, since qmpy is currently only python 2.7 compatible, CAMD is similarly
only python 2.7 compatible.

### Linux

Update packages via apt and set pathing for MySQL dependency:

```angular2
apt-get update
apt install -y default-libmysqlclient-dev gcc
export PATH=$PATH:/usr/local/mysql/bin
```

Install numpy via pip first, since the build depends on this and numpy has some difficulty recognizing
its own install:

```angular2
pip install numpy
```

Then use the included setup.py procedure, from the cloned directory.

```angular2
pip install -e .
```

### Mac OSX

First dependencies via [homebrew](https://brew.sh/). Thanks to the contributors to this 
[stack exchange thread](https://stackoverflow.com/questions/12218229/my-config-h-file-not-found-when-intall-mysql-python-on-osx-10-8).

```angular2
$ brew install mysql
$ brew unlink mysql
$ brew install mysql-connector-c
$ sed -i -e 's/libs="$libs -l "/libs="$libs -lmysqlclient -lssl -lcrypto"/g' /usr/local/bin/mysql_config
$ pip install MySQL-python
$ brew unlink mysql-connuiesdfuuuuctor-c
$ brew link --overwrite mysql
```

Install numpy via pip first, since the build depends on this and numpy has some difficulty recognizing
its own install:

```angular2
pip install numpy
```

Then use the included setup.py procedure, from the cloned directory.

```angular2
pip install -e .
```

