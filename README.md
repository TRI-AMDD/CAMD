# Computational Autonomy for Materials Discovery (CAMD)
![Testing - main](https://github.com/TRI-AMDD/CAMD/workflows/Testing%20-%20main/badge.svg)
![Linting](https://github.com/TRI-AMDD/CAMD/workflows/Linting/badge.svg)
[![Coverage Status](https://coveralls.io/repos/github/TRI-AMDD/CAMD/badge.svg)](https://coveralls.io/github/TRI-AMDD/CAMD)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/TRI-AMDD/camd/binder?labpath=examples%2Fmain_tutorial.ipynb)

CAMD provides a flexible software framework for sequential / Bayesian optimization type campaigns for materials discovery. Its key features include:
* **Agents**: Decision making entities which select experiments to run from pre-determined candidate sets. Agents can combine machine learning with physical or chemical constructs, logic, heuristics, exploration-exploitation strategies and so on. CAMD comes with several generic and structure-discovery focused agents, which can be used by the users as templates to derive new ones.
* **Experiments**: Entities responsible for carrying out the experiments requested by Agents and reporting back the results.
* **Analyzers**: Post-processing procedures which frame experimental results in the context of candidate or seed datasets.
* **Campaigns**: Loop construct which executes the sequence of hypothesize-experiment-analyze by the Agent, Experiment, and Analyzer, respectively, and facilitates the communication between these entities.
* **Simulations**: Agent performance can be simulated using after-the-fact sampling of known existing data. This allows systematic design and tuning of agents before their deployment using actual Experiments.

A more in-depth description of the scientific framework can be found in [this recent open-access article](https://doi.org/10.1039/D0SC01101K), which demonstrates  an end-to-end CAMD-based framework for autonomous inorganic materials discovery using cloud-based density functional theory calculations.

## Getting started
For a quick start, explore the tutorial with [binder](https://mybinder.org/v2/gh/TRI-AMDD/camd/binder?labpath=examples%2Fmain_tutorial.ipynb).
If you want to install locally, following the instructions
below and explore the [examples](github.com/TRI-AMDD/camd/examples).

## Installation

CAMD can be installed using pip as `pip install camd`. If issues are encountered, we recommend following the installation procedures below:

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

## Citation
If you use CAMD, we kindly ask you to cite the following publication:
* Montoya, J. H., Winther, K. T., Flores, R. A., Bligaard, T., Hummelshøj, J. S., & Aykol, M. "Autonomous intelligent agents for accelerated materials discovery"  *Chemical Science* **11** (2020) 8517–8532 [doi:10.1039/D0SC01101K, open-access](https://doi.org/10.1039/D0SC01101K).
