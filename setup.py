from setuptools import setup, find_packages
import warnings

try:
    import numpy
except ImportError:
    # This is crude, but the best way I can figure to do this
    warnings.warn("Setup requires pre-installation of numpy, run pip "
                  "install numpy before setup.py")

DESCRIPTION = "camd is software designed to support autonomous materials " \
              "research and sequential learning"

LONG_DESCRIPTION = """
camd is software designed to support Computational Autonomy for Materials Discovery
based on ongoing work led by the
[Toyota Research Institute](http://www.tri.global/accelerated-materials-design-and-discovery/).

camd enables the construction of sequential learning pipelines using a set of
abstractions that include
* Agents - decision making entities which select experiments to run from pre-determined
    candidate sets
* Experiments - experimental procedures which augment candidate data in a way that
    facilitates further experiment selection
* Analyzers - Post-processing procedures which frame experimental results in the context
    of candidate or seed datasets

In addition to these abstractions, camd provides a loop construct which executes
the sequence of hypothesize-experiment-analyze by the Agent, Experiment, and Analyzer,
respectively.  Simulations of agent performance can also be conducted using
after the fact sampling of known data.
"""

setup(
    name='camd',
    url="https://github.com/TRI-AMDD/CAMD",
    version="2020.5.19",
    packages=find_packages(),
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    setup_requires=["numpy==1.18.3",
                    "Django==2.2.10",
                    "tensorflow==1.15.2",
                    "gpflow==1.5.0"],  # For qmpy depnedency
    install_requires=["python-dateutil==2.8.0",
                      "networkx==2.2",
                      "matplotlib==3.1.1",
                      "qmpy",  # This version is constrained by the source
                      "pandas==1.0.1",
                      "matminer==0.6.3",
                      "autologging",
                      "awscli==1.18.28",
                      "docopt==0.6.2",
                      "boto3==1.12.28",
                      "scikit-learn==0.22.1",
                      "taburu==2020.5.9"
                      ],
    extras_require={
        "proto_dft": ["protosearch==2020.5.10"],
        "tests": ["pytest",
                  "pytest-cov",
                  "coveralls"
                  ]
    },
    dependency_links=[
        "https://github.com/JosephMontoya-TRI/qmpy_py3/tarball/master#egg=qmpy",
        "https://github.com/ToyotaResearchInstitute/protosearch/tarball/master#egg=protosearch",
    ],
    entry_points={
        "console_scripts": [
            "camd_worker = camd.campaigns.worker:main",
            "camd_runner = camd.campaigns.runner:main"
        ]
    },
    classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: Apache Software License",
          "Operating System :: OS Independent",
    ],
    include_package_data=True,
    author="AMDD - Toyota Research Institute",
    author_email="murat.aykol@tri.global",
    maintainer="Murat Aykol, Joseph Montoya",
    maintainer_email="murat.aykol@tri.global",
    license="Apache",
    keywords=[
        "materials", "battery", "chemistry", "science",
        "density functional theory", "energy", "AI", "artificial intelligence",
        "sequential learning", "active learning"
    ],
    )