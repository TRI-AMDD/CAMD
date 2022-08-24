from setuptools import setup, find_packages


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
    version="2022.8.24",
    packages=find_packages(),
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    install_requires=["python-dateutil==2.8.2",
                      "networkx==2.8.5",
                      "matplotlib==3.5.3",
                      "matminer==0.7.8",
                      "awscli",
                      "boto3==1.24.51",
                      "docopt==0.6.2",
                      "taburu==2020.5.9",
                      "GPy==1.10.0",
                      "watchtower==3.0.0",
                      "qmpy-tri>=2021.7.21"
                      ],
    extras_require={
        "proto_dft": ["protosearch==2020.5.10"],
        "m3gnet": ["m3gnet"],
        "atomate": ["atomate"],
        "tests": ["pytest",
                  "pytest-cov",
                  "coveralls"
                  ]
    },
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
