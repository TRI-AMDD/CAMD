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

LONG_DESCRIPTION = "camd is "

setup(
    name='CAMD',
    url="https://github.com/ToyotaResearchInstitute/beep",
    version="2020.2.24",
    packages=find_packages(),
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    setup_requires=["numpy==1.18",
                    "Django==2.2",
                    "tensorflow==1.15.0",
                    "gpflow==1.5.0"],  # For qmpy depnedency
    install_requires=["numpy==1.18",
                      "python-dateutil==2.8.0",
                      "networkx==2.2",
                      "matplotlib==3.1.1",
                      "qmpy",  # This version is constrained by the source
                      "pandas==0.24.2",
                      "boto3==1.9.136",
                      "matminer==0.5.5",
                      "autologging",
                      "awscli==1.16.199",
                      "docopt==0.6.2",
                      ],
    extras_require={
        "proto_dft": ["protosearch"],
        "tests": ["nose",
                  "coverage",
                  "pylint",
                  "memory_profiler",
                  "matplotlib"]
    },
    dependency_links=[
        "http://github.com/JosephMontoya-TRI/qmpy_py3/tarball/master#egg=qmpy",
        "http://github.com/ToyotaResearchInstitute/protosearch/tarball/master#egg=protosearch",
    ],
    entry_points={
        "console_scripts": [
            "camd_worker = camd.campaigns.worker:main"
        ]
    },
    classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: Apache Software License",
          "Operating System :: OS Independent",
    ],
    package_data={
        "beep.conversion_schemas": ["*.yaml", "*.md"],
        "beep.procedure_templates": ["*.000", "*.csv", "*.json"],
        "beep.validation_schemas": ["*.yaml"],
    },
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
