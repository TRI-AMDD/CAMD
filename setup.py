from setuptools import setup, find_packages
import warnings

try:
    import numpy
except ImportError:
    # This is crude, but the best way I can figure to do this
    warnings.warn("Setup requires pre-installation of numpy, run pip "
                  "install numpy before setup.py")


setup(
    name='CAMD',
    version='1.0',
    packages=find_packages(),
    url='',
    license='',
    author='muratahan.aykol',
    author_email='murat.aykol@tri.global',
    description='',
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
                      "sqlalchemy",
                      "matminer==0.5.5",
                      "psycopg2==2.8.2",
                      "protosearch",
                      "autologging",
                      "awscli==1.16.199",
                      "docopt==0.6.2",
                      ],
    extras_require={
        "proto_dft": ["protosearch", "bulk_enumerator"]
    },
    dependency_links=[
        "http://github.com/JosephMontoya-TRI/qmpy_py3/tarball/master#egg=qmpy",
        "http://github.com/ToyotaResearchInstitute/bulk_enumerator/tarball/master#egg=bulk_enumerator",
        "http://github.com/ToyotaResearchInstitute/protosearch/tarball/master#egg=protosearch",
    ],
    entry_points={
        "console_scripts": [
            "camd_worker = camd.campaigns.worker:main"
        ]
    }
)
