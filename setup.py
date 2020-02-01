from setuptools import setup, find_packages
import warnings

try:
    import numpy
except ImportError:
    # This is goofy, but the best way I can figure to do this
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
    # Since qmpy can't be bothered to maintain
    # a proper install, pin numpy/networkx/matplotlib
    setup_requires=["numpy>=1.16"],
    install_requires=["numpy>=1.16",
                      "python-dateutil==2.8.0",
                      "networkx==2.2",
                      "matplotlib==3.1.1",
                      "qmpy",  # This version is constrained by the source
                      "tqdm==4.19.1",
                      "pandas==0.24.2",
                      "scikit-learn==0.22.1",
                      "boto3==1.9.136",
                      "monty==2.0.6",
                      "sqlalchemy",
                      "matminer==0.5.5",
                      "psycopg2==2.8.2",
                      "protosearch",
                      "autologging",
                      "awscli==1.16.199",
                      "docopt==0.6.2",
                      "tensorflow==1.15.0",
                      "gpflow==1.5.0"
                      ],
    # TODO: make this materials?
    dependency_links=[
        "http://github.com/JosephMontoya-TRI/qmpy_py3/tarball/master#egg=qmpy",
        "git+ssh://git@github.awsinternal.tri.global/materials/bulk_enumerator#egg=bulk_enumerator",
        "git+ssh://git@github.awsinternal.tri.global/materials/protosearch#egg=protosearch"
    ],
    entry_points={
        "console_scripts": [
            "camd_worker = camd.campaigns.worker:main"
        ]
    }
)
