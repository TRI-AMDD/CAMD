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
                      "networkx==2.2",
                      "matplotlib",
                      "qmpy",
                      "tqdm",
                      "pandas",
                      "sklearn",
                      "boto3",
                      "monty",
                      "sqlalchemy",
                      "matminer",
                      "psycopg2",
                      "protosearch",
                      "autologging",
                      "aws-logging-handlers"],
    # TODO: make this materials?
    dependency_links=[
        "http://github.com/JosephMontoya-TRI/qmpy_py3/tarball/master#egg=qmpy",
        "git+ssh://git@github.awsinternal.tri.global/materials/bulk_enumerator#egg=bulk_enumerator",
        "git+ssh://git@github.awsinternal.tri.global/materials/protosearch#egg=protosearch"
    ]
)
