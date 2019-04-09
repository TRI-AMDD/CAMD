from setuptools import setup, find_packages
import warnings

try:
    import numpy
except ImportError:
    warnings.warn("numpy must be installed prior to building, run pip install"
                  "numpy before installation")

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
    install_requires=["numpy",
                      "networkx==2.2",
                      "matplotlib==2.2",
                      "MySQL-python",
                      "qmpy",
                      "tqdm",
                      "pandas",
                      "sklearn",
                      "boto3"]
)
