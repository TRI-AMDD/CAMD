from setuptools import setup, find_packages

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
    install_requires=["numpy==1.15",
                      "networkx==2.2",
                      "matplotlib==2.2",
                      "MySQL-python",
                      "qmpy",
                      "tqdm",
                      "pandas",
                      "sklearn",
                      "boto3"]
)
