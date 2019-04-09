from setuptools import setup, find_packages

setup(
    name='CAMD',
    version='1.0',
    packages=find_packages(),
    install_requires=['python-dateutil',
                      'pytz',
                      'numpy',
                      'scikit-learn',
                      'pandas',
                      'ConfigParser',
                      'qmpy',
                      'django',
                      'pyyaml',
                      'matplotlib'],
    url='',
    license='',
    author='muratahan.aykol',
    author_email='murat.aykol@tri.global',
    description=''
)
