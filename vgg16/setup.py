from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['Keras>=2.0.7', 'h5py>=2.7.0', 'Pillow==4.2.1']

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='My trainer application package.'
)
