from setuptools import setup, find_packages, Extension

# To get the version
exec(open('mgr/version.py').read())
# To get the long description from the README file
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='mgr',
    version=__version__,
    author='Diego Bonato, Alberto Saretto, Luca Menti',
    description='Package for the Music Genre Recognition (MGR) project',
    packages=["mgr"],
    long_description=long_description,
    install_requires=["numpy","torch"]
)