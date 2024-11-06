from setuptools import setup, find_packages

"""
PyLearn
=======

PyLearn is a Python machine learning library that provides implementations of various machine learning algorithms, 
including neural networks, classification, and clustering.

Usage
------------

Import PyLearn:

.. code-block:: python

   import pylearn as pl

Documentation
-------------

The official documentation can be found at: https://pylearn-ml.readthedocs.io/en/latest/

License
-------

This project is licensed under the MIT License - see the LICENSE file for details.
"""


# --- version ---
# MAJOR.MINOR.PATCH
# MAJOR for backwards incompatible changes.
# MINOR for backwards-compatible new functions.
# PATCH for backwards-compatible error corrections.

with open("README.md", "r") as f:
    long_description = f.read()

with open("docs/requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="pylearn-ml",
    version="1.2.0",
    author="Jan Skowron",
    url="https://github.com/JanSkn/machine-learning-library", 
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),   
    install_requires=requirements
)

# to upload/update version:
# twine upload dist/*
