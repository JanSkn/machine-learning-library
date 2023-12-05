from setuptools import setup, find_packages

"""
pylearn
=======

pylearn is a Python machine learning library that provides implementations of various machine learning algorithms, 
including neural networks, classification, and clustering.

Installation
------------

You can (hopefully soon) install pylearn using pip:

.. code-block:: bash

   pip install pylearn

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

with open('docs/requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="pylearn",
    version="1.0.0",      
    author="Jan Skowron",
    url="https://github.com/JanSkn/machine-learning-library",       
    packages=find_packages(),   
    install_requires=requirements
)

# to upload/update version:
# twine upload dist/*
