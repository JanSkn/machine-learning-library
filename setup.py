from setuptools import setup, find_packages

# --- version ---
# MAJOR.MINOR.PATCH
# MAJOR for backwards incompatible changes.
# MINOR for backwards-compatible new functions.
# PATCH for backwards-compatible error corrections.

with open('requirements.txt') as f:
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
