<a name="readme-top"></a>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/janskn/machine-learning-library">
    <img src="https://drive.google.com/uc?export=view&id=1pDgbi7uZqIGRK4wllBcQRvJNYDTvrIe5" alt="Logo" width="150" height="150">
  </a>

  <h3 align="center">PyLearn</h3>

  <p align="center">
    A simple library for machine learning topics.
    <br />
    <a href="https://pylearn-ml.readthedocs.io/en/latest/"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/janskn/machine-learning-library/issues">Report Bug</a>
    ·
    <a href="https://github.com/janskn/machine-learning-library/issues">Request Feature</a>
  </p>

  <br />

  [![PyPI version](https://badge.fury.io/py/pylearn-ml.svg)](https://badge.fury.io/py/pylearn-ml)
  ![PyPI - Downloads](https://img.shields.io/pypi/dm/pylearn-ml?label=PyPI%20downloads)
  ![Pepy Total Downloads](https://img.shields.io/pepy/dt/pylearn-ml?label=Total%20downloads)
  <br />
  [![Test](https://github.com/JanSkn/machine-learning-library/actions/workflows/tests.yml/badge.svg)](https://github.com/JanSkn/machine-learning-library/actions/workflows/tests.yml)
  [![Coverage](https://github.com/JanSkn/machine-learning-library/actions/workflows/coverage.yml/badge.svg)](https://github.com/JanSkn/machine-learning-library/actions/workflows/coverage.yml)
  [![Coverage Status](https://coveralls.io/repos/github/JanSkn/machine-learning-library/badge.svg?branch=main)](https://coveralls.io/github/JanSkn/machine-learning-library?branch=main)

<br />
</div>



<!-- ABOUT THE PROJECT -->
## About The Project

PyLearn implements machine learning features from scratch.
It supports basic features of supervised and unsupervised learning.
<br />
<br />
You can
- create neural networks with dense layers, different activation functions and loss functions,
- cluster your data without previously known classes,
- classify data
- evaluate models
- and more

<br />
    Read the <a href="https://pylearn-ml.readthedocs.io/en/latest/">Documentation</a> for more information.
    <br />
    <br />

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Installation

Install PyLearn using `pip`:
```sh
pip install pylearn-ml
````

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

The source code was built with Python, mainly using NumPy and Pandas.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Requirements

Requirements can be found under `docs/requirements.txt`.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

Import the library:
```python
import pylearn as pl
````

Most models have a _fit_ and a _predict_ function.

Just create a model, train it and use it for predictions.

```python
model = pl.Model()

model.fit(x_train, y_train)
...
model.predict(y_test)
```

For details of usage, have a look at the `examples` folder.
<br />
Or read the <a href="https://pylearn-ml.readthedocs.io/en/latest/">Documentation</a>



<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

**Please follow the [Contributing](.github/CONTRIBUTING.md) guidelines.**

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>
