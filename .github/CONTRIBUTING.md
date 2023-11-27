# Contributing to PyLearn

Thank you for considering to contribute! Here are some guidelines to help you contribute to the project.

## Issue Reporting

- If you discover a bug, please create an issue to report it.
- Please describe the bug in detail, including steps to reproduce.

## Pull Requests

- Before creating a pull request, open an issue to discuss your proposed changes.
- Follow the provided template to ensure your pull request includes all necessary information.
- Ensure your code adheres to the project's coding standards.

## Documentation

**Updating the documentation (in `docs`):**

The documentation is partly generated automatically with Sphinx.
Everything under `/custom` is manual documentation.

To update it:

1. Navigate to the `docs` folder
2. Enter
   ```sh
   sphinx-apidoc --ext-autodoc -o . ..
   ```
3. To see the changes locally under Mac/Linux:
   ```sh
   make html
   ```
   or on Windows:
   ```sh
   make.bat html
   ```
   and then open `_build/index.html`
4. If you made changes on an existing file, delete the corresponding .rst file and redo (2.)
5. Add the file to `pylearn.rst`
6. Add explaining text and/or examples to the documentation under `custom/usage.rst`.
7. Commit the changes

**Type hints:**

Please use type hints for parameters and return values.

**Docstrings:**

Please follow this schema for variables in docstrings:

name (data type): Description
<br />
*if optional variable:*
<br />
name (data type, optional): Description, default: the default value

Docstring for a class:
- Description
- Attributes

Example:

```python
"""
This is a class description.

Attributes:
  :x (int): This is a variable
  :y (float, optional): This is another variable, default: 1.2
"""
```

Docstring for a function:
- Description
- Parameters
- Returns

Example:

```python
"""
This is a function description.

Parameters:
  :x (int): This is a variable
  :y (float, optional): This is another variable, default: 1.2

Returns:
  Returns the sum of x and y as a float
"""
```

## Code of Conduct

- Please adhere to the [Code of Conduct](CODE_OF_CONDUCT.md) in all interactions within the project.

Thank you for your participation!
