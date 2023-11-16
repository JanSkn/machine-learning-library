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

- TODO
  
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

- Please adhere to the [Code of Conduct](.github/CODE_OF_CONDUCT.md) in all interactions within the project.

Thank you for your participation!
