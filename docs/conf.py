# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# in the docs/ folder, enter 
# sphinx-apidoc --ext-autodoc -o . .. and then
# make html

import os
import sys
sys.path.insert(0, os.path.abspath('..'))
import sphinx_theme_pd

project = 'PyLearn'
copyright = '2023, Jan Skowron'
author = 'Jan Skowron'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode'
    ]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_theme_pd'
html_theme_path = [sphinx_theme_pd.get_html_theme_path()]
html_static_path = ['_static']