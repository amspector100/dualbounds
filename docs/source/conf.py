# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'dualbounds'
copyright = '2024, Asher Spector'
author = 'Asher Spector'

### Autoversioning
# Import the right package!
import sys
import os
import sphinx_rtd_theme
sys.path.insert(0, os.path.abspath('../../'))

# The full version, including alpha/beta/rc tags
import dualbounds
release = dualbounds.__version__
version = release

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx_automodapi.automodapi',
    'numpydoc',
    'nbsphinx',
    'sphinx.ext.doctest',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    #'sphinx_multiversion',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', '**.ipynb_checkpoints']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

## Change to Furo, sphinx_rtd_theme, or pydata_sphinx_theme?
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'display_version':True,
}
html_static_path = ['_static']
