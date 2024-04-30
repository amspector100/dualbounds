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
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('../../'))


# The full version, including alpha/beta/rc tags
import dualbounds
# for autodoc
release = dualbounds.__version__
version = release

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    #'sphinx_automodapi.automodapi',
    #'numpydoc',
    'sphinx.ext.napoleon',
    'nbsphinx',
    'sphinx.ext.doctest',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx_immaterial',
    # this conflicts with nbsphinx
    #"sphinx_immaterial.apidoc.python.apigen",
    #'sphinx_multiversion',
]

# autosummary generates stub articles for each class
autosummary_generate = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', '**.ipynb_checkpoints']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Theme
html_theme = 'sphinx_immaterial'
# Sphinx Immaterial theme option
shtml_theme_options = {
    # "icon": {
    #     "repo": "fontawesome/brands/github",
    # },
    # "site_url": "https://galois.readthedocs.io/",
    "repo_url": "https://github.com/amspector100/dualbounds",
    "repo_name": "dualbounds",
    # "social": [
    #     {
    #         "icon": "fontawesome/brands/github",
    #         "link": "https://github.com/mhostetter/galois",
    #     },
    #     {
    #         "icon": "fontawesome/brands/python",
    #         "link": "https://pypi.org/project/galois/",
    #     },
    #     {
    #         "icon": "fontawesome/brands/twitter",
    #         "link": "https://twitter.com/galois_py",
    #     },
    # ],
    "edit_uri": "",
    "globaltoc_collapse": True,
    "features": [
        # "navigation.expand",
        "navigation.tabs",
        "navigation.tabs.sticky",
        # "toc.integrate",
        # "navigation.sections",
        # "navigation.instant",
        # "header.autohide",
        "navigation.top",
        "navigation.tracking",
        "navigation.prune",
        "toc.follow",
        ## This is an important one to think about
        #"toc.integrate",
        # "toc.sticky",
        "content.tabs.link",
        "announce.dismiss",
    ],
    "palette": { "primary": "blue" }
    # "palette": [
    #     {
    #         "media": "(prefers-color-scheme: light)",
    #         "scheme": "default",
    #         "toggle": {
    #             "icon": "material/weather-night",
    #             "name": "Switch to dark mode",
    #         },
    #     },
    #     {
    #         "media": "(prefers-color-scheme: dark)",
    #         "scheme": "slate",
    #         "toggle": {
    #             "icon": "material/weather-sunny",
    #             "name": "Switch to light mode",
    #         },
    #     },
    # ],
    # "version_dropdown": True,
    # "version_json": "../versions.json",
}

html_last_updated_fmt = ""
html_use_index = True
html_domain_indices = True
html_logo = "images/smalllogo.svg"
html_favicon = "images/favicon.svg"

python_module_names_to_strip_from_xrefs = ["dualbounds"]
html_static_path = ['_static']
