# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
# -- Path setup --------------------------------------------------------------
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import re
import sys
from datetime import datetime

import rulekit

sys.path.insert(0, os.path.abspath('../..'))

current_path = os.path.dirname(os.path.realpath(__file__))


# -- Project information -----------------------------------------------------

project = f'rulekit v{rulekit.__VERSION__}'
copyright = f'{datetime.now().year}, ADAA'

release = open(f'{current_path}/../../rulekit/main.py',
               mode='r').read().strip()
version_regex = r'__VERSION__\s*=\s*\'{0,1}"{0,1}([^\'"]+)\'{0,1}"{0,1}'
release = re.search(version_regex, release)[1]
master_doc = 'index'

source_suffix = [".rst", ".md"]

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx_rtd_theme',
    'sphinx.ext.autodoc',
    'sphinx.ext.coverage',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'nbsphinx',
    'sphinx_copybutton'
]
autoclass_content = 'both'
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']
nbsphinx_allow_errors = True
