# Configuration file for the Sphinx documentation builder.

import os
import sys
from pathlib import Path

# -- Project information -----------------------------------------------------
project = 'Depression Analysis Classifier'
copyright = '2024, Bianchetti Federico'
author = 'Bianchetti Federico'
release = 'Alpha Version'

# -- General configuration ---------------------------------------------------


sys.path.insert(0, os.path.abspath('/mnt/c/Users/Federico/Documents/Progetti/depression-pre-diagnose-model/modules'))


extensions = [
    'sphinx.ext.autosummary',
    'sphinx.ext.autodoc',  
    'sphinx.ext.napoleon'
]


templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

html_theme = 'press'
html_static_path = ['_static']
