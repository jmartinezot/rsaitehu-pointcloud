# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../../rsaitehu_pointcloud'))

project = 'rsaitehu_pointcloud'
copyright = '2024, José María Martínez-Otzeta'
author = 'José María Martínez-Otzeta'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # For Google or NumPy docstrings
    "sphinx_rtd_theme",
    #'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    #'sphinx.ext.todo', 'sphinx.ext.viewcode', 'sphinx.ext.autodoc', 'sphinx.ext.imgmath', 'sphinx_automodapi.automodapi', 'sphinx.ext.ifconfig'
]


templates_path = ['_templates']
exclude_patterns = []

add_module_names = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_theme = 'furo'
# html_theme = 'alabaster'
# html_theme = 'pydata_sphinx_theme'

html_static_path = ['_static']

# Display function signatures with docstrings
autosummary_generate = True
autodoc_docstring_signature = True
autodoc_member_order = "bysource"
autodoc_inherit_docstrings = True

