# General configuration
extensions = ["sphinx.ext.autodoc", "sphinx.ext.autosectionlabel"]
autoapi_dirs = ["../"]
# templates_path = ['_templates']
source_suffix = ".rst"
master_doc = "index"
project = "OneMod"
copyright = "2022-, University of Washington"
author = "Mathematical Sciences and Computational Algorithms"
language = "en"
todo_include_todos = True
autoclass_content = "init"
pygments_style = "sphinx"

# HTML output options
html_theme = "furo"


# Additional configurations
intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}
graphviz_output_format = "svg"
suppress_warnings = ["autosectionlabel.*"]
