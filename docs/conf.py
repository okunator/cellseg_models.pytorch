"""Sphinx configuration for the user documentation."""

project = "cellseg-models.pytorch"
author = "Oskari Lehtonen"
extensions = ["myst_nb", "sphinx.ext.napoleon"]
html_theme = "pydata_sphinx_theme"
html_title = "cellseg-models.pytorch"
html_theme_options = {
    "github_url": "https://github.com/okunator/cellseg_models.pytorch",
}
exclude_patterns = [
    "_build",
    "**/.ipynb_checkpoints",
    "maintenance-plan.md",
    "module-boundaries-draft.md",
    "releasing.md",
]
nb_execution_mode = "off"
# Real-data/checkpoint notebooks are executed explicitly, outside offline CI.
nb_execution_excludepatterns = [
    "tutorials/prepare-data.ipynb",
    "tutorials/training.ipynb",
    "tutorials/inference.ipynb",
    "tutorials/wsi.ipynb",
]
nb_execution_raise_on_error = True
nb_execution_timeout = 120
napoleon_google_docstring = True
napoleon_numpy_docstring = False
