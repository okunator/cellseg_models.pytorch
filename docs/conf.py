"""Sphinx configuration for the user documentation."""

from cellseg_models_pytorch import __version__

project = "cellseg-models.pytorch"
release = __version__
author = "Oskari Lehtonen"
extensions = ["myst_nb", "sphinx.ext.autodoc", "sphinx.ext.napoleon"]
html_theme = "pydata_sphinx_theme"
html_title = f"{project} {release}"
html_logo = "_static/logo.png"
html_static_path = ["_static"]
html_css_files = ["landing.css"]
html_sidebars = {"quickstart": []}
html_theme_options = {
    "github_url": "https://github.com/okunator/cellseg_models.pytorch",
    "logo": {"alt_text": "cellseg-models.pytorch documentation home"},
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
    "tutorials/evaluation.ipynb",
]
nb_execution_raise_on_error = True
nb_execution_timeout = 120
napoleon_google_docstring = True
napoleon_numpy_docstring = False
autoclass_content = "both"
autodoc_typehints = "description"
add_module_names = False
