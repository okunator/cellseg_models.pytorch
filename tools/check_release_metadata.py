"""Check package versions without importing runtime dependencies."""

import ast
import os
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


def check_versions(root: Path, tag: str = "") -> str:
    """Require matching manifest, module, and optional release tag versions."""
    manifest = tomllib.loads((root / "pyproject.toml").read_text())
    version = manifest["tool"]["poetry"]["version"]
    module = ast.parse((root / "cellseg_models_pytorch/__init__.py").read_text())
    module_version = None
    for statement in module.body:
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in statement.targets
        ):
            module_version = ast.literal_eval(statement.value)
    if module_version != version:
        raise ValueError(
            f"Manifest version {version} != module version {module_version}"
        )
    if tag and tag.removeprefix("v") != version:
        raise ValueError(f"Release tag {tag} != package version {version}")
    return version


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    print(check_versions(root, os.environ.get("RELEASE_TAG", "")))
