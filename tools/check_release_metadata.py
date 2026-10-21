"""Check package versions without importing runtime dependencies."""

import argparse
import ast
import os
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


def check_versions(root: Path, tag: str = "", *, metrics: bool = False) -> str:
    """Check main or standalone metrics versions without importing the package.

    Metrics checks use ``packages/metrics`` and require ``metrics-v<version>`` tags.
    An empty tag checks the manifest and module versions only.
    """
    if metrics:
        root = root / "packages/metrics"
    manifest = tomllib.loads((root / "pyproject.toml").read_text())
    version = manifest["project"]["version"]
    if not isinstance(version, str):
        raise ValueError("Manifest version must be a string")
    module_name = "cellseg_metrics" if metrics else "cellseg_models_pytorch"
    module = ast.parse((root / module_name / "__init__.py").read_text())
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
    # Metrics releases have their own tag prefix and version sequence.
    valid_tags = {f"metrics-v{version}"} if metrics else {version, f"v{version}"}
    if tag and tag not in valid_tags:
        raise ValueError(f"Release tag {tag} != package version {version}")
    return version


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics", action="store_true", help="Check standalone metrics"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    print(check_versions(root, os.environ.get("RELEASE_TAG", ""), metrics=args.metrics))
