"""Exercise release guards without importing the segmentation library."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "check_release_metadata", Path(__file__).parents[1] / "check_release_metadata.py"
)
release_metadata = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release_metadata)


class ReleaseMetadataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "cellseg_models_pytorch").mkdir()
        (self.root / "pyproject.toml").write_text('[tool.poetry]\nversion = "0.1.30"\n')
        self.module = self.root / "cellseg_models_pytorch/__init__.py"
        self.module.write_text('__version__ = "0.1.30"\n')

    def test_matching_versions_with_or_without_tag_prefix(self):
        for tag in ("", "0.1.30", "v0.1.30"):
            with self.subTest(tag=tag):
                self.assertEqual(
                    release_metadata.check_versions(self.root, tag), "0.1.30"
                )

    def test_rejects_mismatched_module_version(self):
        self.module.write_text('__version__ = "0.1.29"\n')
        with self.assertRaisesRegex(ValueError, "Manifest version"):
            release_metadata.check_versions(self.root)

    def test_rejects_mismatched_tag(self):
        with self.assertRaisesRegex(ValueError, "Release tag"):
            release_metadata.check_versions(self.root, "v0.1.29")

    def test_rejects_missing_module_version(self):
        self.module.write_text('"""No version declaration."""\n')
        with self.assertRaisesRegex(ValueError, "module version None"):
            release_metadata.check_versions(self.root)

    def test_does_not_import_module_to_check_version(self):
        self.module.write_text(
            'raise RuntimeError("Do not import me")\n__version__ = "0.1.30"\n'
        )
        self.assertEqual(release_metadata.check_versions(self.root), "0.1.30")


if __name__ == "__main__":
    unittest.main()
