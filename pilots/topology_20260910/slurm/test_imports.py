"""Allocated-CPU checks for import-bundle integrity and extraction boundaries."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from .imports import PACKAGES, prepare


class ImportBundleTests(unittest.TestCase):
    def fixture(self, directory, name="transformers/__init__.py", kind=None):
        root = Path(directory) / "root"
        bundle = root / "import_bundle"
        bundle.mkdir(parents=True)
        archive = bundle / "imports.tar"
        with tarfile.open(archive, "w") as output:
            entry = tarfile.TarInfo(name)
            data = b"exact-installed-source\n"
            if kind == "symlink":
                entry.type, entry.linkname = tarfile.SYMTYPE, "../../outside"
                output.addfile(entry)
            else:
                entry.size = len(data)
                output.addfile(entry, io.BytesIO(data))
        sha = hashlib.sha256(archive.read_bytes()).hexdigest()
        (bundle / "complete.json").write_text(json.dumps({"status": "complete", "sha256": sha, "packages": PACKAGES}))
        return root, sha

    def test_exact_source_and_import_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            root, sha = self.fixture(directory)
            env, record = prepare(root, Path(directory) / "scratch", sha)
            first = Path(env["PYTHONPATH"].split(":")[0])
            self.assertEqual((first / "transformers/__init__.py").read_bytes(), b"exact-installed-source\n")
            self.assertEqual(record["archive_sha256"], sha)

    def test_archive_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root, sha = self.fixture(directory)
            with (root / "import_bundle/imports.tar").open("ab") as output:
                output.write(b"changed")
            with self.assertRaisesRegex(RuntimeError, "checksum"):
                prepare(root, Path(directory) / "scratch", sha)

    def test_path_traversal_absolute_path_and_symlinks_are_rejected(self):
        for name, kind in [("transformers/../../outside", None), ("/outside", None),
                           ("transformers/link", "symlink"), ("unexpected/module.py", None)]:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root, sha = self.fixture(directory, name, kind)
                with self.assertRaisesRegex(RuntimeError, "Unexpected path or link"):
                    prepare(root, Path(directory) / "scratch", sha)
                self.assertFalse((Path(directory) / "outside").exists())

    def test_wrong_reviewed_identity_fails_before_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _ = self.fixture(directory)
            with self.assertRaisesRegex(RuntimeError, "reviewed runtime identity"):
                prepare(root, Path(directory) / "scratch", "0" * 64)


if __name__ == "__main__":
    unittest.main()
