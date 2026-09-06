"""Integrity failures must stop verification, including path escapes."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from verify_archive import verify


class ArchiveIntegrityTests(unittest.TestCase):
    def test_valid_missing_and_modified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / 'data.csv'
            data.write_bytes(b'original')
            (root / 'ARCHIVE_MANIFEST.json').write_text(json.dumps({'files': {'data.csv': hashlib.sha256(b'original').hexdigest()}}))
            self.assertEqual(verify(root), 1)
            data.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                verify(root)
            data.unlink()
            with self.assertRaises(FileNotFoundError):
                verify(root)

    def test_path_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'ARCHIVE_MANIFEST.json').write_text(json.dumps({'files': {'../outside': 'unused'}}))
            with self.assertRaises(ValueError):
                verify(root)


if __name__ == '__main__':
    unittest.main()
