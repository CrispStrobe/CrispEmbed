"""Small transport controls; no model, provider or CI execution."""
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import io

spec = importlib.util.spec_from_file_location('publish', Path(__file__).resolve().parents[1] / 'tools/ci-heavy/spacy_w7_publish.py')
publish = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publish)


class Response(io.BytesIO):
    status = 200
    headers = {'X-Repo-Commit': 'a' * 40}


class PublicationControls(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'fresh'
        self.data = b'synthetic transport input, not a model'
        self.sha = hashlib.sha256(self.data).hexdigest()

    def test_complete_download_hash_and_size(self):
        with patch.object(publish.urllib.request, 'urlopen', return_value=Response(self.data)):
            result = publish.download('https://example.invalid/immutable', self.path, self.sha, len(self.data))
        self.assertEqual(self.path.read_bytes(), self.data)
        self.assertEqual(result['sha256'], self.sha)

    def test_different_bytes_or_length_do_not_pass(self):
        for expected_sha, expected_size in [('0' * 64, len(self.data)), (self.sha, 1)]:
            with self.subTest(expected_sha=expected_sha), patch.object(publish.urllib.request, 'urlopen', return_value=Response(self.data)):
                with self.assertRaises(AssertionError):
                    publish.download('https://example.invalid/immutable', self.path, expected_sha, expected_size)
                self.path.unlink()

    def test_existing_cache_file_cannot_be_a_fresh_download(self):
        self.path.write_bytes(self.data)
        with patch.object(publish.urllib.request, 'urlopen', return_value=Response(self.data)), self.assertRaises(FileExistsError):
            publish.download('https://example.invalid/immutable', self.path, self.sha, len(self.data))


if __name__ == '__main__':
    unittest.main()
