"""Small byte-identity controls; no rendering, Pillow, corpus or model download."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('successor_posformer_corpus', ROOT / 'tools/ci-heavy/successor_posformer_corpus.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class CorpusIdentity(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.corpus = Path(self.tmp.name)
        self.cases = []
        for i in range(50):
            name = f'case{i:02}'
            data = b'encoded-image-fixture:' + name.encode()
            (self.corpus / (name + '.png')).write_bytes(data)
            self.cases.append({'id': name, 'image_sha256': hashlib.sha256(data).hexdigest()})
        manifest = json.dumps({'cases': [{'id': c['id']} for c in self.cases]}).encode()
        (self.corpus / 'manifest.json').write_bytes(manifest)
        self.manifest_sha = hashlib.sha256(manifest).hexdigest()

    def verify(self):
        return audit.verify(self.corpus, self.cases, self.manifest_sha)

    def test_all_fifty_encoded_bytes(self):
        rows = self.verify()
        self.assertEqual([r['id'] for r in rows], [r['id'] for r in self.cases])
        self.assertEqual(len(rows), 50)

    def test_one_changed_image_rejected_with_identity(self):
        (self.corpus / 'case49.png').write_bytes(b'changed encoding')
        with self.assertRaisesRegex(ValueError, 'case49 expected=.* actual='):
            self.verify()

    def test_missing_image_rejected(self):
        (self.corpus / 'case13.png').unlink()
        with self.assertRaisesRegex(ValueError, 'member coverage'):
            self.verify()

    def test_extra_input_rejected(self):
        (self.corpus / 'unexpected.png').write_bytes(b'extra')
        with self.assertRaisesRegex(ValueError, 'member coverage'):
            self.verify()

    def test_manifest_byte_change_rejected(self):
        with (self.corpus / 'manifest.json').open('ab') as stream:
            stream.write(b'\n')
        with self.assertRaisesRegex(ValueError, 'manifest bytes'):
            self.verify()

    def test_manifest_order_rejected_even_with_matching_digest(self):
        data = json.dumps({'cases': [{'id': c['id']} for c in reversed(self.cases)]}).encode()
        (self.corpus / 'manifest.json').write_bytes(data)
        with self.assertRaisesRegex(ValueError, 'case order'):
            audit.verify(self.corpus, self.cases, hashlib.sha256(data).hexdigest())

    def test_frozen_reference_has_fifty_distinct_inputs(self):
        rows = audit.frozen_cases()
        self.assertEqual(len(rows), 50)
        self.assertEqual(rows[0]['image_sha256'], '963e427aec65cee40c231dabecd938283a81b24b9880d72af67ddaee8f84a6ef')


if __name__ == '__main__':
    unittest.main()
