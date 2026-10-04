"""Small artifact inventory rejection controls, executed on hosted runners."""
import contextlib
import io
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest import mock

from runtime_artifact_manifest import manifest


@contextlib.contextmanager
def workspace():
    with tempfile.TemporaryDirectory() as directory:
        previous = os.getcwd()
        os.chdir(directory)
        try:
            Path('src').mkdir()
            Path('src/posformer_ocr.cpp').write_text('// production only\n')
            Path('VERSION').write_text('0.17.12\n')
            Path('artifacts').mkdir()
            with mock.patch('runtime_artifact_manifest.subprocess.check_output',
                            side_effect=['a' * 40 + '\n', '160000 commit ' + 'b' * 40 + '\tggml\n']):
                yield
        finally:
            os.chdir(previous)


def archive(file_name='libcrispembed.so.0.17.12', link_target='libcrispembed.so.0.17.12'):
    with tarfile.open('artifacts/runtime.tar.gz', 'w:gz') as output:
        regular = tarfile.TarInfo(file_name)
        regular.size = 3
        output.addfile(regular, io.BytesIO(b'ELF'))
        link = tarfile.TarInfo('libcrispembed.so')
        link.type = tarfile.SYMTYPE
        link.linkname = link_target
        output.addfile(link)


class ArtifactManifestTest(unittest.TestCase):
    def test_actual_file_hash_and_loader_symlink_recorded(self):
        with workspace():
            archive()
            result = manifest('artifacts')
            self.assertEqual(result['version'], '0.17.12')
            row = result['files'][0]
            self.assertEqual(len(row['sha256']), 64)
            self.assertIn('libcrispembed.so', row['members'])
            self.assertEqual(row['links'], [{'path': 'libcrispembed.so',
                             'target': 'libcrispembed.so.0.17.12', 'kind': 'symlink'}])

    def test_missing_artifacts_and_diagnostic_source_rejected(self):
        with workspace(), self.assertRaisesRegex(ValueError, 'Empty'):
            manifest('artifacts')
        with workspace():
            archive()
            Path('src/posformer_ocr.cpp').write_text('int crispembed_posformer_pool_ceil_probe();\n')
            with self.assertRaisesRegex(ValueError, 'diagnostic'):
                manifest('artifacts')

    def test_bundled_weights_or_weight_symlink_rejected(self):
        for file_name, target in [('candidate.gguf', 'libcrispembed.so.0.17.12'),
                                  ('libcrispembed.so.0.17.12', 'candidate.gguf')]:
            with self.subTest(file_name=file_name, target=target), workspace():
                archive(file_name, target)
                with self.assertRaisesRegex(ValueError, 'Model weights'):
                    manifest('artifacts')


if __name__ == '__main__':
    unittest.main()
