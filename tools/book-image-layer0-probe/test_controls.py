import hashlib
from pathlib import Path
import subprocess
import tempfile
import unittest
import prepare


class Controls(unittest.TestCase):
    def test_reject_drift_before_mutation(self):
        with self.assertRaisesRegex(ValueError, 'driver drift'):
            prepare.generate(b'wrong')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'CMakeLists.txt'
            path.write_bytes(b'wrong')
            with self.assertRaisesRegex(ValueError, 'CMake drift'):
                prepare.prepare(directory)
            self.assertEqual(path.read_bytes(), b'wrong')

    def test_exact_historical_driver_generation(self):
        root = Path(__file__).resolve().parents[2]
        original = subprocess.check_output(['git', 'show', prepare.SOURCE + ':tests/lighton4b_native_document.cpp'], cwd=root)
        generated = prepare.generate(original)
        self.assertEqual(hashlib.sha256(original).hexdigest(), prepare.DRIVER_SHA)
        self.assertIn('core_util::clean_exit(run(argc, argv))', generated)
        # Removing the two additions reproduces the exact ordinary executable source.
        self.assertEqual(generated.replace(prepare.PROBE + '\n', '').replace(prepare.STOP, '').encode(), original)
        self.assertEqual(generated.count('BOOK_IMAGE_LAYER0_CAPTURE PASS'), 1)


if __name__ == '__main__':
    unittest.main()
