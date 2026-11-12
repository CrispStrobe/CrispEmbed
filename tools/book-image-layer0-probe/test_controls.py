import hashlib
import importlib.util
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

    def test_positive_preparation_keeps_ordinary_source(self):
        root = Path(__file__).resolve().parents[2]
        driver = subprocess.check_output(['git', 'show', prepare.SOURCE + ':tests/lighton4b_native_document.cpp'], cwd=root)
        cmake = subprocess.check_output(['git', 'show', prepare.SOURCE + ':CMakeLists.txt'], cwd=root)
        with tempfile.TemporaryDirectory() as directory:
            task = Path(directory)
            (task / 'tests').mkdir()
            (task / 'tests/lighton4b_native_document.cpp').write_bytes(driver)
            (task / 'CMakeLists.txt').write_bytes(cmake)
            result = prepare.prepare(task)
            self.assertEqual((task / 'tests/lighton4b_native_document.cpp').read_bytes(), driver)
            self.assertIn('EXCLUDE_FROM_ALL', (task / 'CMakeLists.txt').read_text()[len(cmake):])
            self.assertEqual(result['target'], prepare.TARGET)
            self.assertEqual(hashlib.sha256((task / 'tools/book-image-layer0-probe/generated_driver.cpp').read_bytes()).hexdigest(), result['driver_sha256'])

    def test_worker_stdlib_import(self):
        spec = importlib.util.spec_from_file_location('image_worker', Path(__file__).with_name('worker.py'))
        worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(worker)
        self.assertEqual(worker.prepare.SOURCE, prepare.SOURCE)


if __name__ == '__main__':
    unittest.main()
