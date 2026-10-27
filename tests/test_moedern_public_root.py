"""Stdlib controls for the root-wired MoEdern public text closure. No model.

The draft test's unwired-root assertion correctly rejects this source; here the
engine bytes and inference bodies stay pinned to the reviewed predecessor and the
root additions are explicit and default-off.
"""
import importlib.util
from pathlib import Path
import subprocess
import unittest
ROOT = Path(__file__).resolve().parents[1]
BASE = '1a053116908224cf1c9358d7d24c85ba36e82eff'
spec = importlib.util.spec_from_file_location('root_guard', ROOT/'tools/moedern-public/root_contract.py')
guard = importlib.util.module_from_spec(spec); spec.loader.exec_module(guard)
def show(rev, path): return subprocess.check_output(['git', 'show', rev+':'+path], cwd=ROOT)


class Controls(unittest.TestCase):
    def test_closure(self): guard.source()

    def test_engine_and_inference_body_unchanged_from_reviewed_predecessor(self):
        self.assertEqual((ROOT/'src/moedern_text_engine.cpp').read_bytes(), show(BASE, 'src/moedern_text_engine.cpp'))
        marker = 'std::string text_frontend::normalize('
        before = show(BASE, 'src/moedern_text_frontend.cpp').decode(); after = (ROOT/'src/moedern_text_frontend.cpp').read_text()
        self.assertEqual(before[before.index(marker):], after[after.index(marker):])

    def test_root_additions_exact_and_default_off(self):
        cmake = (ROOT/'CMakeLists.txt').read_text()
        self.assertIn('set_source_files_properties(${CRISPEMBED_MOEDERN_SOURCES} PROPERTIES COMPILE_OPTIONS -ffp-contract=off)', cmake)
        self.assertNotIn('CRISPEMBED_MOEDERN_COMPOSED_TESTING', cmake)
        self.assertIn('core_env::on("CRISPEMBED_MOEDERN_PUBLIC_TEXT")', (ROOT/'src/moedern_text_api.cpp').read_text())
        for path in ('src/crispembed.h', 'examples/cli/main.cpp', 'examples/server/server.cpp'):
            self.assertEqual((ROOT/path).read_bytes(), show(BASE, path), path)


if __name__ == '__main__': unittest.main()
