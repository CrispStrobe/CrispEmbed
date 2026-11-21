#!/usr/bin/env python3
"""Collection hygiene for the plain-environment pytest lane.

A test_*.py file that reads required CLI arguments while it is imported
(``argparse.parse_args()`` or ``sys.argv[i]`` at module or class level) breaks
``pytest tests``: pytest's own argv is parsed instead, so the import either
aborts the whole session or binds a wrong value (a path under ``tests/``).
Such a file must read its arguments under ``if __name__ == "__main__":`` (or
inside a function), or be listed in ``collect_ignore`` in tests/conftest.py.
Optional flags tested with ``'--flag' in sys.argv`` are fine.
"""
import ast
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _is_sys_argv(node):
    return (isinstance(node, ast.Attribute) and node.attr == 'argv'
            and isinstance(node.value, ast.Name) and node.value.id == 'sys')


def _guarded(node):
    """`if __name__ == ...` blocks and `if '--x' in sys.argv` blocks run only when asked."""
    if not isinstance(node, (ast.If, ast.IfExp)):
        return False
    test = node.test
    if isinstance(test, ast.Compare) and isinstance(test.left, ast.Name) and test.left.id == '__name__':
        return True
    return any(_is_sys_argv(n) for n in ast.walk(test))


def import_time_argv_reads(tree):
    def scan(body):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or _guarded(node):
                continue
            if isinstance(node, ast.ClassDef):
                yield from scan(node.body)
                continue
            stack = [node]
            while stack:
                sub = stack.pop()
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)) and sub is not node:
                    continue
                if isinstance(sub, ast.IfExp) and _guarded(sub):
                    stack.append(sub.orelse)  # `argv[1] if __name__ == '__main__' else None`
                    continue
                if isinstance(sub, ast.Subscript) and _is_sys_argv(sub.value):
                    yield node.lineno, 'sys.argv[...]'
                if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
                        and sub.func.attr in ('parse_args', 'parse_known_args')):
                    yield node.lineno, sub.func.attr + '()'
                stack.extend(ast.iter_child_nodes(sub))
    return sorted(set(scan(tree.body)))


class CollectionHygiene(unittest.TestCase):
    def test_detector_controls(self):
        bad = ast.parse("import sys, argparse\nA = argparse.ArgumentParser().parse_args()\n"
                        "class C:\n    root = sys.argv[1]\n    other = sys.argv[1] if __name__ == '__main__' else sys.argv[2]\n")
        self.assertEqual([what for _, what in import_time_argv_reads(bad)], ['parse_args()', 'sys.argv[...]', 'sys.argv[...]'])
        good = ast.parse("import sys\nX = '--x' in sys.argv\nif '--x' in sys.argv:\n    sys.argv.remove('--x')\n"
                         "def main():\n    return sys.argv[1]\nif __name__ == '__main__':\n    print(sys.argv[1])\n")
        self.assertEqual(import_time_argv_reads(good), [])

    def test_no_collected_file_reads_cli_arguments_at_import(self):
        sys.path.insert(0, str(ROOT / 'tests'))
        try:
            import conftest
            ignored = set(conftest.collect_ignore)
        except ImportError:  # run as a plain script without pytest installed
            ignored = {'test_scan_rule_geometry.py'}
        finally:
            sys.path.pop(0)
        offenders = {}
        for path in sorted((ROOT / 'tests').glob('test_*.py')):
            if path.name in ignored:
                continue
            hits = import_time_argv_reads(ast.parse(path.read_text(encoding='utf-8')))
            if hits:
                offenders[path.name] = hits
        self.assertEqual(offenders, {}, 'guard these with `if __name__ == "__main__":` or add to collect_ignore')

    def test_ignored_files_still_exist(self):
        sys.path.insert(0, str(ROOT / 'tests'))
        try:
            import conftest
        except ImportError:
            self.skipTest('requires pytest (tests/conftest.py)')
        finally:
            sys.path.pop(0)
        for name in conftest.collect_ignore:
            self.assertTrue((ROOT / 'tests' / name).is_file(), name)


if __name__ == '__main__':
    unittest.main()
