"""Model-free registry coverage and preserved-artifact identity controls."""
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import fetch_model_hashes as pins


class Registry(unittest.TestCase):
    def test_adjacent_immutable_urls_keep_hash_and_size_coverage(self):
        text = '''static const ModelEntry k_registry[] = {
 { "first", "a.gguf", "https://huggingface.co/owner/model/resolve/"
 "0123456789012345678901234567890123456789/a.gguf", "description", "15 MB", "apache-2.0", "card" },
 { "second", "b.gguf", "https://huggingface.co/owner/model/resolve/main/b.gguf", "description", "2 GB", "mit", "card" },
};'''
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'registry.cpp'; path.write_text(text)
            urls = pins.registry_urls(path); sizes = pins.registry_sizes(path)
        self.assertEqual(len(urls), 2)
        self.assertEqual(set(urls), set(sizes))
        self.assertEqual(sorted(row[1] for row in sizes.values()), [15e6, 2e9])

    def test_actual_registry_is_fully_pinned(self):
        urls = pins.registry_urls(pins.REGISTRY_CPP)
        header = pins.HEADER_OUT.read_text()
        current = dict(re.findall(r'\{ "(https://[^"\n]+)",\s*"([0-9a-f]{64})" \}', header))
        self.assertEqual(set(current), set(urls))
        self.assertEqual(header, pins.render_header({url: current[url] for url in urls}, []))
        discovery = json.loads((ROOT / 'tests/results/registry-pin-audit/discovery.json').read_text())
        sizes = pins.registry_sizes(pins.REGISTRY_CPP)
        for row in discovery['added'] + discovery['changed']:
            parts = pins.URL_RE.fullmatch(row['url'])
            revision = row.get('trusted_revision', row['current_revision'])
            url = row['url'].replace('/resolve/' + parts['rev'] + '/', '/resolve/' + revision + '/')
            self.assertEqual(current[url], row['old_sha256'] or row['current_sha256'])
            # Secondary mmproj size is outside this primary-entry parser's scope.
            if not row['file'].startswith('mmproj-'): self.assertIn(url, sizes)


if __name__ == '__main__': unittest.main()
