"""Model-free registry coverage and preserved-artifact identity controls."""
import io
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest import mock
import urllib.error

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


class TreePagination(unittest.TestCase):
    URL = pins.API.format(owner="example", repo="model", rev="main")
    NEXT = URL + "&cursor=second"

    @staticmethod
    def page(entries, link="", url=None):
        response = io.BytesIO(json.dumps(entries).encode())
        response.headers = {"Link": link}
        response.geturl = lambda: url or TreePagination.URL
        return response

    @staticmethod
    def entry(path):
        return {"type": "file", "path": path, "lfs": {"oid": "a" * 64, "size": 42}}

    def test_later_page_resolves_hash_and_size(self):
        with mock.patch.object(pins._TREE_OPENER, "open", side_effect=[
            self.page([self.entry("a.gguf"), {"type": "directory", "path": "dir"}],
                      f'<{self.NEXT}>; rel="next"'),
            self.page([self.entry("b.gguf"), {"type": "file", "path": "plain", "oid": "b" * 40}]),
        ]) as get:
            tree = pins.fetch_tree("example", "model", "main")
        self.assertEqual(tree, {name: ("a" * 64, 42) for name in ["a.gguf", "b.gguf"]})
        self.assertEqual([call.args[0].full_url for call in get.call_args_list], [self.URL, self.NEXT])

    def test_canonicalized_repository_uses_response_endpoint(self):
        canonical = self.URL.replace("/example/model/", "/canonical/model/")
        continuation = canonical + "&cursor=second"
        with mock.patch.object(pins._TREE_OPENER, "open", side_effect=[
            self.page([self.entry("a.gguf")], f'<{continuation}>; rel="next"', canonical),
            self.page([self.entry("b.gguf")], url=continuation),
        ]) as get:
            tree = pins.fetch_tree("example", "model", "main")
        self.assertEqual(set(tree), {"a.gguf", "b.gguf"})
        self.assertEqual(get.call_args_list[1].args[0].full_url, continuation)

    def test_later_page_retry_does_not_repeat_first_page(self):
        error = urllib.error.HTTPError(self.NEXT, 429, "rate limited", {"Retry-After": "1"}, None)
        with mock.patch.object(pins._TREE_OPENER, "open", side_effect=[
            self.page([], f'<{self.NEXT}>; rel="next"'), error, self.page([self.entry("b.gguf")]),
        ]) as get, mock.patch.object(pins.time, "sleep") as sleep:
            self.assertIn("b.gguf", pins.fetch_tree("example", "model", "main"))
        self.assertEqual([call.args[0].full_url for call in get.call_args_list],
                         [self.URL, self.NEXT, self.NEXT])
        sleep.assert_called_once_with(1)

    def test_invalid_or_unsafe_links_never_receive_credentials(self):
        links = [
            '<http://huggingface.co/api/models/example/model/tree/main>; rel="next"',
            '<https://other.example/tree>; rel="next"',
            '<https://huggingface.co@other.example/tree>; rel="next"',
            '<https://huggingface.co:443/api/models/example/model/tree/main>; rel="next"',
            '<https://huggingface.co/api/models/other/model/tree/main>; rel="next"',
            f'<{self.NEXT}#fragment>; rel="next"',
            f'<{self.NEXT}>; rel="next", <{self.NEXT}>; rel="next"',
            f'<{self.NEXT}>; rel="next"; rel="prev"',
            'malformed; rel="next"',
            f'<{self.NEXT}>',
        ]
        for link in links:
            with self.subTest(link=link), mock.patch.dict(pins.os.environ, {"HF_TOKEN": "synthetic-test-token"}), \
                    mock.patch.object(pins._TREE_OPENER, "open", return_value=self.page([], link)) as get:
                with self.assertRaises(pins.TreePaginationError):
                    pins.fetch_tree("example", "model", "main")
                self.assertEqual(get.call_count, 1)

    def test_redirect_credentials_cannot_escape_hub_https(self):
        handler = pins._TreeRedirectHandler()
        request = pins.urllib.request.Request(self.URL, headers={"Authorization": "Bearer synthetic-test-token"})
        for target in ["https://other.example/", "http://huggingface.co/", "https://huggingface.co@other.example/"]:
            with self.subTest(target=target), self.assertRaises(pins.TreePaginationError):
                handler.redirect_request(request, None, 302, "Found", {}, target)
        redirected = handler.redirect_request(request, None, 302, "Found", {}, self.NEXT)
        self.assertEqual(redirected.full_url, self.NEXT)
        self.assertEqual(redirected.get_header("Authorization"), request.get_header("Authorization"))

    def test_cycle_fails_without_refetch(self):
        with mock.patch.object(pins._TREE_OPENER, "open", return_value=
                               self.page([], f'<{self.URL}>; rel="next"')) as get:
            with self.assertRaises(pins.TreePaginationError):
                pins.fetch_tree("example", "model", "main")
        self.assertEqual(get.call_count, 1)

    def test_relative_link_and_other_relations(self):
        link = f'<{self.URL}>; rel="prev", <?recursive=1&cursor=second>; rel="next"'
        self.assertEqual(pins._next_tree_page(link, self.URL), self.NEXT)
        self.assertIsNone(pins._next_tree_page(f'<{self.URL}>; rel="prev"', self.URL))

    def test_incomplete_tree_never_returns_first_page_pins(self):
        failures = [
            self.page({"error": "not a tree"}),
            io.BytesIO(b'invalid JSON'),
            urllib.error.HTTPError(self.NEXT, 404, "missing page", {}, None),
        ]
        for failure in failures:
            with self.subTest(failure=type(failure).__name__), mock.patch.object(
                    pins._TREE_OPENER, "open", side_effect=[
                        self.page([self.entry("a.gguf")], f'<{self.NEXT}>; rel="next"'), failure]):
                with self.assertRaises(pins.TreePaginationError):
                    pins.fetch_tree("example", "model", "main")

    def test_transient_failure_drops_entire_tree_and_marks_unreachable(self):
        url = "https://huggingface.co/example/model/resolve/main/a.gguf"
        with mock.patch.object(pins._TREE_OPENER, "open", side_effect=[
            self.page([self.entry("a.gguf")], f'<{self.NEXT}>; rel="next"'),
            *[TimeoutError("synthetic timeout") for _ in range(4)],
        ]), mock.patch.object(pins.time, "sleep"):
            resolved, unpinned, unreachable = pins.resolve({url: pins.URL_RE.fullmatch(url).groupdict()}, False)
        self.assertEqual(resolved, {})
        self.assertEqual(len(unpinned), 1)
        self.assertEqual(unreachable, {("example", "model", "main")})


if __name__ == '__main__': unittest.main()
