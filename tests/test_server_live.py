"""Live HTTP test for the embedding endpoints' JSON input parsing (issue #34).

Boots ``crispembed-server`` with a text-embedding model, then sends the exact
escaped payloads from issue #34 (values containing ``]``, ``\\"`` and ``\\\\``)
to the three embedding routes and asserts the returned count matches the number
of inputs. Before the fix these returned the wrong cardinality
("returned 7 embeddings for 6 inputs").

Because it needs a real model, this test *loads whatever embedding GGUF you point
it at* — so pointing it at ``nomic-embed-text-v2-moe`` doubles as the load check
for issue #33 (the model that previously aborted with
``missing required tensor ... attn.q.weight``).

Model + binary discovery (skips cleanly if unavailable, so PR CI without models
just skips instead of failing):

  CRISPEMBED_TEST_EMBED_MODEL=/path/to/text-embed.gguf   # required to run
  CRISPEMBED_TEST_RERANK_MODEL=/path/to/cross-encoder.gguf  # /v1/rerank (issue #51)
  CRISPEMBED_SERVER_BIN=/path/to/crispembed-server       # else auto-detected

Usage:
  CRISPEMBED_TEST_EMBED_MODEL=~/crispembed-live-cache/nomic-embed-text-v2-moe.Q4_K_M.gguf \
      python -m unittest tests/test_server_live.py
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def _find_server_bin() -> str | None:
    env = os.environ.get("CRISPEMBED_SERVER_BIN")
    if env and Path(env).is_file():
        return env
    for cand in (
        REPO / "build" / "crispembed-server",
        REPO / "build" / "bin" / "crispembed-server",
    ):
        if cand.is_file():
            return str(cand)
    return None


def _find_model() -> str | None:
    env = os.environ.get("CRISPEMBED_TEST_EMBED_MODEL")
    if env and Path(env).expanduser().is_file():
        return str(Path(env).expanduser())
    return None


def _find_rerank_model() -> str | None:
    """A cross-encoder GGUF for the /rerank + /v1/rerank routes (issue #51).

    Separate from the embedding model on purpose: the rerank routes are guarded
    by ``crispembed_is_reranker`` and a text-embedding model gets a 400 there,
    so they need their own server instance with a cross-encoder loaded.
    """
    env = os.environ.get("CRISPEMBED_TEST_RERANK_MODEL")
    if env and Path(env).expanduser().is_file():
        return str(Path(env).expanduser())
    return None


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _post(url: str, payload: str, timeout: float = 60.0) -> dict:
    req = urllib.request.Request(
        url, data=payload.encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


SERVER_BIN = _find_server_bin()
MODEL = _find_model()
RERANK_MODEL = _find_rerank_model()


@unittest.skipUnless(SERVER_BIN and MODEL, "needs crispembed-server + CRISPEMBED_TEST_EMBED_MODEL")
class ServerJsonInputLive(unittest.TestCase):
    proc: subprocess.Popen
    port: int

    @classmethod
    def setUpClass(cls) -> None:
        cls.port = _free_port()
        cls.proc = subprocess.Popen(
            [SERVER_BIN, "-m", MODEL, "--host", "127.0.0.1", "--port", str(cls.port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        base = f"http://127.0.0.1:{cls.port}"
        deadline = time.time() + 120  # model load can be slow on CPU
        while time.time() < deadline:
            if cls.proc.poll() is not None:
                raise RuntimeError("server exited during startup (model failed to load?)")
            try:
                urllib.request.urlopen(base + "/health", timeout=2).read()
                return
            except Exception:
                time.sleep(1)
        raise RuntimeError("server did not become ready in time")

    @classmethod
    def tearDownClass(cls) -> None:
        if getattr(cls, "proc", None):
            cls.proc.terminate()
            try:
                cls.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                cls.proc.kill()

    def _base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def test_openai_embeddings_escaped_array(self) -> None:
        # Issue #34 primary reproduction: 6 inputs with ], escaped quotes, backslash.
        payload = (
            '{"model":"m","input":['
            '"normal text",'
            '"text with ] bracket inside",'
            '"text with escaped quote: \\"hello\\"",'
            '"text with backslash: \\\\",'
            '"another normal text",'
            '"final item"]}'
        )
        d = _post(self._base() + "/v1/embeddings", payload)
        self.assertEqual(len(d["data"]), 6, "one embedding per input")

    def test_ollama_embed_escaped_array(self) -> None:
        # Issue #34 second reproduction: 4 inputs incl. an escaped newline.
        payload = '{"input":["plain text","text with ] bracket","text with \\"quoted\\" part","line\\nbreak"]}'
        d = _post(self._base() + "/api/embed", payload)
        self.assertEqual(len(d["embeddings"]), 4, "one embedding per input")

    def test_ollama_embeddings_single_prompt(self) -> None:
        # Single prompt containing a backslash and a bracket must not truncate.
        payload = '{"model":"m","prompt":"a \\\\ b ] c"}'
        d = _post(self._base() + "/api/embeddings", payload)
        self.assertGreater(len(d["embedding"]), 0, "non-empty embedding vector")

    def test_native_embed_texts_escaped_array(self) -> None:
        # A1: /embed ("texts") carried the same delimiter-scan bug as the three
        # endpoints issue #34 named — it was outside that issue's repro scope.
        payload = '{"texts":["plain","has ] bracket","has \\"quote\\"","has \\\\ backslash"]}'
        d = _post(self._base() + "/embed", payload)
        embs = d.get("embeddings", d.get("data"))
        self.assertEqual(len(embs), 4, "one embedding per input text")

    def test_native_embed_single_text(self) -> None:
        # "texts" absent -> falls back to "text" single string.
        d = _post(self._base() + "/embed", '{"text":"a ] b"}')
        embs = d.get("embeddings", d.get("data"))
        self.assertEqual(len(embs), 1)

    def test_scan_split_image_field_decoy(self) -> None:
        # core_json scalars/strings: /scan/split is model-free and reads the
        # "image" field. A decoy key whose VALUE is the string "image" must not
        # be mistaken for the image key (the pre-fix naive find() would grab it
        # and fail to load the image). Skips cleanly if no test image is set.
        img = os.environ.get("CRISPEMBED_TEST_IMAGE")
        if not img or not Path(img).is_file():
            self.skipTest("set CRISPEMBED_TEST_IMAGE to a readable image")
        payload = json.dumps({"note": "image", "image": img})
        d = _post(self._base() + "/scan/split", payload)
        self.assertIn("width", d, "must have loaded the real image, not the decoy value")
        self.assertGreater(d["width"], 0)

    def test_b2_decoy_key_value_ignored(self) -> None:
        # B2: an unrelated key whose VALUE array contains the literal "input"
        # must not be mistaken for the input key. The real "input" has 3 items.
        payload = '{"labels":["input"],"input":["one","two","three"]}'
        d = _post(self._base() + "/v1/embeddings", payload)
        self.assertEqual(len(d["data"]), 3, "must locate the real input key, not the decoy value")

    def test_b1_responses_are_strict_json(self) -> None:
        # B1 regression guard for the escaper refactor: the server's own
        # response must be STRICT JSON. Python's json.loads (used by _post)
        # rejects raw control chars, so a valid parse here proves the response
        # went through the escaper. (The escaper's exhaustive proof — decode(
        # escape(x))==x over all 256 bytes — lives in test-server-json-input;
        # embedding responses don't echo arbitrary control chars, the OCR/NER
        # endpoints do, and those need their own models.)
        for route, payload in (
            ("/v1/embeddings", '{"input":["a","b"]}'),
            ("/api/embed", '{"input":["x"]}'),
            ("/api/embeddings", '{"prompt":"hi"}'),
        ):
            d = _post(self._base() + route, payload)  # raises on a strict-JSON failure
            self.assertIsInstance(d, dict)


@unittest.skipUnless(SERVER_BIN and MODEL, "needs crispembed-server + CRISPEMBED_TEST_EMBED_MODEL")
class ServerJsonInputLegacyGateLive(unittest.TestCase):
    """A/B the CRISPEMBED_SERVER_LEGACY_JSON=1 gate end-to-end.

    Proves the env gate actually switches the running server's parser: the same
    escaped payload that returns 6 embeddings by default returns the wrong count
    under the legacy scan, while a benign payload is unaffected either way.
    Keeps the gate honest as a bisection switch (it is never removed).
    """

    proc: subprocess.Popen
    port: int

    @classmethod
    def setUpClass(cls) -> None:
        cls.port = _free_port()
        env = dict(os.environ, CRISPEMBED_SERVER_LEGACY_JSON="1")
        cls.proc = subprocess.Popen(
            [SERVER_BIN, "-m", MODEL, "--host", "127.0.0.1", "--port", str(cls.port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )
        base = f"http://127.0.0.1:{cls.port}"
        deadline = time.time() + 120
        while time.time() < deadline:
            if cls.proc.poll() is not None:
                raise RuntimeError("server exited during startup")
            try:
                urllib.request.urlopen(base + "/health", timeout=2).read()
                return
            except Exception:
                time.sleep(1)
        raise RuntimeError("server did not become ready in time")

    @classmethod
    def tearDownClass(cls) -> None:
        if getattr(cls, "proc", None):
            cls.proc.terminate()
            try:
                cls.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                cls.proc.kill()

    def _base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def test_benign_payload_is_gate_neutral(self) -> None:
        # No escapes/brackets -> legacy must agree with the default parser.
        payload = '{"input":["alpha","beta","gamma"]}'
        d = _post(self._base() + "/v1/embeddings", payload)
        self.assertEqual(len(d["data"]), 3, "gate is output-neutral on benign payloads")

    def test_legacy_gate_reproduces_the_bug(self) -> None:
        # The issue #34 payload: legacy must NOT return 6 (that's the bug we keep
        # the gate to reproduce). If this ever returns 6, the gate stopped working.
        payload = (
            '{"input":["normal text","text with ] bracket inside",'
            '"text with escaped quote: \\"hello\\"","text with backslash: \\\\",'
            '"another normal text","final item"]}'
        )
        try:
            d = _post(self._base() + "/v1/embeddings", payload)
            n = len(d["data"])
        except urllib.error.HTTPError as e:
            n = 0 if e.code == 400 else -1  # legacy can also drop everything -> 400
        self.assertNotEqual(n, 6, "legacy scan must still mis-parse (gate is live)")


@unittest.skipUnless(SERVER_BIN and RERANK_MODEL, "needs crispembed-server + CRISPEMBED_TEST_RERANK_MODEL")
class ServerRerankLive(unittest.TestCase):
    """Issue #51 — /v1/rerank in the de-facto Cohere / Jina shape.

    Everything here is a property of the response, not a hard-coded score: the
    point is that the endpoint is a drop-in for clients that expect that schema,
    and that it agrees with the native /rerank route on the only thing a
    reranker is actually for — the ORDER.

    The fixture is deliberately unambiguous (one document answers the query,
    one is off-topic) so the ranking assertion tests the endpoint rather than
    the model's discrimination.
    """

    QUERY = "What is the capital of France?"
    DOCS = [
        "Bananas are a good source of potassium.",
        "Paris is the capital and most populous city of France.",
        "The Rust compiler enforces memory safety without a garbage collector.",
    ]
    RELEVANT = 1  # index of the Paris document

    proc: subprocess.Popen
    port: int

    @classmethod
    def setUpClass(cls) -> None:
        cls.port = _free_port()
        cls.proc = subprocess.Popen(
            [SERVER_BIN, "-m", RERANK_MODEL, "--host", "127.0.0.1", "--port", str(cls.port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        base = f"http://127.0.0.1:{cls.port}"
        deadline = time.time() + 120
        while time.time() < deadline:
            if cls.proc.poll() is not None:
                raise RuntimeError("server exited during startup (model failed to load?)")
            try:
                health = json.loads(urllib.request.urlopen(base + "/health", timeout=2).read().decode("utf-8"))
                if not health.get("reranker"):
                    raise unittest.SkipTest("CRISPEMBED_TEST_RERANK_MODEL is not a cross-encoder")
                return
            except unittest.SkipTest:
                raise
            except Exception:
                time.sleep(1)
        raise RuntimeError("server did not become ready in time")

    @classmethod
    def tearDownClass(cls) -> None:
        if getattr(cls, "proc", None):
            cls.proc.terminate()
            try:
                cls.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                cls.proc.kill()

    def _base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def _v1(self, payload: dict) -> dict:
        return _post(self._base() + "/v1/rerank", json.dumps(payload))

    def test_string_documents(self) -> None:
        d = self._v1({"model": "m", "query": self.QUERY, "documents": self.DOCS})
        self.assertEqual(len(d["results"]), len(self.DOCS), "one result per document")
        self.assertEqual(d["results"][0]["index"], self.RELEVANT, "the on-topic document ranks first")
        for r in d["results"]:
            self.assertIn("relevance_score", r, "Cohere/Jina score key")
            self.assertNotIn("score", r, "the native raw-logit key must not leak into /v1")
            self.assertNotIn("document", r, "documents are opt-in (Cohere default)")

    def test_object_documents_same_result(self) -> None:
        # The shape json_extract_strings would have mis-split into 2x garbage
        # documents. Same query, same corpus, expressed as objects: identical
        # cardinality and identical winner.
        d = self._v1({"query": self.QUERY, "documents": [{"text": t} for t in self.DOCS]})
        self.assertEqual(len(d["results"]), len(self.DOCS), "object form must not double the count")
        self.assertEqual(d["results"][0]["index"], self.RELEVANT)

    def test_return_documents_echoes_text(self) -> None:
        d = self._v1({"query": self.QUERY, "documents": self.DOCS, "return_documents": True})
        top = d["results"][0]
        self.assertEqual(top["document"]["text"], self.DOCS[top["index"]], "echoed text matches its index")

    def test_relevance_scores_are_probabilities_and_sorted(self) -> None:
        d = self._v1({"query": self.QUERY, "documents": self.DOCS})
        scores = [r["relevance_score"] for r in d["results"]]
        for s in scores:
            # Bounds, not strict inequality: the response rounds to 9 dp, so a
            # very confident logit can legitimately print as 1.000000000. The
            # strict (0,1) property of the sigmoid itself is pinned in
            # test-server-json-input (R10).
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 1.0)
        self.assertEqual(scores, sorted(scores, reverse=True), "results are descending by score")

    def test_top_n_truncates(self) -> None:
        d = self._v1({"query": self.QUERY, "documents": self.DOCS, "top_n": 1})
        self.assertEqual(len(d["results"]), 1)
        self.assertEqual(d["results"][0]["index"], self.RELEVANT)

    def test_ranking_agrees_with_native_route(self) -> None:
        # sigmoid is monotonic, so /v1/rerank must not reorder anything relative
        # to /rerank. This is the guard that keeps the score transform honest:
        # a non-monotonic "normalization" would break here, not in a unit test.
        native = _post(self._base() + "/rerank", json.dumps({"query": self.QUERY, "documents": self.DOCS}))
        v1 = self._v1({"query": self.QUERY, "documents": self.DOCS})
        self.assertEqual(
            [r["index"] for r in native["results"]],
            [r["index"] for r in v1["results"]],
            "sigmoid must preserve the native ordering",
        )

    def test_envelope_fields(self) -> None:
        d = self._v1({"model": "whatever", "query": self.QUERY, "documents": self.DOCS})
        self.assertEqual(d["object"], "list")
        self.assertIn("id", d)
        self.assertIn("model", d)
        self.assertIn("meta", d)

    def test_missing_documents_is_400(self) -> None:
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self._v1({"query": self.QUERY})
        self.assertEqual(cm.exception.code, 400)

    def test_missing_query_is_400(self) -> None:
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self._v1({"documents": self.DOCS})
        self.assertEqual(cm.exception.code, 400)

    def test_native_route_is_unchanged(self) -> None:
        # /rerank keeps its original contract: raw `score`, always-echoed
        # `document`, no relevance_score.
        d = _post(self._base() + "/rerank", json.dumps({"query": self.QUERY, "documents": self.DOCS}))
        self.assertEqual(d["query"], self.QUERY)
        top = d["results"][0]
        self.assertIn("score", top)
        self.assertNotIn("relevance_score", top)
        self.assertEqual(top["document"], self.DOCS[top["index"]])


if __name__ == "__main__":
    if not (SERVER_BIN and MODEL):
        print("SKIP: set CRISPEMBED_TEST_EMBED_MODEL (and build crispembed-server) to run")
    if not (SERVER_BIN and RERANK_MODEL):
        print("SKIP: set CRISPEMBED_TEST_RERANK_MODEL to run the /v1/rerank tests (issue #51)")
    unittest.main()
