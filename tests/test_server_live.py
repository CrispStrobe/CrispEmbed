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


if __name__ == "__main__":
    if not (SERVER_BIN and MODEL):
        print("SKIP: set CRISPEMBED_TEST_EMBED_MODEL (and build crispembed-server) to run")
    unittest.main()
