"""Tracked files must not contain credentials.

Three Kaggle API tokens sat in public tools/kaggle/* scripts from 2026-07 to
2026-09 (export KAGGLE_API_TOKEN=KGAT_...) and had to be revoked and scrubbed
from history. This scans every tracked text file for token formats we use, so
the next one fails CI instead of shipping.
"""
import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATTERNS = {
    "kaggle api token": re.compile(r"KGAT_[0-9a-f]{20,}"),
    "huggingface token": re.compile(r"\bhf_[A-Za-z0-9]{30,}"),
    "github token": re.compile(r"\b(?:ghp|gho|ghs|ghu)_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}"),
    "openai-style key": re.compile(r"\bsk-[A-Za-z0-9]{32,}"),
    "aws access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "private key block": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |)PRIVATE KEY-----"),
}


# Deliberately public, documented material. Keep this list tiny and justified.
ALLOW = {
    # Fixed self-signed C2PA "AI-generated" marker key, public by design (see the
    # header comment in src/core/crispasr_c2pa_default_cert.h): not a trust anchor.
    ("assets/c2pa/crispasr-default-c2pa.key", "private key block"),
    ("src/core/crispasr_c2pa_default_cert.h", "private key block"),
}


def scan(root: Path):
    files = subprocess.run(["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True).stdout.split(b"\0")
    hits = []
    for rel in filter(None, files):
        p = root / rel.decode()
        try:
            data = p.read_bytes()
        except OSError:
            continue
        if b"\0" in data[:8192]:  # binary
            continue
        text = data.decode("utf-8", "replace")
        for name, rx in PATTERNS.items():
            if (rel.decode(), name) in ALLOW:
                continue
            for m in rx.finditer(text):
                line = text.count("\n", 0, m.start()) + 1
                hits.append(f"{rel.decode()}:{line}: {name}")
    return hits


def kaggle_metadata_violations(root: Path):
    """Inspect public tracked templates, never ignored private submission files."""
    import json
    files = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).split(b"\0")
    bad = []
    for rel in filter(None, files):
        path = Path(rel.decode())
        if path.name != "kernel-metadata.json":
            continue
        metadata = json.loads((root / path).read_text())
        for field in ("id", "dataset_sources", "kernel_sources", "model_sources"):
            refs = [metadata.get(field, "")] if field == "id" else metadata.get(field) or []
            if any(ref and not ref.startswith("${KAGGLE_ACCOUNT}/") for ref in refs):
                # A failing guard must not echo an accidentally tracked private ID.
                bad.append(f"{path.as_posix()}: {field} needs an account placeholder")
    return bad


class NoSecretsTest(unittest.TestCase):
    def test_no_credentials_in_tracked_files(self):
        hits = scan(ROOT)
        self.assertEqual(hits, [], "credentials in tracked files:\n" + "\n".join(hits))

    def test_secret_diagnostics_retain_location_and_type_without_token(self):
        import tempfile
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            token = "KGAT_" + "0" * 32
            (root / "fixture.txt").write_text("first line\n" + token + "\n")
            subprocess.run(["git", "add", "fixture.txt"], cwd=root, check=True)
            hits = scan(root)
            self.assertEqual(hits, ["fixture.txt:2: kaggle api token"])
            self.assertNotIn(token, "\n".join(hits))
            self.assertNotIn(token[:8], "\n".join(hits))

    def test_kaggle_metadata_uses_account_placeholder(self):
        # The Kaggle account comes from the KAGGLE_ACCOUNT secret, rendered at push
        # time by tools/kaggle/kpush.py - never a literal owner in the repo.
        bad = kaggle_metadata_violations(ROOT)
        self.assertEqual(bad, [], "literal Kaggle owners in kernel metadata:\n" + "\n".join(bad))

    def test_metadata_guard_skips_private_files_and_redacts_tracked_violations(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            (root / ".gitignore").write_text("**/kernel-metadata.json\n")
            public = root / "tools/kaggle/public/kernel-metadata.json"
            private = root / "tools/kaggle/private/kernel-metadata.json"
            public.parent.mkdir(parents=True)
            private.parent.mkdir(parents=True)
            public.write_text(json.dumps({"id": "${KAGGLE_ACCOUNT}/example"}))
            # Invalid JSON proves that an ignored local file is never read.
            private.write_text("private local configuration, deliberately not JSON")
            subprocess.run(["git", "add", "-f", "--", str(public)], cwd=root, check=True)
            subprocess.run(["git", "check-ignore", "-q", str(private)], cwd=root, check=True)
            self.assertEqual(kaggle_metadata_violations(root), [])
            fields = ("id", "dataset_sources", "kernel_sources", "model_sources")
            for field in fields:
                with self.subTest(field=field):
                    metadata = {"id": "${KAGGLE_ACCOUNT}/example"}
                    literal = "private-placeholder/private-execution-placeholder"
                    metadata[field] = literal if field == "id" else [literal]
                    public.write_text(json.dumps(metadata))
                    self.assertEqual(kaggle_metadata_violations(root), [
                        f"tools/kaggle/public/kernel-metadata.json: {field} needs an account placeholder"
                    ])

    def test_kaggle_account_name_not_committed(self):
        # CI passes the KAGGLE_ACCOUNT secret; the name itself must not appear in
        # any tracked file (skipped where the secret is unavailable, e.g. forks).
        import os
        name = os.environ.get("KAGGLE_ACCOUNT", "").strip()
        if not name:
            self.skipTest("KAGGLE_ACCOUNT not set")
        rx = re.compile(re.escape(name), re.I)
        files = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout.split(b"\0")
        hits = []
        for rel in filter(None, files):
            try:
                data = (ROOT / rel.decode()).read_bytes()
            except OSError:
                continue
            if b"\0" not in data[:8192] and rx.search(data.decode("utf-8", "replace")):
                hits.append(rel.decode())
        self.assertEqual(hits, [], "Kaggle account name committed in:\n" + "\n".join(hits))

    def test_patterns_fire(self):
        # Positive control: each pattern must match a synthetic sample.
        samples = {
            "kaggle api token": "KGAT_" + "0" * 32,
            "huggingface token": "hf_" + "a" * 34,
            "github token": "ghp_" + "b" * 36,
            "openai-style key": "sk-" + "c" * 40,
            "aws access key": "AKIA" + "D" * 16,
            "private key block": "-----BEGIN RSA " + "PRIVATE KEY-----",  # split: this file is scanned too
        }
        for name, s in samples.items():
            self.assertTrue(PATTERNS[name].search(s), name)


if __name__ == "__main__":
    sys.exit(unittest.main())
