#!/usr/bin/env python3
"""Push a Kaggle kernel directory without hardcoding the Kaggle account.

Kernel dirs under tools/kaggle/ carry the placeholder ${KAGGLE_ACCOUNT} in
kernel-metadata.json (id, dataset_sources, kernel_sources, model_sources) and
may use it in the kernel's code file. This renders a temporary copy with the
account from the environment and pushes that, authenticated with the token
from the environment - nothing account- or token-specific is ever committed.

    KAGGLE_ACCOUNT=<username> KAGGLE_TOKEN=<api token> \\
        python3 tools/kaggle/kpush.py tools/kaggle/<kernel-dir> [--sha <commit>] \\
            [--refresh-ccache] [extra kaggle push args]

--sha replaces the __SHA__ placeholder in the code file (exact clone pin).
Unless kernel-metadata.json sets docker_image_pinning_type, the push uses
"latest", so every kernel (and the ccache seed) runs on the same Kaggle image.
--refresh-ccache waits for the pushed kernel to finish, then runs
tools/kaggle/ccache_refresh.py --from-kernel <its id>, which versions the
private crispembed-ccache seed only if the kernel exported one (stale seed).

In GitHub Actions both come from the repository secrets of the same names.
Locally, export them from a private env file (never from a command line).
One Kaggle account per person (Kaggle Terms of Use): do not point this at a
second account to get around quotas or session limits.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PLACEHOLDER = "${KAGGLE_ACCOUNT}"


def render(text: str, account: str) -> str:
    return text.replace(PLACEHOLDER, account)


def main(argv):
    if len(argv) < 2:
        sys.exit(__doc__)
    argv = list(argv)
    refresh = "--refresh-ccache" in argv
    if refresh:
        argv.remove("--refresh-ccache")
    sha = None
    if "--sha" in argv:
        i = argv.index("--sha")
        sha = argv[i + 1]
        del argv[i:i + 2]
        if not re.fullmatch(r"[0-9a-f]{40}", sha):
            sys.exit("kpush: --sha needs a full 40-hex commit")
    src = Path(argv[1]).resolve()
    account = os.environ.get("KAGGLE_ACCOUNT", "").strip()
    token = os.environ.get("KAGGLE_TOKEN", "").strip()
    if not account or not token:
        sys.exit("kpush: set KAGGLE_ACCOUNT and KAGGLE_TOKEN (GitHub secrets / private env file)")
    meta_path = src / "kernel-metadata.json"
    if not meta_path.is_file():
        sys.exit(f"kpush: no kernel-metadata.json in {src}")
    with tempfile.TemporaryDirectory(prefix="kpush-") as tmp:
        dst = Path(tmp) / src.name
        shutil.copytree(src, dst)
        meta = json.loads(render(meta_path.read_text(), account))
        # Run on the current Kaggle image unless the metadata pins one. An existing
        # kernel otherwise stays on the image of its first push, and ccache
        # (CCACHE_COMPILERCHECK=content) misses every object across images: the
        # seed on GCC 11.4 / CUDA 12.8 gave a new kernel on GCC 13.3 / CUDA 13.0
        # zero hits (#216). Seed and consumers on "latest" share one toolchain.
        meta.setdefault("docker_image_pinning_type", "latest")
        code = meta.get("code_file")
        if code and (dst / code).is_file():
            text = render((dst / code).read_text(), account)
            if sha:
                text = text.replace("__SHA__", sha)
            (dst / code).write_text(text)
        (dst / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
        leftover = [p.name for p in (dst / "kernel-metadata.json", dst / (code or "")) if p.is_file() and PLACEHOLDER in p.read_text()]
        if leftover:
            sys.exit(f"kpush: unrendered placeholder in {leftover}")
        env = dict(os.environ, KAGGLE_API_TOKEN=token)
        r = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "push", "-p", str(dst)] + argv[2:],
                           env=env, text=True, capture_output=True)
        sys.stdout.write(r.stdout)
        sys.stderr.write(r.stderr)
        sys.stdout.flush()
        rc = r.returncode
        # The kaggle CLI exits 0 on a refused push ("Kernel push error: Maximum
        # batch GPU session count ..."); without this --refresh-ccache would wait
        # on, and report, the previous version's output.
        if not rc and re.search(r"push error|error:", r.stdout + r.stderr, re.I):
            rc = 1
    if rc or not refresh:
        return rc
    time.sleep(30)  # let the new version register before polling its status
    return subprocess.call([sys.executable, str(Path(__file__).with_name("ccache_refresh.py")),
                            "--from-kernel", meta["id"], "--wait"], env=env)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
