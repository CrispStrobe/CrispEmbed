"""Pure reviewed OvisOCR3 native source, CMake and ggml gitlink closure.

The exact source set follows the native14 producer's glob and named checker
list. A later source addition, removal, or byte change requires a new audit.
"""

import hashlib
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[2]
NAMED = (
    "src/crispembed_diff.h", "tests/qwen35_parity.h",
    "tests/test_qwen35_decoder.cpp", "tests/test_qwen35_vision.cpp",
    "tests/test_qwen35_delta.cpp", "tests/test_ovisocr3_context.cpp",
)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def expected_native_sources(root=ROOT):
    """Mirror native14's qwen35* C++ headers/sources plus six named probes."""
    root = Path(root)
    named = {root / name for name in NAMED}
    if any(not path.is_file() for path in named):
        raise ValueError("Reviewed named native source/checker file absent")
    qwen = {path for path in (root / "src").glob("qwen35*")
            if path.is_file() and path.suffix in (".cpp", ".h")}
    if not qwen:
        raise ValueError("Reviewed qwen35 source family absent")
    return {str(path.relative_to(root)) for path in qwen | named}


def git_output(args, *, root=ROOT):
    return subprocess.check_output(["git", *args], cwd=root,
                                   stderr=subprocess.DEVNULL, timeout=120)


def fetch_reviewed_commit(head, *, root=ROOT):
    # Hosted checkout is shallow. Fetch only the reviewed commit object, never
    # full history, submodules, model weights, or any working-tree sources.
    subprocess.run(["git", "fetch", "--no-tags", "--depth=1", "origin", head],
                   cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                   timeout=120, check=True)


def gitlink_line(value, *, staged=False):
    pattern = (rb"160000 ([0-9a-f]{40}) 0\tggml\n" if staged
               else rb"160000 commit ([0-9a-f]{40})\tggml\n")
    match = re.fullmatch(pattern, value)
    if match is None:
        raise ValueError("Reviewed ggml gitlink metadata differs")
    return match.group(1).decode()


def tree_oid(value):
    oid = value.decode().strip()
    if re.fullmatch(r"[0-9a-f]{40}", oid) is None:
        raise ValueError("Reviewed src Git tree identity absent")
    return oid


def clean_paths(call, root, commands, label):
    for args in commands:
        if call(args, root=root).strip():
            raise ValueError(label + " has tracked, staged, or untracked source changes")


def require_native_source_closure(producer, head, *, root=ROOT,
                                  call=git_output, fetch=fetch_reviewed_commit):
    """Fail before build/model access when any qualified native input drifts."""
    root = Path(root)
    if (not isinstance(producer, dict) or not isinstance(head, str)
            or re.fullmatch(r"[0-9a-f]{40}", head) is None
            or producer.get("source_commit") != head):
        raise ValueError("Reviewed native source head absent or differs")
    source = producer.get("native_source_sha256")
    expected = expected_native_sources(root)
    if (not isinstance(source, dict) or set(source) != expected
            or any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
                   for value in source.values())):
        raise ValueError("Reviewed native source/checker file set differs")
    for name in sorted(expected):
        if digest(root / name) != source[name]:
            raise ValueError("Reviewed native source/checker SHA differs: " + name)
    try:
        call(["cat-file", "-e", head + "^{commit}"], root=root)
    except subprocess.CalledProcessError:
        fetch(head, root=root)
        call(["cat-file", "-e", head + "^{commit}"], root=root)
    reviewed_src = tree_oid(call(["rev-parse", head + ":src"], root=root))
    current_src = tree_oid(call(["rev-parse", "HEAD:src"], root=root))
    if reviewed_src != current_src:
        raise ValueError("src Git tree differs from reviewed native head")
    clean_paths(call, root, (
        ["diff", "--name-only", "HEAD", "--", "src"],
        ["diff", "--cached", "--name-only", "HEAD", "--", "src"],
        ["diff", "--name-only", "--", "src"],
        ["ls-files", "--others", "--", "src"],
    ), "src checkout")
    reviewed_cmake = call(["show", head + ":CMakeLists.txt"], root=root)
    current_cmake = (root / "CMakeLists.txt").read_bytes()
    if current_cmake != reviewed_cmake:
        raise ValueError("CMakeLists.txt bytes differ from reviewed native head")
    reviewed_ggml = gitlink_line(call(["ls-tree", head, "ggml"], root=root))
    current_ggml = gitlink_line(call(["ls-tree", "HEAD", "ggml"], root=root))
    staged_ggml = gitlink_line(call(["ls-files", "--stage", "ggml"], root=root), staged=True)
    ggml_path = root / "ggml"
    checkout_root = call(["-C", str(ggml_path), "rev-parse", "--show-toplevel"], root=root).decode().strip()
    checkout_head = call(["-C", str(ggml_path), "rev-parse", "HEAD"], root=root).decode().strip()
    if (reviewed_ggml != current_ggml or reviewed_ggml != staged_ggml
            or Path(checkout_root).resolve() != ggml_path.resolve()
            or checkout_head != reviewed_ggml):
        raise ValueError("ggml gitlink differs from reviewed native head")
    clean_paths(call, root, (
        ["-C", str(ggml_path), "diff", "--name-only", "HEAD", "--"],
        ["-C", str(ggml_path), "diff", "--cached", "--name-only", "HEAD", "--"],
        ["-C", str(ggml_path), "diff", "--name-only", "--"],
        ["-C", str(ggml_path), "ls-files", "--others", "--exclude-standard"],
    ), "ggml checkout")
    return {"reviewed_head": head, "file_count": len(expected),
            "file_set_sha256": hashlib.sha256("\n".join(sorted(expected)).encode()).hexdigest(),
            "src_tree_oid": reviewed_src, "src_checkout_clean": True,
            "cmake_sha256": hashlib.sha256(current_cmake).hexdigest(),
            "ggml_gitlink": reviewed_ggml, "gitlink_index_exact": True,
            "ggml_checkout_exact": True, "ggml_checkout_clean": True}
