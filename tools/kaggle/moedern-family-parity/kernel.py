"""Kaggle CPU kernel: native MoEdern family engine vs a frozen original panel.

The original capture kernel's output is mounted read-only via kernel_sources.
Builds only the small standalone probe (ggml CPU) from an exact commit. Writes the
default-path gate to parity/ (unchanged) and, from compare_paths.py, the default
and CRISPEMBED_MOEDERN_FAST arms (threads 2 and 4) with per-path error/timing to
paths/paths.json.
"""
import os, subprocess, sys
COMMIT, MOUNT = '@COMMIT@', '/kaggle/input'
def sh(*a, **k): print('+', *a, flush=True); subprocess.run(list(a), check=True, **k)
sh('git', 'clone', '-q', '--filter=blob:none', 'https://github.com/CrispStrobe/CrispEmbed.git', '/kaggle/temp/ce')
os.chdir('/kaggle/temp/ce'); sh('git', 'checkout', '-q', COMMIT); sh('git', 'submodule', 'update', '-q', '--init', '--depth', '1', 'ggml')
try:  # ce-ccache-bootstrap: CrispEmbed harness from the pinned clone warms + auto-refreshes the seed (docs/kaggle-ccache.md)
    import glob as _ccg, importlib.util as _ccu, os as _cco
    for _cch in ["kaggle_harness.py"] + [_p for _g in ("/kaggle/working/*/", "/kaggle/temp/*/", "/tmp/*/", "/kaggle/working/*/*/", "/kaggle/temp/*/*/", "/tmp/*/*/") for _p in sorted(_ccg.glob(_g + "kaggle_harness.py"))]:
        if _cco.path.isfile(_cco.path.join(_cco.path.dirname(_cco.path.abspath(_cch)), "CMakeLists.txt")) and _cco.path.isfile(_cch):
            _ccs = _ccu.spec_from_file_location("ce_kaggle_harness", _cch)
            _ccm = _ccu.module_from_spec(_ccs)
            _ccs.loader.exec_module(_ccm)
            _ccm.ensure_build_cache()
            break
except Exception as _cce:
    print("ce-ccache: bootstrap skipped:", _cce, flush=True)
sh('cmake', '-S', 'tests/moedern-family', '-B', '/kaggle/temp/b', '-DCMAKE_BUILD_TYPE=Release')
sh('cmake', '--build', '/kaggle/temp/b', '--parallel', '4')
r = subprocess.run([sys.executable, 'tests/moedern-family/compare.py', '/kaggle/temp/b/moedern-family-probe', MOUNT, '/kaggle/working/parity'])
p = subprocess.run([sys.executable, 'tests/moedern-family/compare_paths.py', '/kaggle/temp/b/moedern-family-probe', MOUNT, '/kaggle/working/paths'])
open('/kaggle/working/commit.txt', 'w').write(COMMIT+'\n'); sys.exit(r.returncode or p.returncode)
