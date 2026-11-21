#!/usr/bin/env python3
"""ccache host-ISA keying proof (docs/kaggle-ccache.md, the SIGILL seed).

Seeds a ccache with a `-march=native` object on this host (AVX-512), then
compiles the same command line as an AVX2-only host: the gcc driver runs under
`qemu-x86_64 -cpu Haswell`, so `-march=native` resolves to haswell there, and
the result runs under the same emulated CPU. Old harness behaviour (no ISA key)
reuses the AVX-512 object and dies with SIGILL; with
kaggle_harness._key_cache_on_host_isa() the AVX2 host misses, compiles AVX2
code, runs, and hits on its own second build.

Needs gcc, ccache and qemu-user (qemu-x86_64) on an AVX-512 x86-64 host.

    python3 tools/kaggle/ccache_isa_proof.py [--out receipt.json]
"""
import argparse, json, os, shutil, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = r'''#include <stdio.h>
#include <immintrin.h>
float dot(const float *a, const float *b, int n) {
#if defined(__AVX512F__)
    __m512 acc = _mm512_setzero_ps();               /* ggml-style: kernel chosen on __AVX512F__ */
    for (int i = 0; i < n; i += 16) acc = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), acc);
    return _mm512_reduce_add_ps(acc);
#elif defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < n; i += 8) acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc);
    float t[8]; _mm256_storeu_ps(t, acc); float s = 0; for (int i = 0; i < 8; i++) s += t[i]; return s;
#else
    float s = 0; for (int i = 0; i < n; i++) s += a[i] * b[i]; return s;
#endif
}
int main(void) {
    float a[64], b[64]; for (int i = 0; i < 64; i++) { a[i] = i; b[i] = 1; }
#if defined(__AVX512F__)
    const char *isa = "avx512";
#elif defined(__AVX2__)
    const char *isa = "avx2";
#else
    const char *isa = "scalar";
#endif
    printf("%s %.0f\n", isa, dot(a, b, 64));
    return 0;
}
'''
AVX512_GCC = "#!/bin/sh\nexec /usr/bin/gcc \"$@\"\n"
AVX2_GCC = "#!/bin/sh\nexec qemu-x86_64 -cpu Haswell /usr/bin/gcc \"$@\"\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    sys.path.insert(0, str(ROOT))
    import kaggle_harness as kh
    with tempfile.TemporaryDirectory(prefix="ce-ccache-isa-") as tmp:
        P = Path(tmp)
        for side, text in (("a", AVX512_GCC), ("b", AVX2_GCC)):
            (P / side).mkdir(); (P / side / "gcc").write_text(text); (P / side / "gcc").chmod(0o755)
        (P / "src").mkdir(); (P / "src/k.c").write_text(SOURCE)
        def run(cmd, env): return subprocess.run(cmd, env=env, capture_output=True, text=True)
        def step(label, side, keyed, cache):
            env = {k: v for k, v in os.environ.items() if not k.startswith('CCACHE')}
            env.update(CCACHE_DIR=str(cache), CCACHE_COMPILERCHECK='string:same-gcc-13', CCACHE_NOHASHDIR='1', CC=str(P / side / 'gcc'))
            os.environ.pop('CCACHE_EXTRAFILES', None); os.environ['CC'] = env['CC']
            digest = None
            if keyed:
                digest = kh._key_cache_on_host_isa(P / side)
                env['CCACHE_EXTRAFILES'] = os.environ['CCACHE_EXTRAFILES']
            subprocess.run(['ccache', '-z'], env=env, capture_output=True)
            obj, exe = P / f'{label}.o', P / f'{label}.bin'
            subprocess.run(['ccache', str(P / side / 'gcc'), '-O2', '-march=native', '-c', str(P / 'src/k.c'), '-o', str(obj)], env=env, check=True)
            st = dict(l.split('\t', 1) for l in run(['ccache', '--print-stats'], env).stdout.splitlines() if '\t' in l)
            subprocess.run(['/usr/bin/gcc', str(obj), '-o', str(exe)], check=True)
            cpu = [] if side == 'a' else ['qemu-x86_64', '-cpu', 'Haswell']
            r = run(cpu + [str(exe)], env)
            return {'step': label, 'host': 'avx512 (native)' if side == 'a' else 'avx2 (qemu Haswell)', 'isa_key': digest,
                    'hit': int(st.get('direct_cache_hit', 0)) + int(st.get('preprocessed_cache_hit', 0)), 'miss': int(st.get('cache_miss', 0)),
                    'exit': r.returncode, 'signal': 'SIGILL' if r.returncode in (-4, 132) else None, 'stdout': r.stdout.strip(), 'stderr': r.stderr.strip()[-120:]}
        out = []
        for keyed in (False, True):
            cache = P / ('cache-keyed' if keyed else 'cache-old'); shutil.rmtree(cache, ignore_errors=True)
            out.append(step(('new' if keyed else 'old') + '-seed', 'a', keyed, cache))
            out.append(step(('new' if keyed else 'old') + '-avx2', 'b', keyed, cache))
            out.append(step(('new' if keyed else 'old') + '-avx2-warm', 'b', keyed, cache))
        receipt = {"schema": "crispembed-ccache-host-isa-proof/v1", "ccache": subprocess.run(["ccache", "--version"], capture_output=True, text=True).stdout.splitlines()[0],
                   "steps": [{k: v for k, v in s.items() if k != "stderr"} for s in out]}
        old, new = out[1], out[4]
        receipt["old_behaviour_reuses_avx512_object_and_sigills"] = old["hit"] == 1 and old["signal"] == "SIGILL"
        receipt["new_behaviour_misses_and_runs_avx2"] = new["hit"] == 0 and new["exit"] == 0 and new["stdout"].startswith("avx2")
        text = json.dumps(receipt, indent=1) + "\n"
        print(text, end="")
        if args.out:
            args.out.write_text(text)
        return 0 if receipt["old_behaviour_reuses_avx512_object_and_sigills"] and receipt["new_behaviour_misses_and_runs_avx2"] else 1


if __name__ == "__main__":
    sys.exit(main())