# M1 bounded MI-GAN build handoff

This is an optional independent build-only lane while hosted execution is
queued. It does not replace the Linux original/native numerical panel. Do not
run these commands on the shared development host.

Use the MIT fixed256 prototype at immutable source
`2849cb784f4de22fbcb3763d66d6be09cfa7dc56` from branch
`test/migan-reference-dispatch-20261026`. Read AGENTS/CLAUDE/contributing first.
Check load, memory pressure, swap change and free storage before starting:

```sh
uptime
vm_stat
memory_pressure
sysctl vm.swapusage
df -h .
git fetch origin test/migan-reference-dispatch-20261026
git worktree add --detach ../CrispEmbed-m1-migan-build 2849cb784f4de22fbcb3763d66d6be09cfa7dc56
cd ../CrispEmbed-m1-migan-build
git submodule update --init --recursive
```

Defer if the machine is under pressure; keep compilation at two jobs and stop
if pressure/swapouts rise materially. No framework installation, checkpoint,
GGUF or raw-reference download is needed for this first stage. Build only the
explicit prototype targets, using the CPU configuration of the hosted lane:

```sh
cmake -S . -B build-m1-audit -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=OFF -DCRISPEMBED_NATIVE=OFF -DBUILD_SHARED_LIBS=OFF \
  -DGGML_BLAS=OFF -DGGML_METAL=OFF -DGGML_CUDA=OFF \
  -DGGML_OPENMP=OFF -DGGML_LLAMAFILE=OFF -DCRISPEMBED_BUILD_SHARED=OFF
python3 scripts/check-cpu-baseline.py build-m1-audit
/usr/bin/time -l cmake --build build-m1-audit \
  --target crispembed-migan test-migan-diff -j2
build-m1-audit/test-migan-diff --contracts
build-m1-audit/test-migan-diff --gate-controls
shasum -a 256 build-m1-audit/test-migan-diff \
  build-m1-audit/libcrispembed-migan.a build-m1-audit/libcrispembed-migan-test.a
nm -gU build-m1-audit/libcrispembed-migan.a | c++filt
nm -gU build-m1-audit/libcrispembed-migan-test.a | c++filt
```

Require successful configure/build/baseline check, `MIGAN_CONTRACT` passed,
and all unset/empty/literal-zero gate rejections. The ordinary archive must
exclude `migan::test::run_observed`; the test archive must include it. Record
compiler/CMake versions, source/ggml commits, flags, output hashes, exit codes,
peak RSS and before/after memory/swap measurements. A compile/helper pass is
not model parity, image quality, publication or production-artifact qualification.

Keep raw machine reports and paths in ignored local notes/build outputs.
A public handback may contain normalized source/toolchain/hash/results only;
exclude operator identities and host locations. Preserve any failure log before
fixing it. Do not fake `GITHUB_ACTIONS=true` to run hosted-only drivers locally.
The separately reviewed Linux original40330674649 and native40336246545 remain
the owning numerical panel, with all eight fixtures/494 declared comparisons,
original transport and tensor checks, unchanged image gates and retained limits.
