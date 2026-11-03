# Portable build mode: MinerU2.5 Q4_K determinism (3 November 2026)

Three hosted runs of `tools/ci-heavy/qwen2vl_q4k_sweep.sh mineru r2,r4` at `33ecd1d6` with
`build_mode=portable` (configure log: `CrispEmbed CPU build mode: portable (GGML_NATIVE=OFF,
CRISPEMBED_NATIVE=OFF ...)`). The source F32 GGUF had the same SHA-256 in all three runs.

| run | CPU | r2 final cos min | r4 final cos min |
|---|---|---|---|
| 41587614152 | AMD EPYC 9V45 (AVX-512) | 0.952109 (japanese) | 0.970185 (table) |
| 41587618648 | AMD EPYC 9V45 (AVX-512) | 0.952109 (japanese) | 0.970185 (table) |
| 41587621427 | AMD EPYC 9V45 (AVX-512) | 0.952109 (japanese) | 0.970185 (table) |

The `r2-cpu.cosine.json` and `r4-cpu.cosine.json` files are byte-identical across the three runs.

Before this change, `-march=native` builds of the same bytes scored 0.9673 (r2) and 0.9732 (r4) on an
EPYC 9V45 (the `mineru_q4k_v1` receipt), and 0.9521 and 0.9702 on another runner. The portable build on the
EPYC 9V45 reproduces the lower pair to six digits. So the 0.015 spread came from the AVX-512 kernels that
native compiles on AVX-512 runners, not from the files. The portable value is the one a qualification
receipt now quotes. Both recipes stay above the 0.94 Q4_K bar.

Limits: all three runs landed on the same CPU model, so the cross-vendor case (an Intel runner) is not
measured here. A portable build compiles one fixed x86 baseline with no runtime ISA dispatch in the CPU
backend, so it is expected to match as well.
