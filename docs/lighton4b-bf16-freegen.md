# Original LightOnOCR-3-4B CPU BF16 free generation

This separate experimental protocol captures official original-model free
generation on the frozen document image, empty image-only prompt and 256-token
cap. It does not admit a producer into document-v1/v2 or replace strict F32
references. Use exact executable source
`07543dcc80ebcecad644815d9397be2570a51dd4` and
[the frozen protocol](lighton4b-bf16-freegen-v1.json).

Install `tools/lighton4b-bf16-requirements.txt` using the CPU wheel index required
by its included requirements, then run on an authorized hosted CPU worker:

```sh
python tools/lighton4b_bf16_freegen.py \
  --source-sha 07543dcc80ebcecad644815d9397be2570a51dd4 \
  --output /kaggle/working/lighton4b-bf16-freegen \
  --scratch /tmp/lighton4b-bf16 --threads 2
```

`--prepare-only` verifies pinned metadata/full tokenizer and actual official
processor inputs without acquiring the checkpoint or loading the model.
`--model-dir` may supply an existing snapshot, whose full payload is verified.
The model revision is `b71010f095dc3735de7ee6b9969bff8f8f50b39a`.
Full checkpoint bytes/hash, tokenizer hash, image/prompt/config and source hashes
are checked or recorded. No remote custom model code is enabled.

The driver loads one BF16 model with eager attention and uses official greedy
`generate()` with cache. It retains generated scores (at most254,279,680 bytes)
until generation ends, then writes full-vocabulary F32 scores per position with
IDs, text, EOS/cap status and hashes. It does not retain copied per-step caches,
legacy full trace dictionaries or a second model. The `capture.json` stage and
`heartbeat.json` expose partial progress, timing, actual RSS/HWM and cgroup
current/peak; a terminating OOM may prevent the final receipt, so the supervising
worker should also preserve logs and final cgroup peak. Successful capture alone
does not establish decoded quality, native parity or observer-off equivalence.

Fresh admission requires22GiB available RAM, conservatively measured against
host and cgroup limits with explicit clean inactive-file reclaim accounting.
Scratch reserves double checkpoint bytes plus tokenizer and4GiB; output reserves
score bytes plus32MiB inputs and1GiB. Shared-filesystem requirements are summed.
Cache and checkpoint temporary files stay under the preflighted scratch root.
The driver rechecks resources after acquisition and fsync. These distinct BF16
rules leave historical F32's40GiB/48GiB guards and hard capture default unchanged.
No GPU is required or implied.

The [preflight receipt](../tests/results/qwen35-ocr/lightonocr3-4b-bf16-preflight.json)
preserves the initial missing-torchvision failure and corrected hosted success.
All5 synthetic methods cover cgroup v1/v2 limits/reclaim/peak, untouched F32
protocol guards, all-vocabulary score/ID/hash checks, partial write failure and
wrong shape/ID/count rejection. Official prepare-only then independently yields
grid `[1,20,28]`, pixels `[560,1536]` F32 and tokens `[1,154]` int64 with the
Qwen3VLProcessor/Qwen2VLImageProcessor torchvision backend. No checkpoint or
model forward occurred in those controls.

Hosted CPython3.12 torchvision0.28.0+cpu wheel SHA256 is
`b545d46f4d2f9d30381281cf22874bfe1d32a8a7b0ee8396fccde89f30c6a9d9`;
torch2.13.0+cpu is
`4ca4a9394b0c771238a4f73590fdbbc4debad85ed0fa63d026ae1b085da7d6e2`.
These are ABI-specific wheel receipts; another Python ABI must record its own
resolved wheel hashes. Actual CPU execution speed, peak memory, final text and
large-capture retention remain unproven until a fresh admitted worker completes.
