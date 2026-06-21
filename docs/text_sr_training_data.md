# text_sr: training-data licensing & sourcing

The `text_sr` engine (NAFNet-SR: NAFNet U-Net + PixelShuffle + bicubic residual,
see `src/text_sr.cpp`) is architecturally complete but has **no public pretrained
checkpoint** — the registry entry (`examples/cli/model_mgr.cpp`, "text-sr") ships
with an empty download URL and expects a user-supplied trained GGUF. This note
records what's available to train one, with verified licenses, so a future
effort doesn't re-derive it.

> For shipping text super-resolution **today**, you don't need any of this: the
> `tbsrn` engine already ships a real, verified, **Apache-2.0** TextZoom-trained
> model (PaddleOCR / scene-text-telescope). `text_sr` is a redundant second
> architecture; train it only if you specifically want the NAFNet backbone or
> fully self-owned training data.

## Key fact: SR doesn't need real LR/HR pairs

Super-resolution training pairs are normally produced by taking **HR** images and
**synthesizing the LR** via a degradation model (downscale + blur + noise + JPEG,
or a learned pipeline). The degradation is an *algorithm* — no data license
attaches. So you only need permissively-licensed **HR text images**, which frees
you from every non-commercial *paired* dataset (TextZoom, Real-CE).

## Licensing status of candidate sources (verified June 2026)

| Source | What | License | Commercial? |
|---|---|---|---|
| **TextZoom** | real LR/HR text pairs (ECCV'20) | **none stated** (default copyright; derived from RealSR + SR-RAW, research datasets) | ❌ no |
| **Real-CE** | real CN/EN text SR | academic / non-commercial | ❌ no |
| **SynthText** (Oxford VGG) | synthetic text-in-scene | **non-commercial only** (contact authors for commercial) | ❌ no |
| **MJSynth / Synth90k** | synthetic word images | research-only | ❌ no |
| **SynthTIGER** (clovaai) | synthetic text-image *generator* | **MIT** | ✅ yes |
| **TRDG** (Belval) | synthetic text-image *generator* | **MIT** | ✅ yes |
| **Google Fonts** | fonts for rendering | **OFL / Apache-2.0** | ✅ yes |
| **Real-ESRGAN** | degradation pipeline (code) | **BSD-3-Clause** | ✅ yes |
| **TextOCR** | real scene-text crops (on OpenImages) | **CC-BY 4.0** | ✅ w/ attribution |
| **COCO-Text** | real scene-text crops (on MS-COCO) | annotations **CC-BY 4.0** | ✅ w/ attribution* |

\* OpenImages / MS-COCO photos are **per-image mixed CC** — filter to
CC-BY / CC-BY-SA / CC0 and drop CC-BY-NC / -ND if you need commercial use.

## Recommended pipeline (fully Apache/MIT-grade, self-owned)

1. **Render HR text lines** with **SynthTIGER** or **TRDG** (both MIT) using
   **Google Fonts** (OFL/Apache). The rendered output is your work — license it
   however you like.
2. **Synthesize LR** from HR with a classical degradation (area/bicubic
   downscale + Gaussian blur + noise + JPEG) or **Real-ESRGAN/BSRGAN** degradation
   (BSD-3). Match the engine's text-line geometry — e.g. HR 32×128 → LR 16×64
   (TBSRN convention), or the `text_sr` upscale factor (x2/x4).
3. **Train NAFNet-SR** on the pairs; convert with `models/convert-text-sr-to-gguf.py`
   (`--upscale {2,4}`, ending conv must output `3·r²` channels for PixelShuffle).

For natural-image variety, optionally mix in **TextOCR** (CC-BY) or **COCO-Text**
(CC-BY) HR crops and self-degrade them — filtering the underlying photos to
commercial-OK CC types.

## Training on Kaggle, in resumable stages (model it on PosFormer)

Kaggle GPU sessions are capped (~9–12 h) and can be pre-empted, so training must
be **multi-session resumable** — exactly how `tools/kaggle/posformer-train/`
already works. Reuse that layout rather than inventing one:

- **Shared harness** `tools/kaggle/kaggle_harness.py` — line-buffered progress +
  JSONL, build toolchain, CUDA-arch detect, 3-tier HF auth, heartbeat, and a live
  `progress.jsonl` mirror pushed to an HF dataset (poll mid-run; Kaggle gates
  `/kaggle/working` until the run ends). Import it after cloning the repo.
- **Cross-session resume** (the key trick, from `posformer_train.py`): a
  per-hour callback uploads the checkpoint to an HF repo as **`latest.ckpt`**
  (for resume) *and* a timestamped copy (history); on startup the trainer
  `hf_hub_download`s `latest.ckpt` and continues. So each Kaggle run trains until
  it nears the wall, uploads, exits; the next run resumes. Survives crashes/OOM.
- **Staged** like PosFormer's `pretrain` → `finetune` subcommands: stage each as
  its own resumable run.
- `kernel-metadata.json` per kernel; a data-prep script (cf.
  `prepare_mixed_data.py`) and an `eval_checkpoint.py`.

Suggested stages for a NAFNet text-SR model:
1. **Data-gen** (CPU kernel, or local): render HR text lines with SynthTIGER/TRDG
   + Google Fonts, self-degrade to LR (§ pipeline above); publish as a Kaggle
   Dataset (or generate on the fly each epoch). Geometry per the engine (e.g.
   HR 32×128 → LR 16×64, or the chosen `--upscale`).
2. **Pretrain** on the synthetic set (GPU kernel; hourly HF checkpoint + resume to
   `cstr/text-sr-training-checkpoints`).
3. **Finetune** (optional) on CC-BY real crops (TextOCR / COCO-Text), resumable.
4. **Convert + upload**: `models/convert-text-sr-to-gguf.py --upscale {2,4}` →
   GGUF; verify with the parity harness below; register the URL in
   `examples/cli/model_mgr.cpp`.

NAFNet-SR is small (the `text_sr` forward is ~40–60 GFLOP/tile vs PosFormer's much
heavier encoder-decoder), so expect far fewer sessions than PosFormer needed.

## Verify the engine without a trained model

Independent of training, the `text_sr` C++ engine math can be parity-checked with
**synthetic random weights**: build a random-init `NAFNetSR` matching the arch
(`tools/dump_text_sr_reference.py` already has the torch `NAFNetSR` +
`forward_with_intermediates`), convert → GGUF, dump the reference, add a
`test_text_sr_diff.cpp` target, and confirm cosine parity. Output is visually
meaningless but parity validates the implementation (the only new math vs the
already-verified `nafnet_denoise` backbone is PixelShuffle + bicubic, both also
exercised by the verified `pan`/`tbsrn` engines).

## Sources

- TextZoom: <https://github.com/WenjiaWang0312/TextZoom> (no LICENSE file);
  paper <https://arxiv.org/abs/2005.03341>
- SynthTIGER (MIT): <https://github.com/clovaai/synthtiger>
- TRDG (MIT): <https://github.com/Belval/TextRecognitionDataGenerator>
- SynthText (non-commercial): <https://www.robots.ox.ac.uk/~vgg/data/scenetext/>
- TextOCR (CC-BY 4.0): <https://arxiv.org/pdf/2105.05486>
- OpenImages per-image CC licenses: <https://learn.library.torontomu.ca/openimages/licenses>
- COCO-Text (CC-BY 4.0 annotations): <https://vision.cornell.edu/se3/coco-text-2/>
- Real-ESRGAN (BSD-3): <https://github.com/xinntao/Real-ESRGAN>
