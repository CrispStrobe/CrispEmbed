# Pinned OvisOCR2 original vLLM reference

This private Kaggle script records one genuine T4 execution of the pinned
[OvisOCR2 example](https://huggingface.co/ATH-MaaS/OvisOCR2/blob/cbdede82e048659b83310476ff807da825f8fe78/README.md)
after the corrected frontend and original metadata CI runs succeed. It uses
vLLM 0.22.1's published CUDA 12.9 wheel, pinned by release SHA-256, with
Torch 2.11.0, Transformers 5.16.1 and tokenizers 0.23.3. The v0.22.1 release
does not publish a CUDA 12.8 wheel. The installed local toolkit version does
not establish whether a CUDA wheel can run: the script checks the actual T4
and driver against NVIDIA's [CUDA 12.9 GA corresponding driver
version](https://docs.nvidia.com/cuda/archive/12.9.0/cuda-toolkit-release-notes/index.html)
before installing packages or downloading weights. A lower driver yields an
inconclusive result with no device redraw or wheel fallback.

Provide `GITHUB_TOKEN` as a Kaggle secret to read the original GitHub Actions
artifact. Do not put token values or account names in this directory. The
recipe uses the normal `${KAGGLE_ACCOUNT}` placeholder in its kernel metadata.
The top-level script writes its phase journal, command log hashes, and small
processor replay files to Kaggle output. The checkpoint and wheel stay in
remote temporary storage.

The README omits `dtype`, so vLLM's `auto` policy selects FP16 on T4 from the
model's BF16 config. This is the original example's T4 route, **not a BF16
parity result**. The script keeps the README's Triton GDN request and
multimodal kwargs. Its one greedy generation is explicitly capped at 256
tokens instead of the README's 16,384, and the image is a frozen synthetic
processor fixture. Raw IDs, raw text and the README's processed text are
retained; no OCR quality or native-model parity claim follows from this run.

The script is prepared for review. A successful frontend prerequisite and an
available GPU slot are required before submission. Run `kpush.py` through the
repository's normal private operator procedure; this directory carries no
credentials.
