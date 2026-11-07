# Amendment to inpaint-publication-v1 (7 November 2026)

`docs/inpaint-publication-v1.json` is a frozen provenance record and is not edited. Its
PixelHacker and SDXL-VAE entries name the private staging repo
`cstr/crispembed-pixelhacker-private` as `source_repo`. That private repo was retired on
7 November 2026 (owner-approved storage cleanup) after verifying:

- all 4 PixelHacker GGUFs (f32/f16/q8_0/q4_k) are byte-identical (LFS sha256) in the public
  [`cstr/PixelHacker-GGUF`](https://huggingface.co/cstr/PixelHacker-GGUF);
- all 4 SDXL-VAE GGUFs are byte-identical in the public
  [`cstr/SDXL-VAE-inpaint-GGUF`](https://huggingface.co/cstr/SDXL-VAE-inpaint-GGUF);
- the converter sidecar metadata (`*.gguf.json`) that existed only privately was mirrored to
  `sidecars/` in those public repos (PixelHacker-GGUF commit `91f092df`, SDXL-VAE-inpaint-GGUF
  commit `7bfaaa58`), each verified by sha256 readback; the private `SOURCE.txt` was
  byte-identical to the public `licenses/SOURCE.txt`.

Use the public repos as the source for any re-run. No published file or hash changed.
