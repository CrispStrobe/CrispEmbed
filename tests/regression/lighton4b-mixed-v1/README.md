# Frozen independent LightOn4B mixed-recipe inputs

Four small PNGs are frozen before model outputs by hosted
[42004846562](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42004846562)
at `98e4b7d71`. The unchanged [manifest](frozen-inputs.json) contains exact
source/PNG/RGB hashes, dimensions, split and expected source-derived processor
geometry. Independent review rehashes all four encoded PNG and raw RGB outputs;
no model was downloaded or executed. The model worker must verify all four RGB
hashes and official processor geometry/token counts before loading a model.

PNG paths are relative to this directory. Raw `.rgb` files remain only in the
hosted artifact; reconstruct their bytes by decoding the committed opaque RGB
PNG and check the manifest SHA. Do not repeat source resizing with another
Pillow version. Source licenses/public pages and transforms are fixed in
[protocolv1](../../../docs/lighton4b-mixed-calibration-v1.json). Every source is
CC0-1.0. These fixtures and case splits must not change after outputs are read.
