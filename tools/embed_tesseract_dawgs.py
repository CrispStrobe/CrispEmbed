#!/usr/bin/env python3
"""Copy a tesseract GGUF verbatim, appending the LSTM DAWG components from a
.traineddata archive as tesseract_lstm.dawg_names + uint8 payload arrays (the
runtime's CRISPEMBED_TESSERACT_DAWG_LOAD channel).

Exists because reconverting is not an option for artifacts with post-convert
treatment (the q8-seeded family): tensors here are copied byte-exactly, only
the dawg metadata is appended. The payload MUST be written as GGUF subtype
UINT8 — the runtime's kv_u8_array() rejects any other subtype and silently
loads 0 graphs (a plain python int list becomes INT32; raw bytes become
UINT8 in the writer's ARRAY packer).

Usage:
  python tools/embed_tesseract_dawgs.py \
      --source tesseract-frk-q8_0-seeded.gguf \
      --traineddata /opt/homebrew/share/tessdata/frk.traineddata \
      --output tesseract-frk-q8_0-seeded-dawg.gguf
"""

import argparse
import importlib.util
import sys
from pathlib import Path

from gguf import GGUFReader, GGUFWriter

_spec = importlib.util.spec_from_file_location(
    "tess_converter", Path(__file__).parents[1] / "models" / "convert-tesseract-to-gguf.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
parse_traineddata = _mod.parse_traineddata

LSTM_DAWG_COMPONENTS = ("lstm-punc-dawg", "lstm-system-dawg", "lstm-number-dawg")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="existing tesseract GGUF (kept byte-identical)")
    p.add_argument("--traineddata", required=True, help=".traineddata carrying the lstm dawgs")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    components = parse_traineddata(Path(args.traineddata).read_bytes())
    dawgs = {n: components[n] for n in LSTM_DAWG_COMPONENTS if n in components}
    if not dawgs:
        sys.exit("no LSTM dawg components in traineddata")
    for n, b in dawgs.items():
        print(f"  {n}: {len(b)} bytes")

    reader = GGUFReader(args.source)
    if any("dawg" in k for k in reader.fields):
        sys.exit("source already has dawg keys; refusing")

    arch = reader.fields["general.architecture"].contents()
    writer = GGUFWriter(args.output, arch)

    for field in reader.fields.values():
        if field.name in ("GGUF.version", "GGUF.tensor_count", "GGUF.kv_count",
                          "general.architecture"):
            continue
        contents = field.contents()
        if isinstance(contents, list) and not contents:
            # gguf's writer refuses empty arrays; the runtime treats a missing
            # key and an empty array identically (kv_*_array -> empty).
            print(f"  skipping empty array key {field.name}")
            continue
        writer.add_key_value(field.name, contents, field.types[0])

    embedded = []
    for name in LSTM_DAWG_COMPONENTS:
        if name in dawgs:
            writer.add_array(f"tesseract_lstm.dawg.{name}", dawgs[name])
            embedded.append(name)
    writer.add_array("tesseract_lstm.dawg_names", embedded)
    writer.add_bool("tesseract_lstm.dawg_embedded", bool(embedded))

    for tensor in reader.tensors:
        writer.add_tensor_info(tensor.name, tensor.data.shape, tensor.data.dtype,
                               tensor.data.nbytes, tensor.tensor_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for tensor in reader.tensors:
        writer.write_tensor_data(tensor.data)
    writer.close()

    check = GGUFReader(args.output)
    sub = check.fields["tesseract_lstm.dawg.lstm-system-dawg"].types[1].name \
        if "tesseract_lstm.dawg.lstm-system-dawg" in check.fields else "MISSING"
    print(f"wrote {args.output} (system-dawg subtype: {sub})")
    if sub != "UINT8":
        sys.exit("payload subtype is not UINT8 — runtime would load 0 graphs")


if __name__ == "__main__":
    main()
