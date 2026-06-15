#!/usr/bin/env python3
"""Convert Tesseract LSTM .traineddata to GGUF.

Parses the binary .traineddata archive, extracts the LSTM network weights,
unicharset, and recoder, and emits a GGUF file for CrispEmbed's Tesseract
LSTM OCR engine.

Supports both tessdata_fast (int8 weights + per-row scales) and
tessdata_best (float64 weights). Int8 weights are dequantized to float32
so that crispembed-quantize can re-quantize to ggml's Q4_K/Q8_0.

Architecture (typical English model):
  Input 1x36x(var)x1 → Convolve 3x3 stacking → FC+tanh 16 →
  MaxPool 3x3 → XYTranspose → SummLSTM(48) → XYTranspose →
  LSTM_fwd(96) → LSTM_rev(96) → LSTM_fwd(256) → Softmax(111)

Usage:
    pip install gguf numpy
    python models/convert-tesseract-to-gguf.py \\
        --model /usr/share/tesseract-ocr/5/tessdata/eng.traineddata \\
        --output /mnt/storage/gguf-models/tesseract-eng-f32.gguf
"""

import argparse
import struct
import sys
from pathlib import Path

import gguf
import numpy as np


# ---------------------------------------------------------------------------
# Binary parsing helpers
# ---------------------------------------------------------------------------

class TessReader:
    """Read Tesseract's binary serialization format."""

    def __init__(self, buf: bytes):
        self.buf = buf
        self.pos = 0

    def read_i8(self):
        v = struct.unpack_from('<b', self.buf, self.pos)[0]
        self.pos += 1
        return v

    def read_u8(self):
        v = self.buf[self.pos]
        self.pos += 1
        return v

    def read_i32(self):
        v = struct.unpack_from('<i', self.buf, self.pos)[0]
        self.pos += 4
        return v

    def read_u32(self):
        v = struct.unpack_from('<I', self.buf, self.pos)[0]
        self.pos += 4
        return v

    def read_f64(self):
        v = struct.unpack_from('<d', self.buf, self.pos)[0]
        self.pos += 8
        return v

    def read_string(self):
        slen = self.read_u32()
        s = self.buf[self.pos:self.pos + slen].decode('utf-8', errors='replace')
        self.pos += slen
        return s

    def read_bytes(self, n):
        b = self.buf[self.pos:self.pos + n]
        self.pos += n
        return b


# ---------------------------------------------------------------------------
# Traineddata archive parsing
# ---------------------------------------------------------------------------

COMPONENT_NAMES = [
    "config", "unicharset", "ambigs", "inttemp", "pffmtable",
    "normproto", "punc-dawg", "system-dawg", "number-dawg", "freq-dawg",
    "fixed-length-dawg", "cube-unicharset", "cube-word-dawg", "shapetable",
    "bigram-dawg", "unambig-dawg", "params-model", "lstm",
    "lstm-punc-dawg", "lstm-system-dawg", "lstm-number-dawg",
    "lstm-unicharset", "lstm-recoder", "version",
]


def parse_traineddata(data: bytes) -> dict:
    """Parse .traineddata archive, return dict of component name → bytes."""
    n_entries = struct.unpack_from('<i', data, 0)[0]
    offsets = []
    for i in range(n_entries):
        off = struct.unpack_from('<q', data, 4 + i * 8)[0]
        offsets.append(off)

    components = {}
    for i in range(n_entries):
        if offsets[i] == -1:
            continue
        # Find next component to determine size
        next_off = len(data)
        for j in range(i + 1, n_entries):
            if offsets[j] != -1:
                next_off = offsets[j]
                break
        size = next_off - offsets[i]
        name = COMPONENT_NAMES[i] if i < len(COMPONENT_NAMES) else f"unknown-{i}"
        components[name] = data[offsets[i]:offsets[i] + size]

    return components


# ---------------------------------------------------------------------------
# Unicharset parsing
# ---------------------------------------------------------------------------

def parse_unicharset(data: bytes) -> list:
    """Parse lstm-unicharset → list of UTF-8 strings indexed by ID."""
    text = data.decode('utf-8', errors='replace')
    lines = text.strip().split('\n')
    # First line: count
    count = int(lines[0].strip())
    tokens = []
    for line in lines[1:count + 1]:
        parts = line.split()
        if parts:
            tok = parts[0]
            # Handle special names
            if tok == "NULL":
                tok = ""  # CTC blank
            tokens.append(tok)
    return tokens


# ---------------------------------------------------------------------------
# Recoder parsing
# ---------------------------------------------------------------------------

def parse_recoder(data: bytes) -> list:
    """Parse lstm-recoder → list of [code_ids] per unichar.

    The recoder maps unicharset indices to sequences of output class indices.
    encoder_[unichar_id] = RecodedCharID (a short sequence of output codes).

    Format: vector<RecodedCharID> where each element is:
      int8_t  self_normalized
      int32_t length
      length × int32_t code values
    """
    r = TessReader(data)
    n_entries = r.read_u32()  # number of unichar entries
    entries = []
    for _ in range(n_entries):
        _self_norm = r.read_i8()
        code_len = r.read_i32()
        codes = [r.read_i32() for _ in range(code_len)]
        entries.append(codes)
    return entries


# ---------------------------------------------------------------------------
# Network type names (must match Tesseract's kTypeNames order)
# ---------------------------------------------------------------------------

NET_TYPES = [
    "Invalid", "Input", "Convolve", "Maxpool", "Parallel", "Replicated",
    "ParBidiLSTM", "DepParUDLSTM", "Par2dLSTM", "Series", "Reconfig",
    "RTLReversed", "TTBReversed", "XYTranspose", "LSTM", "SummLSTM",
    "Logistic", "LinLogistic", "LinTanh", "Tanh", "Relu", "Linear",
    "Softmax", "SoftmaxNoCTC", "LSTMSoftmax", "LSTMBinarySoftmax",
]

# FullyConnected-based types (have a WeightMatrix)
FC_TYPES = {"Softmax", "SoftmaxNoCTC", "Logistic", "LinLogistic",
            "LinTanh", "Tanh", "Relu", "Linear"}

NF_LAYER_SPECIFIC_LR = 64


# ---------------------------------------------------------------------------
# WeightMatrix parsing
# ---------------------------------------------------------------------------

def read_weight_matrix(r: TessReader):
    """Read a WeightMatrix → dict with 'weight' (np array) and 'bias' (np array)."""
    mode = r.read_u8()
    int_mode = (mode & 1) != 0       # kInt8Flag
    use_adam = (mode & 4) != 0       # kAdamFlag
    double_flag = (mode & 128) != 0  # kDoubleFlag

    if not double_flag:
        # Old format (pre-double)
        if int_mode:
            dim1 = r.read_i32(); dim2 = r.read_i32(); _empty = r.read_i8()
            raw = np.frombuffer(r.read_bytes(dim1 * dim2), dtype=np.int8).reshape(dim1, dim2)
            n_scales = r.read_u32()
            scales = np.array([struct.unpack_from('<f', r.buf, r.pos + i*4)[0]
                               for i in range(n_scales)])
            r.pos += n_scales * 4
            weights_f32 = raw.astype(np.float32) * scales[:, np.newaxis]
        else:
            dim1 = r.read_i32(); dim2 = r.read_i32()
            r.read_bytes(4)  # empty float
            weights_f32 = np.frombuffer(r.read_bytes(dim1*dim2*4),
                                        dtype=np.float32).copy().reshape(dim1, dim2)
    else:
        # New format (double): scales and float arrays use double
        if int_mode:
            dim1 = r.read_i32(); dim2 = r.read_i32(); _empty = r.read_i8()
            raw = np.frombuffer(r.read_bytes(dim1 * dim2), dtype=np.int8).reshape(dim1, dim2)
            n_scales = r.read_u32()
            scales_raw = np.array([r.read_f64() for _ in range(n_scales)])
            # Dequant: float_w = int8_w * stored_scale (raw value before /127)
            # because runtime does: dot(int8_w, int8_input) * loaded_scale
            # = dot(int8_w, float_x * 127) * (stored / 127) = dot(int8_w, float_x) * stored
            weights_f32 = raw.astype(np.float32) * scales_raw[:, np.newaxis]
        else:
            dim1 = r.read_i32(); dim2 = r.read_i32()
            _empty = r.read_f64()  # empty double
            weights_f64 = np.frombuffer(r.read_bytes(dim1 * dim2 * 8),
                                        dtype=np.float64).reshape(dim1, dim2)
            weights_f32 = weights_f64.astype(np.float32)

    # Split weight and bias. The last column is the bias.
    weight = weights_f32[:, :-1]  # (no, ni)
    bias = weights_f32[:, -1]     # (no,)

    return {"weight": weight, "bias": bias, "shape": (dim1, dim2),
            "int_mode": int_mode if int_mode else False}


# ---------------------------------------------------------------------------
# Recursive network parsing
# ---------------------------------------------------------------------------

def read_network_header(r: TessReader):
    """Read Network base class header fields."""
    type_byte = r.read_i8()
    if type_byte == 0:  # NT_NONE → type string follows
        type_name = r.read_string()
    else:
        type_name = NET_TYPES[type_byte] if 0 < type_byte < len(NET_TYPES) else f"unknown_{type_byte}"

    training = r.read_i8()
    needs_bp = r.read_i8()
    net_flags = r.read_i32()
    ni = r.read_i32()
    no = r.read_i32()
    num_w = r.read_i32()
    name = r.read_string()

    return {
        "type": type_name, "ni": ni, "no": no, "num_weights": num_w,
        "name": name, "training": training, "flags": net_flags,
    }


def parse_network(r: TessReader, depth=0):
    """Recursively parse the network tree, return topology + weights."""
    hdr = read_network_header(r)
    t = hdr["type"]
    indent = "  " * depth

    node = {**hdr, "children": [], "weights": {}}

    if t == "Input":
        node["batch"] = r.read_i32()
        node["height"] = r.read_i32()
        node["width"] = r.read_i32()
        node["depth"] = r.read_i32()
        node["loss"] = r.read_i32()
        print(f"{indent}Input: {node['batch']}x{node['height']}x{node['width']}x{node['depth']}")

    elif t in ("Series", "Parallel"):
        count = r.read_u32()
        print(f"{indent}{t} '{hdr['name']}': {count} children, ni={hdr['ni']} no={hdr['no']}")
        for _ in range(count):
            child = parse_network(r, depth + 1)
            node["children"].append(child)
        # Read learning rates if NF_LAYER_SPECIFIC_LR is set
        if hdr["flags"] & NF_LAYER_SPECIFIC_LR:
            lr_size = r.read_u32()
            r.read_bytes(lr_size * 4)  # vector<float>

    elif t == "Convolve":
        half_x = r.read_i32()
        half_y = r.read_i32()
        node["half_x"] = half_x
        node["half_y"] = half_y
        print(f"{indent}Convolve: kernel {2*half_x+1}x{2*half_y+1}")

    elif t == "Maxpool":
        # Maxpool extends Reconfig
        x_scale = r.read_i32()
        y_scale = r.read_i32()
        node["x_scale"] = x_scale
        node["y_scale"] = y_scale
        print(f"{indent}Maxpool: {x_scale}x{y_scale}")

    elif t == "Reconfig":
        x_scale = r.read_i32()
        y_scale = r.read_i32()
        node["x_scale"] = x_scale
        node["y_scale"] = y_scale
        print(f"{indent}Reconfig: {x_scale}x{y_scale}")

    elif t in ("LSTM", "SummLSTM"):
        na = r.read_i32()
        ns = hdr["no"]
        node["na"] = na
        node["ns"] = ns
        print(f"{indent}{t} '{hdr['name']}': na={na} ns={ns} ni={hdr['ni']}")
        # Read 4 gate weight matrices: CI, GI, GF1, GO
        gate_names = ["CI", "GI", "GF1", "GO"]
        for gn in gate_names:
            wm = read_weight_matrix(r)
            node["weights"][gn] = wm
            print(f"{indent}  {gn}: ({wm['weight'].shape[0]}, {wm['weight'].shape[1]}) + bias")

    elif t in ("RTLReversed", "TTBReversed", "XYTranspose"):
        # Reversed extends Plumbing — reads count + child networks
        count = r.read_u32()
        print(f"{indent}{t} '{hdr['name']}': {count} children")
        for _ in range(count):
            child = parse_network(r, depth + 1)
            node["children"].append(child)
        if hdr["flags"] & NF_LAYER_SPECIFIC_LR:
            lr_size = r.read_u32()
            r.read_bytes(lr_size * 4)

    elif t in FC_TYPES:
        wm = read_weight_matrix(r)
        node["weights"]["fc"] = wm
        print(f"{indent}{t} '{hdr['name']}': weight ({wm['weight'].shape[0]}, {wm['weight'].shape[1]})")

    else:
        print(f"{indent}WARNING: unhandled type '{t}' — output may be corrupt")

    return node


# ---------------------------------------------------------------------------
# Flatten network tree into ordered layer list
# ---------------------------------------------------------------------------

def flatten_layers(node, layers=None, lstm_idx=None):
    """Walk the tree depth-first, collect weight-bearing layers in order."""
    if layers is None:
        layers = []
    if lstm_idx is None:
        lstm_idx = [0]

    t = node["type"]

    if t in ("Series", "Parallel"):
        for child in node["children"]:
            flatten_layers(child, layers, lstm_idx)

    elif t in ("RTLReversed", "TTBReversed", "XYTranspose"):
        for child in node["children"]:
            if t in ("RTLReversed", "TTBReversed"):
                child["_reversed"] = True
            flatten_layers(child, layers, lstm_idx)

    elif t in ("LSTM", "SummLSTM"):
        idx = lstm_idx[0]
        lstm_idx[0] += 1
        node["_layer_idx"] = idx
        node["_is_reversed"] = node.get("_reversed", False)
        layers.append(node)

    elif t in FC_TYPES:
        if node["name"] in ("ConvNL",):
            node["_role"] = "conv"
        else:
            node["_role"] = "output"
        layers.append(node)

    return layers


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Convert Tesseract .traineddata to GGUF")
    p.add_argument("--model", required=True,
                   help="Path to .traineddata file")
    p.add_argument("--output", required=True, help="Output GGUF path")
    p.add_argument("--fp16", action="store_true",
                   help="Store weights in FP16 (halves file size)")
    args = p.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: {model_path} not found")
        sys.exit(1)

    data = model_path.read_bytes()
    print(f"Input: {model_path.name} ({len(data)} bytes)")

    # -----------------------------------------------------------------------
    # Parse traineddata archive
    # -----------------------------------------------------------------------
    components = parse_traineddata(data)
    print(f"\nComponents: {', '.join(components.keys())}")

    if "lstm" not in components:
        print("ERROR: no lstm component found (not an LSTM model?)")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Parse unicharset
    # -----------------------------------------------------------------------
    tokens = []
    if "lstm-unicharset" in components:
        tokens = parse_unicharset(components["lstm-unicharset"])
        print(f"Unicharset: {len(tokens)} tokens")
    else:
        print("WARNING: no lstm-unicharset — tokens will be missing")

    # -----------------------------------------------------------------------
    # Parse recoder
    # -----------------------------------------------------------------------
    recoder = []
    if "lstm-recoder" in components:
        recoder = parse_recoder(components["lstm-recoder"])
        print(f"Recoder: {len(recoder)} output codes")
    else:
        print("WARNING: no lstm-recoder — using identity mapping")

    # -----------------------------------------------------------------------
    # Parse LSTM network
    # -----------------------------------------------------------------------
    print("\n--- Network topology ---")
    r = TessReader(components["lstm"])
    root = parse_network(r)

    # Read LSTMRecognizer metadata after the network tree
    vgsl_spec = r.read_string()
    training_flags = r.read_i32()
    training_iteration = r.read_i32()
    sample_iteration = r.read_i32()
    null_char = r.read_i32()

    # adam_beta, learning_rate, momentum: stored as float32 in older versions
    # (tessdata 4.00.00alpha), float64 in newer versions.
    remaining = len(r.buf) - r.pos
    if remaining >= 24:
        # 3 × f64 = 24 bytes (modern format)
        adam_beta = r.read_f64()
        learning_rate = r.read_f64()
        momentum = r.read_f64()
    elif remaining >= 12:
        # 3 × f32 = 12 bytes (old format, e.g. tessdata 4.00.00alpha)
        adam_beta = struct.unpack_from('<f', r.buf, r.pos)[0]; r.pos += 4
        learning_rate = struct.unpack_from('<f', r.buf, r.pos)[0]; r.pos += 4
        momentum = struct.unpack_from('<f', r.buf, r.pos)[0]; r.pos += 4
    else:
        adam_beta = 0.999
        learning_rate = 0.001
        momentum = 0.5

    print(f"\nVGSL spec: {vgsl_spec}")
    print(f"Null char (CTC blank): {null_char}")
    print(f"Parsed {r.pos}/{len(components['lstm'])} bytes of lstm component")

    # -----------------------------------------------------------------------
    # Extract input shape from Input node
    # -----------------------------------------------------------------------
    def find_input(node):
        if node["type"] == "Input":
            return node
        for child in node.get("children", []):
            result = find_input(child)
            if result:
                return result
        return None

    input_node = find_input(root)
    if input_node is None:
        print("ERROR: no Input node found")
        sys.exit(1)

    input_height = input_node["height"]
    print(f"Input height: {input_height}")

    # -----------------------------------------------------------------------
    # Flatten layers and collect weights
    # -----------------------------------------------------------------------
    layers = flatten_layers(root)
    print(f"\nWeight-bearing layers: {len(layers)}")

    # -----------------------------------------------------------------------
    # Write GGUF
    # -----------------------------------------------------------------------
    writer = gguf.GGUFWriter(str(args.output), arch="tesseract_lstm")

    # Metadata
    writer.add_string("general.name",
                      f"tesseract-lstm-{model_path.stem}")
    writer.add_string("general.license", "Apache-2.0")
    writer.add_string("general.source",
                      "https://github.com/tesseract-ocr/tessdata_best")

    # Network hyperparameters
    writer.add_string("tesseract_lstm.vgsl_spec", vgsl_spec)
    writer.add_uint32("tesseract_lstm.input_height", input_height)
    writer.add_uint32("tesseract_lstm.null_char", null_char)
    writer.add_uint32("tesseract_lstm.num_classes", root["no"])

    # Tokenizer (unicharset)
    if tokens:
        writer.add_array("tokenizer.tokens", tokens)
        writer.add_uint32("tokenizer.count", len(tokens))

    # Recoder: maps network output class → unicharset ID(s)
    # For simple (non-composed) characters, recoder[i] = [unichar_id]
    # Store as flat array + offsets for composed characters
    if recoder:
        # Simple mapping: most codes are single unichar
        recoder_flat = []
        recoder_offsets = []
        for code in recoder:
            recoder_offsets.append(len(recoder_flat))
            recoder_flat.extend(code)
        recoder_offsets.append(len(recoder_flat))  # sentinel
        writer.add_array("tesseract_lstm.recoder_map",
                         np.array(recoder_flat, dtype=np.int32).tolist())
        writer.add_array("tesseract_lstm.recoder_offsets",
                         np.array(recoder_offsets, dtype=np.int32).tolist())

        # Reverse mapping: output_class → unichar_id (for CTC decode)
        n_classes = root["no"]
        rev = np.full(n_classes, -1, dtype=np.int32)
        for uid, codes in enumerate(recoder):
            if len(codes) == 1 and rev[codes[0]] == -1:
                rev[codes[0]] = uid
        writer.add_array("tesseract_lstm.output_to_unichar",
                         rev.tolist())

    # Count LSTM layers for metadata
    lstm_layers = [l for l in layers if l["type"] in ("LSTM", "SummLSTM")]
    n_lstm = len(lstm_layers)
    writer.add_uint32("tesseract_lstm.num_lstm_layers", n_lstm)

    # Per-LSTM-layer metadata
    lstm_hidden_sizes = []
    lstm_types = []  # "fwd", "rev", "y_sum"
    for l in lstm_layers:
        lstm_hidden_sizes.append(l["ns"])
        if l["type"] == "SummLSTM":
            lstm_types.append("y_sum")
        elif l.get("_is_reversed", False):
            lstm_types.append("rev")
        else:
            lstm_types.append("fwd")

    writer.add_array("tesseract_lstm.lstm_hidden_sizes", lstm_hidden_sizes)
    writer.add_array("tesseract_lstm.lstm_types", lstm_types)

    dtype_np = np.float16 if args.fp16 else np.float32
    dtype_gguf = gguf.GGMLQuantizationType.F16 if args.fp16 else gguf.GGMLQuantizationType.F32

    total_params = 0
    tensor_count = 0

    def add_tensor(name, data_arr):
        nonlocal total_params, tensor_count
        d = data_arr.astype(dtype_np)
        total_params += d.size
        writer.add_tensor(name, d, raw_dtype=dtype_gguf)
        tensor_count += 1

    # -----------------------------------------------------------------------
    # Write tensors
    # -----------------------------------------------------------------------
    print("\n--- Tensors ---")

    for layer in layers:
        t = layer["type"]

        if t in FC_TYPES:
            role = layer.get("_role", "output")
            if role == "conv":
                prefix = "conv"
            else:
                prefix = "output"
            wm = layer["weights"]["fc"]

            # Reorder gate weights: Tesseract order CI,GI,GF1,GO maps to
            # cell_input, input_gate, forget_gate, output_gate.
            # Our LSTM cell expects PyTorch order: i, f, g, o
            # Tesseract:  CI=g, GI=i, GF1=f, GO=o
            # No reordering needed for FC layers.
            add_tensor(f"{prefix}.weight", wm["weight"])
            add_tensor(f"{prefix}.bias", wm["bias"])
            print(f"  {prefix}.weight: {wm['weight'].shape}")

        elif t in ("LSTM", "SummLSTM"):
            idx = layer["_layer_idx"]
            prefix = f"lstm.{idx}"

            # Tesseract gate order: CI (cell input), GI (input gate),
            # GF1 (forget gate), GO (output gate)
            # PyTorch order: i (input), f (forget), g (cell input), o (output)
            # Mapping: PyTorch_i = Tess_GI, PyTorch_f = Tess_GF1,
            #          PyTorch_g = Tess_CI, PyTorch_o = Tess_GO
            #
            # We reorder here to match PyTorch convention, so the C++ engine
            # can reuse lstm_forward_one_dir() from gliner_ner.cpp unchanged.
            gate_reorder = [
                ("GI",  "i"),   # input gate
                ("GF1", "f"),   # forget gate
                ("CI",  "g"),   # cell input
                ("GO",  "o"),   # output gate
            ]

            # Stack all 4 gates into a single (4*ns, na) weight and (4*ns,) bias
            # matching PyTorch's packed LSTM format
            ws = []
            bs = []
            for tess_name, _ in gate_reorder:
                wm = layer["weights"][tess_name]
                ws.append(wm["weight"])
                bs.append(wm["bias"])

            # Each gate weight is (ns, na) where na = ni + ns
            # Stack to (4*ns, na) — matches PyTorch weight_ih/weight_hh when split
            stacked_w = np.concatenate(ws, axis=0)  # (4*ns, na)
            stacked_b = np.concatenate(bs, axis=0)  # (4*ns,)

            ns = layer["ns"]
            na = stacked_w.shape[1]
            ni_in = na - ns  # input features

            # Split into weight_ih (4*ns, ni) and weight_hh (4*ns, ns)
            weight_ih = stacked_w[:, :ni_in]  # input-to-hidden
            weight_hh = stacked_w[:, ni_in:]  # hidden-to-hidden

            add_tensor(f"{prefix}.weight_ih", weight_ih)
            add_tensor(f"{prefix}.weight_hh", weight_hh)
            add_tensor(f"{prefix}.bias", stacked_b)

            ltype = lstm_types[idx]
            print(f"  {prefix}: {ltype}, ns={ns}, ni={ni_in}, "
                  f"weight_ih={weight_ih.shape}, weight_hh={weight_hh.shape}")

    # -----------------------------------------------------------------------
    # Finalize
    # -----------------------------------------------------------------------
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    out_size = Path(args.output).stat().st_size
    print(f"\nWrote {args.output}")
    print(f"  Tensors: {tensor_count}")
    print(f"  Parameters: {total_params:,}")
    print(f"  File size: {out_size:,} bytes ({out_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
