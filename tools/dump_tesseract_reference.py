#!/usr/bin/env python3
"""Dump Tesseract LSTM per-stage reference activations for parity testing.

Parses a .traineddata file, loads weights, runs a pure-numpy forward pass
on a grayscale text line image, and captures intermediate activations at
each stage. Writes to GGUF for crispembed_diff comparison.

Also runs `tesseract --oem 1 --psm 7` on the same image to get the
ground-truth text output for end-to-end CER comparison.

Usage:
    python tools/dump_tesseract_reference.py \
        --model /usr/share/tesseract-ocr/5/tessdata/eng.traineddata \
        --image /tmp/test-line.png \
        --output /tmp/tess-ref.gguf
"""

import argparse
import subprocess
import struct
import sys
from pathlib import Path

import gguf
import numpy as np

# ---------------------------------------------------------------------------
# Import converter's parsing code
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
from importlib import import_module


# ---------------------------------------------------------------------------
# Traineddata parsing (duplicated from converter for self-containment)
# ---------------------------------------------------------------------------

class TessReader:
    def __init__(self, buf):
        self.buf = buf
        self.pos = 0
    def read_i8(self):
        v = struct.unpack_from('<b', self.buf, self.pos)[0]; self.pos += 1; return v
    def read_u8(self):
        v = self.buf[self.pos]; self.pos += 1; return v
    def read_i32(self):
        v = struct.unpack_from('<i', self.buf, self.pos)[0]; self.pos += 4; return v
    def read_u32(self):
        v = struct.unpack_from('<I', self.buf, self.pos)[0]; self.pos += 4; return v
    def read_f64(self):
        v = struct.unpack_from('<d', self.buf, self.pos)[0]; self.pos += 8; return v
    def read_string(self):
        slen = self.read_u32()
        s = self.buf[self.pos:self.pos + slen].decode('utf-8', errors='replace')
        self.pos += slen
        return s
    def read_bytes(self, n):
        b = self.buf[self.pos:self.pos + n]; self.pos += n; return b


def parse_traineddata(data):
    n = struct.unpack_from('<i', data, 0)[0]
    offsets = [struct.unpack_from('<q', data, 4 + i*8)[0] for i in range(n)]
    NAMES = ["config","unicharset","ambigs","inttemp","pffmtable",
             "normproto","punc-dawg","system-dawg","number-dawg","freq-dawg",
             "fixed-length-dawg","cube-unicharset","cube-word-dawg","shapetable",
             "bigram-dawg","unambig-dawg","params-model","lstm",
             "lstm-punc-dawg","lstm-system-dawg","lstm-number-dawg",
             "lstm-unicharset","lstm-recoder","version"]
    comps = {}
    for i in range(n):
        if offsets[i] == -1: continue
        nxt = len(data)
        for j in range(i+1, n):
            if offsets[j] != -1: nxt = offsets[j]; break
        nm = NAMES[i] if i < len(NAMES) else f"unk-{i}"
        comps[nm] = data[offsets[i]:nxt]
    return comps


def parse_unicharset(data):
    lines = data.decode('utf-8', errors='replace').strip().split('\n')
    count = int(lines[0].strip())
    tokens = []
    for line in lines[1:count+1]:
        parts = line.split()
        tok = parts[0] if parts else ""
        if tok == "NULL": tok = ""
        tokens.append(tok)
    return tokens


def parse_recoder(data):
    r = TessReader(data)
    n = r.read_u32()
    entries = []
    for _ in range(n):
        r.read_i8()  # self_normalized
        code_len = r.read_i32()
        codes = [r.read_i32() for _ in range(code_len)]
        entries.append(codes)
    return entries


# ---------------------------------------------------------------------------
# Network parsing (extract weights + topology)
# ---------------------------------------------------------------------------

NET_TYPES = ["Invalid","Input","Convolve","Maxpool","Parallel","Replicated",
             "ParBidiLSTM","DepParUDLSTM","Par2dLSTM","Series","Reconfig",
             "RTLReversed","TTBReversed","XYTranspose","LSTM","SummLSTM",
             "Logistic","LinLogistic","LinTanh","Tanh","Relu","Linear",
             "Softmax","SoftmaxNoCTC","LSTMSoftmax","LSTMBinarySoftmax"]
FC_TYPES = {"Softmax","SoftmaxNoCTC","Logistic","LinLogistic",
            "LinTanh","Tanh","Relu","Linear"}
NF_LR = 64


def read_wm(r, is_training=False):
    mode = r.read_u8()
    int_mode = bool(mode & 1)    # kInt8Flag
    use_adam = bool(mode & 4)    # kAdamFlag
    dbl_flag = bool(mode & 128)  # kDoubleFlag

    if not dbl_flag:
        # Old format (pre-double): float32 arrays
        if int_mode:
            d1 = r.read_i32(); d2 = r.read_i32(); r.read_i8()
            raw = np.frombuffer(r.read_bytes(d1*d2), dtype=np.int8).reshape(d1, d2)
            # Old format scales: vector<float>
            ns = r.read_u32()
            scales = np.array([struct.unpack_from('<f', r.read_bytes(4))[0] for _ in range(ns)])
            w = raw.astype(np.float32) * scales[:, np.newaxis]
        else:
            d1 = r.read_i32(); d2 = r.read_i32(); r.read_bytes(4)  # empty float
            w = np.frombuffer(r.read_bytes(d1*d2*4), dtype=np.float32).copy().reshape(d1,d2)
    else:
        # New format (double): uses double for arrays and scales
        if int_mode:
            d1 = r.read_i32(); d2 = r.read_i32(); r.read_i8()
            raw = np.frombuffer(r.read_bytes(d1*d2), dtype=np.int8).reshape(d1, d2)
            ns = r.read_u32()
            scales = np.array([r.read_f64() for _ in range(ns)])
            w = raw.astype(np.float32) * scales[:, np.newaxis]
        else:
            d1 = r.read_i32(); d2 = r.read_i32(); r.read_f64()  # empty double
            w = np.frombuffer(r.read_bytes(d1*d2*8), dtype=np.float64).reshape(d1,d2).astype(np.float32)

        # Skip training data if present
        if is_training and not int_mode:
            # updates_: GENERIC_2D_ARRAY<double>
            ud1 = r.read_i32(); ud2 = r.read_i32(); r.read_f64()
            r.read_bytes(ud1 * ud2 * 8)
            if use_adam:
                # dw_sq_sum_: GENERIC_2D_ARRAY<double>
                ad1 = r.read_i32(); ad2 = r.read_i32(); r.read_f64()
                r.read_bytes(ad1 * ad2 * 8)
    return {"weight": w[:, :-1], "bias": w[:, -1]}


def read_hdr(r):
    tb = r.read_i8()
    tn = r.read_string() if tb == 0 else (NET_TYPES[tb] if 0 < tb < len(NET_TYPES) else f"?{tb}")
    tr = r.read_i8(); r.read_i8()  # training, needs_bp
    fl = r.read_i32(); ni = r.read_i32(); no = r.read_i32()
    nw = r.read_i32(); nm = r.read_string()
    return tn, ni, no, nw, nm, tr, fl


def parse_net(r, depth=0):
    tn, ni, no, nw, nm, tr, fl = read_hdr(r)
    node = {"type": tn, "ni": ni, "no": no, "name": nm, "flags": fl,
            "children": [], "weights": {}}
    if tn == "Input":
        node["height"] = (r.read_i32(), r.read_i32(), r.read_i32(), r.read_i32(), r.read_i32())[1]
    elif tn in ("Series", "Parallel", "RTLReversed", "TTBReversed", "XYTranspose"):
        cnt = r.read_u32()
        for _ in range(cnt):
            node["children"].append(parse_net(r, depth+1))
        if fl & NF_LR:
            lr_n = r.read_u32(); r.read_bytes(lr_n * 4)
    elif tn == "Convolve":
        node["half_x"] = r.read_i32(); node["half_y"] = r.read_i32()
    elif tn == "Maxpool":
        node["x_scale"] = r.read_i32(); node["y_scale"] = r.read_i32()
    elif tn in ("LSTM", "SummLSTM"):
        node["na"] = r.read_i32(); node["ns"] = no
        for g in ["CI","GI","GF1","GO"]:
            node["weights"][g] = read_wm(r)
    elif tn in FC_TYPES:
        node["weights"]["fc"] = read_wm(r)
    return node


# ---------------------------------------------------------------------------
# Forward pass (pure numpy)
# ---------------------------------------------------------------------------

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def lstm_forward(x, W_ih, W_hh, bias, ns, reverse=False):
    """LSTM forward pass over time axis.

    x:    (T, ni) input
    W_ih: (4*ns, ni) input-to-hidden weights
    W_hh: (4*ns, ns) hidden-to-hidden weights
    bias: (4*ns,) bias
    ns:   hidden size

    Returns: (T, ns) hidden states

    Gate order (PyTorch): i, f, g, o
    """
    T, ni = x.shape
    h = np.zeros(ns, dtype=np.float32)
    c = np.zeros(ns, dtype=np.float32)
    output = np.zeros((T, ns), dtype=np.float32)

    for step in range(T):
        t = (T - 1 - step) if reverse else step
        xt = x[t]

        # gates = W_ih @ x + W_hh @ h + bias
        gates = W_ih @ xt + W_hh @ h + bias

        i_gate = sigmoid(gates[0*ns:1*ns])
        f_gate = sigmoid(gates[1*ns:2*ns])
        g_gate = np.tanh(gates[2*ns:3*ns])
        o_gate = sigmoid(gates[3*ns:4*ns])

        c = f_gate * c + i_gate * g_gate
        h = o_gate * np.tanh(c)

        output[t] = h

    return output


def convolve_stack(x, half_x, half_y):
    """Tesseract 'Convolve' layer: stack 3x3 neighborhood (no learned weights).

    x:  (H, W, C) input feature map
    Returns: (H, W, C * kw * kh) stacked neighborhoods
    """
    H, W, C = x.shape
    kw = 2 * half_x + 1
    kh = 2 * half_y + 1
    out = np.zeros((H, W, C * kw * kh), dtype=np.float32)

    for dx in range(-half_x, half_x + 1):
        for dy in range(-half_y, half_y + 1):
            out_offset = ((dx + half_x) * kh + (dy + half_y)) * C
            for y in range(H):
                for xi in range(W):
                    sy = y + dy
                    sx = xi + dx
                    if 0 <= sy < H and 0 <= sx < W:
                        out[y, xi, out_offset:out_offset+C] = x[sy, sx, :]
                    else:
                        # Random fill in Tesseract — use zeros for determinism
                        pass
    return out


def maxpool(x, scale_x, scale_y):
    """Max-pool with integer strides."""
    H, W, C = x.shape
    oh = H // scale_y
    ow = W // scale_x
    out = np.full((oh, ow, C), -1e30, dtype=np.float32)
    for y in range(oh):
        for xi in range(ow):
            for dy in range(scale_y):
                for dx in range(scale_x):
                    sy = y * scale_y + dy
                    sx = xi * scale_x + dx
                    if sy < H and sx < W:
                        out[y, xi] = np.maximum(out[y, xi], x[sy, sx])
    return out


def fc_forward(x, weight, bias, activation="tanh"):
    """Fully-connected: y = activation(W @ x + b)."""
    y = x @ weight.T + bias
    if activation == "tanh":
        y = np.tanh(y)
    elif activation == "sigmoid":
        y = sigmoid(y)
    elif activation == "softmax":
        y = np.exp(y - y.max(axis=-1, keepdims=True))
        y = y / y.sum(axis=-1, keepdims=True)
    return y


def run_forward(root, image_gray, captures):
    """Run the full Tesseract LSTM forward pass, capturing intermediates.

    image_gray: (H, W) float32 grayscale [0, 1]
    root: parsed network tree

    Returns: (T, n_classes) softmax logits
    """
    def find_layers(node, layers=None, lstm_idx=None):
        """Flatten tree into ordered layer list."""
        if layers is None: layers = []
        if lstm_idx is None: lstm_idx = [0]
        t = node["type"]
        if t in ("Series","Parallel"):
            for c in node["children"]: find_layers(c, layers, lstm_idx)
        elif t in ("RTLReversed","TTBReversed","XYTranspose"):
            for c in node["children"]:
                if t in ("RTLReversed","TTBReversed"):
                    c["_reversed"] = True
                find_layers(c, layers, lstm_idx)
        elif t in ("LSTM","SummLSTM"):
            node["_idx"] = lstm_idx[0]; lstm_idx[0] += 1
            node["_reversed"] = node.get("_reversed", False)
            layers.append(node)
        elif t in FC_TYPES:
            node["_role"] = "conv" if node["name"] == "ConvNL" else "output"
            layers.append(node)
        elif t == "Convolve":
            layers.append(node)
        elif t == "Maxpool":
            layers.append(node)
        return layers

    layers = find_layers(root)

    # Input: (H, W) → (H, W, 1)
    H, W = image_gray.shape
    x = image_gray[:, :, np.newaxis].astype(np.float32)
    captures["input_image"] = x.flatten().copy()

    for layer in layers:
        t = layer["type"]

        if t == "Convolve":
            x = convolve_stack(x, layer["half_x"], layer["half_y"])
            captures["after_convolve"] = x.flatten().copy()

        elif t in FC_TYPES and layer.get("_role") == "conv":
            # Flatten spatial: (H, W, C) → (H*W, C)
            H, W, C = x.shape
            x_flat = x.reshape(H * W, C)
            wm = layer["weights"]["fc"]
            act = "tanh" if layer["type"] == "Tanh" else "linear"
            x_flat = fc_forward(x_flat, wm["weight"], wm["bias"], act)
            x = x_flat.reshape(H, W, -1)
            captures["after_conv_fc"] = x.flatten().copy()

        elif t == "Maxpool":
            x = maxpool(x, layer["x_scale"], layer["y_scale"])
            captures["after_maxpool"] = x.flatten().copy()

        elif t == "SummLSTM":
            # y-summarizing: run LSTM over height axis per column,
            # keep only the final hidden state → collapses height to 1
            H, W, C = x.shape
            ns = layer["ns"]
            wm = layer["weights"]

            # Reorder gates: CI→g, GI→i, GF1→f, GO→o → PyTorch order i,f,g,o
            W_ih, W_hh, bias = _pack_lstm_weights(wm, ns)

            out_cols = np.zeros((W, ns), dtype=np.float32)
            for col in range(W):
                col_input = x[:, col, :]  # (H, C) — run LSTM over H steps
                h_seq = lstm_forward(col_input, W_ih, W_hh, bias, ns, reverse=False)
                out_cols[col] = h_seq[-1]  # keep only last hidden state

            # Output: (1, W, ns) → squeeze height
            x = out_cols[np.newaxis, :, :]  # (1, W, ns) for shape consistency
            x = out_cols  # Actually: (W, ns) — time series
            captures[f"after_lstm_{layer['_idx']}"] = x.flatten().copy()

        elif t == "LSTM":
            # x should be (T, D) at this point
            if x.ndim == 3:
                # (1, W, C) → (W, C) — squeeze height=1 from SummLSTM
                assert x.shape[0] == 1
                x = x[0]

            ns = layer["ns"]
            wm = layer["weights"]
            W_ih, W_hh, bias = _pack_lstm_weights(wm, ns)
            rev = layer.get("_reversed", False)
            x = lstm_forward(x, W_ih, W_hh, bias, ns, reverse=rev)
            captures[f"after_lstm_{layer['_idx']}"] = x.flatten().copy()

        elif t in FC_TYPES and layer.get("_role") == "output":
            # x is (T, D)
            if x.ndim == 3:
                assert x.shape[0] == 1
                x = x[0]
            wm = layer["weights"]["fc"]
            act = "softmax" if "Softmax" in layer["type"] else "linear"
            x = fc_forward(x, wm["weight"], wm["bias"], act)
            captures["logits"] = x.flatten().copy()

    return x


def _pack_lstm_weights(wm, ns):
    """Pack Tesseract gate weights into PyTorch-order stacked format.

    Tesseract order: CI (cell input=g), GI (input=i), GF1 (forget=f), GO (output=o)
    PyTorch order:   i, f, g, o
    """
    gate_order = [("GI", 0), ("GF1", 1), ("CI", 2), ("GO", 3)]  # PyTorch slot
    na = wm["CI"]["weight"].shape[1]
    ni = na - ns

    W_ih = np.zeros((4*ns, ni), dtype=np.float32)
    W_hh = np.zeros((4*ns, ns), dtype=np.float32)
    bias = np.zeros(4*ns, dtype=np.float32)

    for tess_name, slot in gate_order:
        w = wm[tess_name]["weight"]  # (ns, na)
        b = wm[tess_name]["bias"]    # (ns,)
        W_ih[slot*ns:(slot+1)*ns] = w[:, :ni]
        W_hh[slot*ns:(slot+1)*ns] = w[:, ni:]
        bias[slot*ns:(slot+1)*ns] = b

    return W_ih, W_hh, bias


def ctc_greedy_decode(logits, null_char):
    """CTC greedy decode: collapse repeats, remove blanks.

    logits: (T, n_classes) — softmax probabilities
    null_char: blank class index
    Returns: list of (class_id, confidence) tuples
    """
    T = logits.shape[0]
    result = []
    prev = -1
    for t in range(T):
        best = int(np.argmax(logits[t]))
        conf = float(logits[t, best])
        if best != null_char and best != prev:
            result.append((best, conf))
        prev = best
    return result


def _tesseract_normalize(img_u8):
    """Replicate Tesseract's ComputeBlackWhite + SetPixel normalization.

    Scans the middle row for local minima/maxima, takes 25th percentile
    of minima as 'black' and 75th percentile of maxima as 'white'.
    Then normalizes: float_pixel = (pixel - black) / contrast - 1.0
    where contrast = (white - black) / 2.0
    Maps pixel range to approximately [-1, 1].
    """
    H, W = img_u8.shape
    y_mid = H // 2
    row = img_u8[y_mid].astype(np.float32)

    mins = []
    maxes = []
    if W >= 3:
        for x in range(1, W - 1):
            prev, curr, nxt = row[x-1], row[x], row[x+1]
            if (curr < prev and curr <= nxt) or (curr <= prev and curr < nxt):
                mins.append(curr)
            if (curr > prev and curr >= nxt) or (curr >= prev and curr > nxt):
                maxes.append(curr)

    if not mins:
        mins = [0.0]
    if not maxes:
        maxes = [255.0]

    # 25th percentile of mins, 75th percentile of maxes
    black = float(np.percentile(mins, 25))
    white = float(np.percentile(maxes, 75))
    contrast = (white - black) / 2.0
    if contrast <= 0:
        contrast = 1.0

    result = (img_u8.astype(np.float32) - black) / contrast - 1.0
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Dump Tesseract LSTM reference activations")
    p.add_argument("--model", required=True, help=".traineddata path")
    p.add_argument("--image", required=True, help="Grayscale text line image")
    p.add_argument("--output", required=True, help="Output reference GGUF")
    p.add_argument("--height", type=int, default=0,
                   help="Override input height (0 = auto from model)")
    args = p.parse_args()

    # ── Load image ────────────────────────────────────────────────────
    try:
        from PIL import Image
        img = Image.open(args.image).convert("L")  # grayscale
        img_u8 = np.array(img, dtype=np.uint8)     # [0, 255]
    except ImportError:
        print("ERROR: Pillow required (pip install Pillow)")
        sys.exit(1)

    print(f"Image: {args.image} ({img_u8.shape[1]}x{img_u8.shape[0]})")

    # ── Parse traineddata ─────────────────────────────────────────────
    data = Path(args.model).read_bytes()
    comps = parse_traineddata(data)
    tokens = parse_unicharset(comps["lstm-unicharset"]) if "lstm-unicharset" in comps else []
    recoder = parse_recoder(comps["lstm-recoder"]) if "lstm-recoder" in comps else []

    r = TessReader(comps["lstm"])
    root = parse_net(r)

    # Read LSTMRecognizer metadata
    vgsl = r.read_string()
    _ = r.read_i32()  # training_flags
    _ = r.read_i32()  # training_iteration
    _ = r.read_i32()  # sample_iteration
    null_char = r.read_i32()

    # Find input height
    def find_input(n):
        if n["type"] == "Input": return n
        for c in n.get("children", []):
            r = find_input(c)
            if r: return r
        return None

    inp = find_input(root)
    input_height = args.height if args.height > 0 else (inp["height"] if inp else 36)
    print(f"VGSL: {vgsl}")
    print(f"Input height: {input_height}, null_char: {null_char}")
    print(f"Unicharset: {len(tokens)} tokens, Recoder: {len(recoder)} codes")

    # ── Resize image to model input height ────────────────────────────
    h_orig, w_orig = img_u8.shape
    scale = input_height / h_orig
    new_w = max(1, int(w_orig * scale + 0.5))
    from PIL import Image as PILImage
    img_resized = PILImage.fromarray(img_u8)
    img_resized = img_resized.resize((new_w, input_height), PILImage.BILINEAR)
    img_u8_resized = np.array(img_resized, dtype=np.uint8)
    print(f"Resized: {new_w}x{input_height}")

    # ── Tesseract-style pixel normalization ───────────────────────────
    # Tesseract's ComputeBlackWhite: scan middle row for local min/max,
    # take 25th percentile of minima as black, 75th percentile of maxima
    # as white. Then: float_pixel = (pixel - black) / contrast - 1.0
    # where contrast = (white - black) / 2.0
    img_input = _tesseract_normalize(img_u8_resized)
    print(f"Normalized: min={img_input.min():.3f} max={img_input.max():.3f}")

    # ── Run forward pass ──────────────────────────────────────────────
    captures = {}
    logits = run_forward(root, img_input, captures)

    # ── CTC decode ────────────────────────────────────────────────────
    decoded = ctc_greedy_decode(logits, null_char)
    # Build reverse recoder: output_class → unichar_id
    # recoder[unichar_id] = [output_class, ...] (encoder direction)
    # We need: output_class → unichar_id
    rev_recoder = {}
    for uid, codes in enumerate(recoder):
        if len(codes) == 1 and codes[0] not in rev_recoder:
            rev_recoder[codes[0]] = uid

    text_chars = []
    for class_id, conf in decoded:
        uid = rev_recoder.get(class_id, -1)
        if 0 <= uid < len(tokens):
            text_chars.append(tokens[uid])
        else:
            text_chars.append(f"<{class_id}>")

    result_text = "".join(text_chars)
    print(f"\nCrispEmbed decode: '{result_text}'")

    # ── Run Tesseract for ground truth ────────────────────────────────
    try:
        tess_result = subprocess.run(
            ["tesseract", args.image, "stdout", "--oem", "1", "--psm", "7",
             "--tessdata-dir", str(Path(args.model).parent)],
            capture_output=True, text=True, timeout=10
        )
        tess_text = tess_result.stdout.strip()
        print(f"Tesseract decode: '{tess_text}'")

        # Character error rate
        if tess_text:
            matches = sum(1 for a, b in zip(result_text, tess_text) if a == b)
            max_len = max(len(result_text), len(tess_text))
            cer = 1.0 - matches / max_len if max_len > 0 else 0.0
            print(f"CER vs Tesseract: {cer:.3f} ({matches}/{max_len} chars match)")
    except Exception as e:
        print(f"Tesseract comparison skipped: {e}")

    # ── Write reference GGUF ──────────────────────────────────────────
    writer = gguf.GGUFWriter(str(args.output), arch="tesseract_lstm_ref")

    writer.add_string("general.name", "tesseract-lstm-reference")
    writer.add_string("tesseract_lstm_ref.model_path", str(args.model))
    writer.add_string("tesseract_lstm_ref.image_path", str(args.image))
    writer.add_string("tesseract_lstm_ref.vgsl_spec", vgsl)
    writer.add_uint32("tesseract_lstm_ref.input_height", input_height)
    writer.add_uint32("tesseract_lstm_ref.null_char", null_char)
    writer.add_string("tesseract_lstm_ref.decoded_text", result_text)

    dtype_gguf = gguf.GGMLQuantizationType.F32
    n_tensors = 0
    for name, arr in captures.items():
        writer.add_tensor(name, arr.astype(np.float32), raw_dtype=dtype_gguf)
        n_tensors += 1
        print(f"  {name}: {arr.shape} ({arr.size} elements)")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    out_size = Path(args.output).stat().st_size
    print(f"\nWrote {args.output}")
    print(f"  Tensors: {n_tensors}")
    print(f"  File size: {out_size:,} bytes ({out_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
