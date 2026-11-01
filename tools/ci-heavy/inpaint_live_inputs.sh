#!/usr/bin/env bash
# Shared live-eval inputs for the inpainting lanes: real photos and the DBNet
# text detector. usage: inpaint_live_inputs.sh PHOTO_DIR DBNET_PATH RECORD_JSON
# Photos: the repository's face.png plus two Wikimedia Commons photos (the
# same set as the Big-LaMa research lane); each digest is recorded, not pinned.
set -uo pipefail
PHOTOS=${1:?}; DBNET=${2:?}; RECORD=${3:?}
mkdir -p "$PHOTOS"
cp tests/regression/images/face.png "$PHOTOS/"
python - "$PHOTOS" > "$RECORD" <<'PY'
import hashlib, json, sys, urllib.request
from pathlib import Path
d = Path(sys.argv[1])
urls = {"landscape.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a8/Tour_Eiffel_Wikimedia_Commons.jpg/960px-Tour_Eiffel_Wikimedia_Commons.jpg",
        "street.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/New_york_times_square-terabass.jpg/1280px-New_york_times_square-terabass.jpg"}
rec = {}
for name, url in urls.items():
    try:
        data = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "CrispEmbed-inpaint-eval/1"}), timeout=60).read()
        (d / name).write_bytes(data)
        rec[name] = {"url": url, "sha256": hashlib.sha256(data).hexdigest()}
    except Exception as e:
        rec[name] = {"url": url, "error": str(e)}
for p in sorted(d.iterdir()):
    rec.setdefault(p.name, {"sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
print(json.dumps(rec, indent=2))
PY
DBNET_URL=https://huggingface.co/cstr/dbnet-ic15-GGUF/resolve/main/dbnet-ic15-q8_0.gguf
DBNET_SHA=163c3b3d1211fd47b4ea323e2e684287bdc75ae68f0ffea397ccfcdc29a74de2
curl -sfL -o "$DBNET" "$DBNET_URL" && echo "$DBNET_SHA  $DBNET" | sha256sum -c - >/dev/null || rm -f "$DBNET"
