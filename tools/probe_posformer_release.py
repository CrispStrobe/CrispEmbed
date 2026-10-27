#!/usr/bin/env python3
"""Bounded public-ABI probe for actual released/development PosFormer libraries.

Run separate processes for each library and set its LD_LIBRARY_PATH externally;
never mix different ggml libraries in one process. This does not rebuild or
instrument a production library. Existing native step diagnostics supply IDs.
"""
import argparse
import ctypes as C
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import sys
import time

MODEL_SHA = '450211ad27ce19e2f30651e69fdc77ea76a13b66e29e045858286024864bf4a4'
MANIFEST_SHA = 'a2edabf8937298a52f072c41917bf4a8022ecaee9a50d00dcc1d52934b711109'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def peak_rss_record(raw, platform_name):
    """getrusage reports bytes on macOS, KiB on Linux; retain original units."""
    return dict(peak_rss_raw=raw,
                peak_rss_raw_unit='bytes' if platform_name == 'darwin' else 'KiB',
                peak_rss_mib=raw/(1024**2 if platform_name == 'darwin' else 1024))


def compare_release_records(released, candidate):
    """Exact released IDs/text/stops; confidence drift stays explicitly measured."""
    for record in (released, candidate):
        if (record['model_sha256'] != MODEL_SHA or record['manifest_sha256'] != MANIFEST_SHA
                or record['threads'] != 2 or not record.get('completed') or len(record['cases']) != 50):
            raise ValueError('Incomplete or changed release comparison')
    if released['library_source'] != '78493a31fc3043f3af4ab6fd2f197a32da6f07f0':
        raise ValueError('Wrong released source')
    if len({c['id'] for c in released['cases']}) != 50:
        raise ValueError('Duplicate released cases')
    confidence_max = mean_max = 0.0
    for old, new in zip(released['cases'], candidate['cases']):
        for key in ('id','image_sha256','text','token_ids','stop'):
            if old[key] != new[key]:raise ValueError('Released output changed: '+key)
        if len(old['confidence']) != len(new['confidence']):raise ValueError('Confidence shape changed')
        for record in (old, new):
            if (not all(math.isfinite(x) and 0 < x <= 1 for x in record['confidence'])
                    or not math.isfinite(record['mean_confidence'])
                    or not 0 <= record['mean_confidence'] <= 1
                    or (record['confidence'] and record['mean_confidence'] == 0)):
                raise ValueError('Nonfinite/invalid confidence')
        confidence_max = max([confidence_max]+[abs(x-y) for x,y in zip(old['confidence'],new['confidence'])])
        mean_max = max(mean_max,abs(old['mean_confidence']-new['mean_confidence']))
    return dict(cases=50,token_ids_exact=True,text_exact=True,stops_exact=True,
                max_confidence_abs=confidence_max,max_mean_confidence_abs=mean_max,
                confidence_scope='Measured; no bitwise claim or fitted numerical tolerance')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library', type=Path, required=True)
    p.add_argument('--library-source', required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--frozen-report', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--logs', type=Path, required=True)
    p.add_argument('--limit', type=int, choices=(1,50), default=50)
    args = p.parse_args()
    from PIL import Image
    if not re.fullmatch('[0-9a-f]{40}',args.library_source):raise ValueError('Unpinned library source')
    if sha(args.model)!=MODEL_SHA or sha(args.corpus/'manifest.json')!=MANIFEST_SHA:
        raise ValueError('Original model/manifest bytes changed')
    frozen=json.loads(args.frozen_report.read_text())['reports']['original-baseline.json']['report']
    cases=frozen['cases']
    if len(cases)!=50:raise ValueError('Incomplete original case panel')
    for case in cases:
        if sha(args.corpus/(case['id']+'.png'))!=case['image_sha256']:
            raise ValueError('Original image bytes changed')
    if os.environ.get('POSFORMER_SCALAR_ENCODER') is not None:
        raise ValueError('Requires ordinary encoder, with scalar override unset')
    lib=C.CDLL(str(args.library.resolve()))
    lib.crispembed_math_ocr_init.argtypes=[C.c_char_p,C.c_int];lib.crispembed_math_ocr_init.restype=C.c_void_p
    lib.crispembed_math_ocr_free.argtypes=[C.c_void_p];lib.crispembed_math_ocr_free.restype=None
    lib.crispembed_math_ocr_recognize.argtypes=[C.c_void_p,C.POINTER(C.c_uint8),C.c_int,C.c_int,C.c_int,C.POINTER(C.c_int)]
    lib.crispembed_math_ocr_recognize.restype=C.c_void_p
    lib.crispembed_math_ocr_confidences.argtypes=[C.c_void_p,C.POINTER(C.c_int)]
    lib.crispembed_math_ocr_confidences.restype=C.POINTER(C.c_float)
    lib.crispembed_math_ocr_mean_confidence.argtypes=[C.c_void_p];lib.crispembed_math_ocr_mean_confidence.restype=C.c_float
    report=dict(library=args.library.name,library_source=args.library_source,library_sha256=sha(args.library),
                tool_sha256=sha(__file__),model_sha256=MODEL_SHA,manifest_sha256=MANIFEST_SHA,threads=2,
                trained_forward_env=os.environ.get('CRISPEMBED_POSFORMER_TRAINED_FORWARD'),
                scope='Actual binary public API/step diagnostics, frozen50 cases; no independent quality claim',cases=[])
    args.logs.mkdir(parents=True,exist_ok=True);args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    save();ctx=lib.crispembed_math_ocr_init(os.fsencode(args.model),2)
    if not ctx:raise ValueError('Actual library/model initialization failed')
    try:
        for case in cases[:args.limit]:
            image=Image.open(args.corpus/(case['id']+'.png')).convert('RGB')
            pixels=image.tobytes();buffer=(C.c_uint8*len(pixels)).from_buffer_copy(pixels)
            log=args.logs/(case['id']+'.log');old=C.c_int();length=C.c_int()
            fd=os.open(log,os.O_CREAT|os.O_TRUNC|os.O_WRONLY,0o600);old_fd=os.dup(2)
            started=time.monotonic()
            try:
                os.dup2(fd,2)
                result=lib.crispembed_math_ocr_recognize(ctx,buffer,image.width,image.height,3,C.byref(length))
            finally:
                os.dup2(old_fd,2);os.close(old_fd);os.close(fd)
            if not result or not 0<=length.value<=1000000:raise ValueError('Invalid native output')
            text=C.string_at(result,length.value).decode('utf-8',errors='strict')
            steps=[(int(a),int(b)) for a,b in re.findall(r'\bstep (\d+): token=(\d+)',log.read_text())]
            if not steps or [x[0] for x in steps]!=list(range(len(steps))):
                raise ValueError('Incomplete actual native token diagnostics')
            n=C.c_int();ptr=lib.crispembed_math_ocr_confidences(ctx,C.byref(n))
            if not 0<=n.value<=200 or (n.value and not ptr):raise ValueError('Invalid confidence buffer')
            confidence=[float(ptr[i]) for i in range(n.value)]
            mean=float(lib.crispembed_math_ocr_mean_confidence(ctx))
            if not all(math.isfinite(x) and 0<=x<=1 for x in confidence+[mean]):
                raise ValueError('Invalid native confidence')
            report['cases'].append(dict(id=case['id'],image_sha256=case['image_sha256'],text=text,
                token_ids=[x[1] for x in steps],stop='eos' if steps[-1][1]==2 else 'length',
                confidence=confidence,mean_confidence=mean,elapsed_seconds=time.monotonic()-started,
                log_sha256=sha(log)))
            save()
    finally:lib.crispembed_math_ocr_free(ctx)
    report.update(completed=True,**peak_rss_record(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,sys.platform))
    save();print('Completed',len(report['cases']),'bounded actual-library cases')


if __name__=='__main__':main()
