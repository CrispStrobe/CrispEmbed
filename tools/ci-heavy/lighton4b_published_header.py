"""Bounded immutable HTTP206 metadata acquisition; never a full-file hash proof."""
import hashlib
import json
import struct
import subprocess
import urllib.request

REPO = 'cstr/LightOnOCR-3-4B-GGUF-experimental-6b9dfe91eaa2'
REVISION = 'a7ec2e54103165a96b8bb6dc3478a2887edd7459'
FILENAME = 'lightonocr3-4b-vision-f32.gguf'
SIZE = 9757493600
DECLARED_SHA256 = '757ad5d0ea7699b3d0ce4cbb70978de442e5b95e660da0a13cf048980e255606'

class NeedMore(Exception):
    pass


def metadata_end(data):
    """Skip bounded GGUF v3 metadata and tensor descriptions without allocations."""
    offset = 0
    def take(n):
        nonlocal offset
        if n < 0 or n > 64 * 1024 * 1024: raise ValueError('unbounded metadata field')
        if offset + n > len(data): raise NeedMore()
        start = offset; offset += n
        return memoryview(data)[start:offset]
    def u32(): return struct.unpack('<I', take(4))[0]
    def u64(): return struct.unpack('<Q', take(8))[0]
    def string(): return bytes(take(u64()))
    widths = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1, 10:8, 11:8, 12:8}
    def skip(kind):
        if kind in widths: take(widths[kind])
        elif kind == 8: take(u64())
        elif kind == 9:
            element, count = u32(), u64()
            if count > 2000000 or element == 9: raise ValueError('unsupported array bound/type')
            if element in widths: take(widths[element] * count)
            elif element == 8:
                for _ in range(count): take(u64())
            else: raise ValueError('unknown array type')
        else: raise ValueError('unknown metadata type')
    if bytes(take(4)) != b'GGUF' or u32() != 3: raise ValueError('expected GGUFv3')
    tensors, fields = u64(), u64()
    if tensors != 723 or fields > 10000: raise ValueError('unexpected GGUF census')
    alignment = 32
    for _ in range(fields):
        key, kind = string(), u32()
        if key == b'general.alignment':
            if kind != 4: raise ValueError('alignment type')
            alignment = u32()
            if alignment < 1 or alignment > 4096 or alignment & (alignment-1): raise ValueError('alignment value')
        else: skip(kind)
    for _ in range(tensors):
        string(); dims = u32()
        if not 1 <= dims <= 4: raise ValueError('tensor rank')
        take(dims * 8); u32(); u64()
    end = (offset + alignment - 1) // alignment * alignment
    if end > len(data): raise NeedMore()
    return end


def inspect_published(binary, out):
    url = f'https://huggingface.co/{REPO}/resolve/{REVISION}/{FILENAME}'
    data = bytearray(); ranges = []; end = None
    report = {'schema':'lighton4b-published-header/v1','status':'failed','repository':REPO,
        'revision':REVISION,'filename':FILENAME,'declared_full_size':SIZE,'declared_full_sha256':DECLARED_SHA256,
        'full_payload_hash_verified':False,'payload_backed_model_admitted':False,'native_inference_observed':False,
        'ranges':ranges}
    try:
        while len(data) < 64*1024*1024:
            start = len(data); stop = start + 4*1024*1024 - 1
            # Range-specific query prevents a CDN cached response for another range.
            req = urllib.request.Request(url + f'?download=true&header_range={start}',
                headers={'Range':f'bytes={start}-{stop}','Accept-Encoding':'identity'})
            with urllib.request.urlopen(req, timeout=120) as response:
                content_range = response.headers.get('Content-Range')
                expected = f'bytes {start}-{stop}/{SIZE}'
                if response.status != 206 or content_range != expected:
                    raise ValueError('server did not honor exact bounded HTTP206 range')
                chunk = response.read(stop-start+2)
                if len(chunk) != stop-start+1: raise ValueError('range byte count differs')
            ranges.append({'start':start,'end':stop,'status':206,'content_range':content_range,
                           'sha256':hashlib.sha256(chunk).hexdigest()})
            data.extend(chunk)
            try: end = metadata_end(data); break
            except NeedMore: pass
        if end is None: raise ValueError('metadata exceeds64MiB protocol bound')
        header = out/'published-header-only.gguf'; header.write_bytes(data[:end])
        report.update({'fetched_prefix_bytes':len(data),'retained_header_bytes':end,
                       'header_sha256':hashlib.sha256(data[:end]).hexdigest()})
        result = subprocess.run([str(binary),'--inspect',str(header)],text=True,capture_output=True,check=True)
        (out/'published-header-controls.log').write_text(result.stdout+result.stderr)
        if 'PUBLISHED_HEADER passed=1' not in result.stdout: raise ValueError('native header receipt missing')
        report['native_header_stdout']=result.stdout; report['status']='passed'
    except Exception as error:
        report['error_type']=type(error).__name__;report['error']=str(error)
        raise
    finally:
        (out/'published-header-receipt.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
