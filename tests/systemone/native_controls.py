#!/usr/bin/env python3
"""Model-free actual installed C ABI controls. Run only on hosted build workers."""
import ctypes
import json
import os
from pathlib import Path
import sys
import threading

GATES = ['CRISPEMBED_JOSEV_PACKED', 'CRISPEMBED_JOSEV_COMPOSED', 'CRISPEMBED_JOSEV_BACKBONE',
         'CRISPEMBED_QWEN35_OCR', 'CRISPEMBED_JOSEV_HEAD', 'CRISPEMBED_JOSEV_TEXT_FRONTEND',
         'CRISPEMBED_JOSEV_ANSWER_SERIALIZER']
assert os.environ.get('GITHUB_ACTIONS') == 'true', 'hosted build only'
lib = ctypes.CDLL(str(Path(sys.argv[1]).resolve()))
lib.crispembed_systemone_init.argtypes = [ctypes.c_char_p, ctypes.c_int]
lib.crispembed_systemone_init.restype = ctypes.c_void_p
lib.crispembed_systemone_free.argtypes = [ctypes.c_void_p]
lib.crispembed_systemone_free.restype = None
for name in ('error', 'profile'):
    fn = getattr(lib, 'crispembed_systemone_'+name)
    fn.argtypes = [ctypes.c_void_p]; fn.restype = ctypes.c_char_p
lib.crispembed_systemone_run_json.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
lib.crispembed_systemone_run_json.restype = ctypes.c_char_p

def error(): return lib.crispembed_systemone_error(None).decode()
for gate in GATES: os.environ.pop(gate, None)
assert lib.crispembed_systemone_profile(None) == b''
lib.crispembed_systemone_free(None)
assert not lib.crispembed_systemone_run_json(None, b'{}')
assert error() == 'SystemOne context is null'
rows = []
for value in (None, '', '0'):
    for gate in GATES:
        if value is None: os.environ.pop(gate, None)
        else: os.environ[gate] = value
    assert not lib.crispembed_systemone_init(b'absent.gguf', 2)
    assert error() == 'Josev packed/component gates are off'
    rows.append({'value':value, 'error':error()})
for gate in GATES: os.environ[gate] = '1'
for count in (-1, 65):
    assert not lib.crispembed_systemone_init(b'absent.gguf', count)
    assert error() == 'Josev packed path/1..64 threads required'
assert not lib.crispembed_systemone_init(None, 2)
assert error() == 'Josev packed path/1..64 threads required'
parent_error = error()
child = []
def worker():
    assert not lib.crispembed_systemone_run_json(None, None)
    child.append(error())
t = threading.Thread(target=worker); t.start(); t.join()
assert child == ['SystemOne context is null'] and error() == parent_error
for gate in GATES: os.environ.pop(gate, None)
assert not lib.crispembed_systemone_init(b'absent.gguf', 0)
assert error() == 'Josev packed/component gates are off'
Path(sys.argv[2]).write_text(json.dumps({'completed':True, 'model_loaded':False,
    'gate_controls':rows, 'null_handle':True, 'thread_local_errors':True,
    'thread_bounds':True, 'zero_thread_default':True}, indent=2)+'\n')
