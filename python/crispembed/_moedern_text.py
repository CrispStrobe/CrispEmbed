"""Dedicated raw-text/role MoEdern1.5M multivectors; no dense-pooling API."""
import ctypes as C
import math
import os
from ._binding import _load_library, CrispEmbed, set_offline

class _Result(C.Structure):
    _fields_=[('abi_version',C.c_uint32),('struct_bytes',C.c_uint32),
              ('role',C.c_int32),('n_tokens',C.c_int32),('n_selected',C.c_int32),
              ('dimension',C.c_int32),('expansion_count',C.c_int32),
              ('ids',C.POINTER(C.c_int32)),('supplied_mask',C.POINTER(C.c_uint8)),
              ('effective_mask',C.POINTER(C.c_uint8)),('scoring_mask',C.POINTER(C.c_uint8)),
              ('expansion_positions',C.POINTER(C.c_uint8)),('all_vectors',C.POINTER(C.c_float)),
              ('selected_vectors',C.POINTER(C.c_float))]

class CrispMoEdern:
    """CPU/F32 packed1.5M context. Five model gates stay opt-in.

    encode returns copied IDs, three masks, expansion positions and both flat
    row-major128-vector arrays. Explicit query/document role is mandatory.
    """
    def __init__(self,model_path,n_threads=2,lib_path=None,auto_download=None,offline=None):
        self._ctx=None
        if not isinstance(model_path,str)or '\0'in model_path:raise ValueError('NUL-free model path required')
        cap=64 if os.environ.get('CRISPEMBED_MOEDERN_FAST','') not in ('','0') else 2  # fast family path only; native re-checks
        if type(n_threads)is not int or not 0<=n_threads<=cap:raise ValueError('MoEdern threads must be0..%d'%cap)
        lib=self._lib=_load_library(lib_path)
        if not all(hasattr(lib,'crispembed_moedern_'+n)for n in ('init','free','encode','score','error')):raise RuntimeError('Native library does not provide MoEdern text support')
        for name,args,result in [('init',[C.c_char_p,C.c_int],C.c_void_p),('free',[C.c_void_p],None),
            ('encode',[C.c_void_p,C.c_char_p,C.c_int],C.POINTER(_Result)),
            ('score',[C.c_void_p,C.c_char_p,C.c_char_p,C.POINTER(C.c_float)],C.c_int),('error',[C.c_void_p],C.c_char_p)]:
            fn=getattr(lib,'crispembed_moedern_'+name);fn.argtypes=args;fn.restype=result
        if offline is not None:set_offline(offline,lib_path=lib_path)
        path=CrispEmbed.resolve_model(model_path,auto_download=auto_download,lib_path=lib_path)
        self._ctx=lib.crispembed_moedern_init(path.encode('utf-8'),n_threads)
        if not self._ctx:raise RuntimeError(self._error())
    def _error(self):
        value=self._lib.crispembed_moedern_error(self._ctx)
        return value.decode('utf-8',errors='replace')if value else 'MoEdern operation failed'
    def _open(self):
        if not self._ctx:raise RuntimeError('MoEdern context is closed')
    @staticmethod
    def _text(value):
        if not isinstance(value,str)or '\0'in value:raise ValueError('NUL-free UTF8 text required')
        data=value.encode('utf-8')
        if len(data)>65536:raise ValueError('MoEdern text exceeds64KiB')
        return data
    def encode(self,text,role):
        self._open()
        if role not in ('query','document'):raise ValueError('Explicit query/document role required')
        kind=0 if role=='query'else 1
        pointer=self._lib.crispembed_moedern_encode(self._ctx,self._text(text),kind)
        if not pointer:raise ValueError(self._error())
        row=pointer.contents
        if not(row.abi_version==1 and row.struct_bytes==C.sizeof(_Result)and row.role==kind and row.dimension==128 and
               1<=row.n_selected<=row.n_tokens<=512 and row.expansion_count==(row.n_tokens if kind==0 else 0)):
            raise RuntimeError('MoEdern native result ABI/shape differs')
        def copy(name,n):
            value=getattr(row,name)
            if n and not value:raise RuntimeError('MoEdern result pointer is null')
            return list(value[:n])if n else []
        result={'role':role,'dimension':128,'n_tokens':row.n_tokens,'n_selected':row.n_selected,
            'ids':copy('ids',row.n_tokens),'supplied_mask':copy('supplied_mask',row.n_tokens),
            'effective_mask':copy('effective_mask',row.n_tokens),'scoring_mask':copy('scoring_mask',row.n_tokens),
            'expansion_positions':copy('expansion_positions',row.expansion_count),
            'all_vectors':copy('all_vectors',row.n_tokens*128),'selected_vectors':copy('selected_vectors',row.n_selected*128)}
        if not all(math.isfinite(x)for k in ('all_vectors','selected_vectors')for x in result[k]):raise RuntimeError('Nonfinite MoEdern vectors')
        return result
    def score(self,query,document):
        self._open();value=C.c_float()
        if self._lib.crispembed_moedern_score(self._ctx,self._text(query),self._text(document),C.byref(value))!=1:raise ValueError(self._error())
        if not math.isfinite(value.value):raise RuntimeError('Nonfinite MoEdern score')
        return value.value
    def close(self):
        if self._ctx:self._lib.crispembed_moedern_free(self._ctx);self._ctx=None
    def __enter__(self):self._open();return self
    def __exit__(self,*_):self.close()
    def __del__(self):
        try:self.close()
        except Exception:pass
