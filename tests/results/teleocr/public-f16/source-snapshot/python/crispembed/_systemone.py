"""Native SystemOne joint-schema text decisions."""
import ctypes
import json
from ._binding import _load_library, CrispEmbed, set_offline


class CrispSystemOne:
    """CPU/F32 Josev packed model, full publisher request and response schema.

    State-only clipping is bounded to512 model tokens. Existing Gemma ``decide``
    and embedding interfaces retain their own semantics. Model gates are opt-in.
    """
    def __init__(self, model_path: str, n_threads: int = 2, lib_path=None,
                 auto_download=None, offline=None):
        self._ctx = None
        if not isinstance(model_path, str) or "\0" in model_path:
            raise ValueError("SystemOne model path must be a NUL-free string")
        self._lib = _load_library(lib_path)
        if not all(hasattr(self._lib, name) for name in (
                "crispembed_systemone_init", "crispembed_systemone_free",
                "crispembed_systemone_run_json", "crispembed_systemone_error",
                "crispembed_systemone_profile")):
            raise RuntimeError("Native library does not provide SystemOne support")
        if not isinstance(n_threads, int) or isinstance(n_threads, bool) or not 0 <= n_threads <= 64:
            raise ValueError("SystemOne n_threads must be0..64")
        lib = self._lib
        lib.crispembed_systemone_init.argtypes = [ctypes.c_char_p, ctypes.c_int]
        lib.crispembed_systemone_init.restype = ctypes.c_void_p
        lib.crispembed_systemone_free.argtypes = [ctypes.c_void_p]
        lib.crispembed_systemone_free.restype = None
        lib.crispembed_systemone_run_json.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.crispembed_systemone_run_json.restype = ctypes.c_char_p
        lib.crispembed_systemone_error.argtypes = [ctypes.c_void_p]
        lib.crispembed_systemone_error.restype = ctypes.c_char_p
        lib.crispembed_systemone_profile.argtypes = [ctypes.c_void_p]
        lib.crispembed_systemone_profile.restype = ctypes.c_char_p
        if offline is not None:
            set_offline(offline, lib_path=lib_path)
        resolved = CrispEmbed.resolve_model(model_path, auto_download=auto_download, lib_path=lib_path)
        self._ctx = lib.crispembed_systemone_init(resolved.encode("utf-8"), n_threads)
        if not self._ctx:
            raise RuntimeError(self._error())

    def _error(self):
        value = self._lib.crispembed_systemone_error(self._ctx)
        return value.decode("utf-8", errors="replace") if value else "SystemOne operation failed"

    @property
    def profile(self):
        self._require_open()
        return self._lib.crispembed_systemone_profile(self._ctx).decode("utf-8")

    def _require_open(self):
        if not self._ctx:
            raise RuntimeError("SystemOne context is closed")

    def run_json(self, request: str) -> str:
        """Run exact raw JSON; preserves question/criterion insertion order."""
        self._require_open()
        if not isinstance(request, str) or "\0" in request:
            raise ValueError("SystemOne request must be a NUL-free JSON string")
        encoded = request.encode("utf-8")
        if len(encoded) > 65536:
            raise ValueError("SystemOne request exceeds64KiB")
        result = self._lib.crispembed_systemone_run_json(self._ctx, encoded)
        if not result:
            raise ValueError(self._error())
        return result.decode("utf-8")

    def run(self, request: dict) -> dict:
        """Return the full model/answers/usage response for a publisher request."""
        if not isinstance(request, dict):
            raise TypeError("SystemOne request must be a dictionary")
        text = json.dumps(request, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        return json.loads(self.run_json(text))

    def close(self):
        if self._ctx:
            self._lib.crispembed_systemone_free(self._ctx)
            self._ctx = None

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
