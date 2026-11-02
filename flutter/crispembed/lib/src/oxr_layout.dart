import 'dart:convert';
import 'dart:ffi';
import 'dart:typed_data';
import 'package:ffi/ffi.dart';
import 'crispembed_bindings.dart';

/// Native OXR-1.0 D-FINE layout detector. Requires CRISPEMBED_OXR=1 and
/// CRISPEMBED_ACCEPT_OXR_LICENSE=1 (OXR License). [detect] returns the OXR
/// layout service's /inference JSON body for one page. Calls are synchronous.
class CrispOXRLayout {
  Pointer<Void> _ctx = nullptr;
  late final OxrFree _free;
  late final OxrLayoutDetect _detect;
  late final OxrString _error;
  late final OxrString _timings;

  CrispOXRLayout(DynamicLibrary lib, String modelPath, {int nThreads = 4}) {
    if (modelPath.isEmpty || modelPath.contains('\u0000')) {
      throw ArgumentError('Model path is empty or contains NUL');
    }
    if (nThreads < 0 || nThreads > 2147483647) throw ArgumentError('Invalid thread count');
    final init = lib.lookupFunction<OxrInitNative, OxrInit>('crispembed_oxr_layout_init');
    _free = lib.lookupFunction<OxrFreeNative, OxrFree>('crispembed_oxr_layout_free');
    _detect = lib.lookupFunction<OxrLayoutDetectNative, OxrLayoutDetect>('crispembed_oxr_layout_detect');
    _error = lib.lookupFunction<OxrStringNative, OxrString>('crispembed_oxr_layout_last_error');
    _timings = lib.lookupFunction<OxrStringNative, OxrString>('crispembed_oxr_layout_timings');
    final path = modelPath.toNativeUtf8();
    try { _ctx = init(path, nThreads); } finally { calloc.free(path); }
    if (_ctx == nullptr) throw StateError('OXR layout model/gate initialization failed');
  }

  /// [pixels] are interleaved 8-bit gray/RGB/RGBA ([channels] 1, 3 or 4).
  /// [confidence] is the service's optional threshold (null = all candidates).
  Map<String, dynamic> detect(Uint8List pixels, int width, int height,
      {int channels = 3, double? confidence}) {
    if (_ctx == nullptr) throw StateError('OXR layout context is disposed');
    if (width <= 0 || height <= 0 || ![1, 3, 4].contains(channels) ||
        pixels.length != width * height * channels) {
      throw ArgumentError('Pixel byte length does not match valid image dimensions');
    }
    if (confidence != null && (!confidence.isFinite || confidence < 0 || confidence > 1)) {
      throw ArgumentError('confidence must be finite and in [0, 1]');
    }
    final buffer = calloc<Uint8>(pixels.length);
    try {
      buffer.asTypedList(pixels.length).setAll(0, pixels);
      final result = _detect(_ctx, buffer, width, height, channels, confidence ?? double.nan);
      if (result == nullptr) {
        final error = _error(_ctx);
        throw StateError(error == nullptr ? 'OXR layout inference failed' : error.toDartString());
      }
      return jsonDecode(result.toDartString()) as Map<String, dynamic>;
    } finally { calloc.free(buffer); }
  }

  /// Timings (ms) of the last [detect] call.
  Map<String, dynamic> timings() {
    if (_ctx == nullptr) throw StateError('OXR layout context is disposed');
    return jsonDecode(_timings(_ctx).toDartString()) as Map<String, dynamic>;
  }

  void dispose() {
    if (_ctx != nullptr) { _free(_ctx); _ctx = nullptr; }
  }
}
