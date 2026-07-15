// Web implementation of CrispEmbed OCR — runs text recognition + pipeline
// client-side via the CrispEmbed OCR WASM module (crispembed_ocr.js + .wasm).
//
// Replaces math_ocr.dart on web via conditional export.
// The WASM module must be loaded via a <script> tag in web/index.html:
//   <script src="wasm/crispembed_ocr.js"></script>
//
// Supports:
//   - Single-model recognition (TrOCR / pix2tex)
//   - Full pipeline (DBNet detection + TrOCR recognition)
//   - Scan cleanup (classical preprocessing, no model needed)

import 'dart:async';
import 'dart:js_interop';
import 'dart:js_interop_unsafe';
import 'dart:typed_data';

// ---------------------------------------------------------------------------
// CrispEmbedOcr — Web WASM implementation (single-model recognition)
// ---------------------------------------------------------------------------

class CrispEmbedOcr {
  late final JSObject _module;
  late final int _ctxPtr;
  bool _disposed = false;

  CrispEmbedOcr._({required JSObject module, required int ctxPtr}) {
    _module = module;
    _ctxPtr = ctxPtr;
  }

  /// Direct constructor not available on web.
  CrispEmbedOcr(String modelPath, {int nThreads = 4, String? libPath}) {
    throw UnsupportedError(
        'CrispEmbedOcr direct constructor not available on web. Use CrispEmbedOcr.load()');
  }

  /// Load the WASM module, fetch the model, and initialize.
  static Future<CrispEmbedOcr> load({
    required String modelUrl,
    String modelPath = '/models/ocr.gguf',
    int nThreads = 1,
    int? maxTokens,
    void Function(double)? onProgress,
  }) async {
    onProgress?.call(0.0);

    // Initialize Emscripten module
    final factory =
        globalContext.getProperty('CrispEmbedOCR'.toJS) as JSFunction;
    final modulePromise = factory.callAsFunction(null, JSObject()) as JSPromise;
    final module = (await modulePromise.toDart) as JSObject;
    onProgress?.call(0.1);

    // Create directory in MEMFS
    final fs = module.getProperty('FS'.toJS) as JSObject;
    try {
      fs.callMethod('mkdir'.toJS, '/models'.toJS);
    } catch (_) {}

    // Fetch model
    final response = await _jsFetch(modelUrl);
    final contentLength = _getContentLength(response);
    final body = response.getProperty('body'.toJS) as JSObject;
    final reader = body.callMethod('getReader'.toJS) as JSObject;
    final chunks = <Uint8List>[];
    var received = 0;

    while (true) {
      final result = (await (reader.callMethod('read'.toJS) as JSPromise)
          .toDart) as JSObject;
      final done = (result.getProperty('done'.toJS) as JSBoolean).toDart;
      if (done) break;
      final chunk = (result.getProperty('value'.toJS) as JSUint8Array).toDart;
      chunks.add(chunk);
      received += chunk.length;
      if (contentLength > 0) {
        onProgress?.call(0.1 + 0.7 * (received / contentLength));
      }
    }

    final allBytes = Uint8List(received);
    var offset = 0;
    for (final chunk in chunks) {
      allBytes.setRange(offset, offset + chunk.length, chunk);
      offset += chunk.length;
    }

    // Write to MEMFS
    fs.callMethod('writeFile'.toJS, modelPath.toJS, allBytes.toJS);
    onProgress?.call(0.85);

    // Initialize OCR context
    final ctxPtr = (_ccall(module, 'wasm_ocr_init', 'number',
            ['string', 'number'], [modelPath, nThreads]) as num)
        .toInt();
    if (ctxPtr == 0) {
      throw Exception('wasm_ocr_init failed — model may be corrupt');
    }

    if (maxTokens != null) {
      _ccall(module, 'wasm_ocr_set_max_tokens', null, ['number', 'number'],
          [ctxPtr, maxTokens]);
    }

    // Clean up MEMFS
    try {
      fs.callMethod('unlink'.toJS, modelPath.toJS);
    } catch (_) {}

    onProgress?.call(1.0);
    return CrispEmbedOcr._(module: module, ctxPtr: ctxPtr);
  }

  /// Recognize text from raw RGB/RGBA pixel bytes.
  String? recognizeRaw(Uint8List bytes, int width, int height, int channels) {
    if (_disposed) return null;

    final pixelPtr = _callMalloc(_module, bytes.length);
    final lenPtr = _callMalloc(_module, 4);
    try {
      // Copy pixels into WASM heap
      final heap = _module.getProperty('HEAPU8'.toJS) as JSObject;
      final heapBuf = (heap.getProperty('buffer'.toJS) as JSArrayBuffer);
      final view = Uint8List.view(heapBuf.toDart, pixelPtr, bytes.length);
      view.setAll(0, bytes);

      final strPtr = (_ccall(
              _module,
              'wasm_ocr_recognize_copy',
              'number',
              ['number', 'number', 'number', 'number', 'number', 'number'],
              [_ctxPtr, pixelPtr, width, height, channels, lenPtr]) as num)
          .toInt();

      if (strPtr == 0) return null;

      final text =
          (_ccall(_module, 'UTF8ToString', 'string', ['number'], [strPtr])
                  as String?) ??
              '';
      _callFree(_module, strPtr);
      return text;
    } finally {
      _callFree(_module, pixelPtr);
      _callFree(_module, lenPtr);
    }
  }

  /// Recognize text from a grayscale float image.
  String? recognizeGray(Float32List pixels, int width, int height) {
    if (_disposed) return null;

    final nBytes = pixels.length * 4;
    final pixelPtr = _callMalloc(_module, nBytes);
    final lenPtr = _callMalloc(_module, 4);
    try {
      final heapF32 = _module.getProperty('HEAPF32'.toJS) as JSObject;
      final buf = (heapF32.getProperty('buffer'.toJS) as JSArrayBuffer);
      final view = Float32List.view(buf.toDart, pixelPtr, pixels.length);
      view.setAll(0, pixels);

      final result = _ccall(
          _module,
          'wasm_ocr_recognize_gray',
          'string',
          ['number', 'number', 'number', 'number', 'number'],
          [_ctxPtr, pixelPtr, width, height, lenPtr]);

      _callFree(_module, pixelPtr);
      _callFree(_module, lenPtr);
      return result as String?;
    } catch (_) {
      _callFree(_module, pixelPtr);
      _callFree(_module, lenPtr);
      return null;
    }
  }

  /// Mean confidence of the last recognition.
  double get meanConfidence {
    if (_disposed) return 0;
    return (_ccall(_module, 'wasm_ocr_mean_confidence', 'number', ['number'],
            [_ctxPtr]) as num)
        .toDouble();
  }

  /// WASM module version string.
  String get version =>
      _ccall(_module, 'wasm_ocr_version', 'string', [], []) as String;

  void dispose() {
    if (_disposed) return;
    _disposed = true;
    try {
      _ccall(_module, 'wasm_ocr_free', null, ['number'], [_ctxPtr]);
    } catch (_) {}
  }

  // -- JS helpers -----------------------------------------------------------

  static Future<JSObject> _jsFetch(String url) async {
    final fetchFn = globalContext.getProperty('fetch'.toJS) as JSFunction;
    final promise = fetchFn.callAsFunction(null, url.toJS) as JSPromise;
    return (await promise.toDart) as JSObject;
  }

  static int _getContentLength(JSObject response) {
    final headers = response.getProperty('headers'.toJS) as JSObject;
    final cl = headers.callMethod('get'.toJS, 'content-length'.toJS);
    if (cl == null || cl.isUndefinedOrNull) return 0;
    return int.tryParse((cl as JSString).toDart) ?? 0;
  }

  static dynamic _ccall(JSObject module, String name, String? returnType,
      List<String> argTypes, List<dynamic> args) {
    final ccallFn = module.getProperty('ccall'.toJS) as JSFunction;
    final result = ccallFn.callAsFunction(
      null,
      name.toJS,
      returnType?.toJS ?? ''.toJS,
      argTypes.map((t) => t.toJS).toList().toJS,
      args
          .map((a) {
            if (a is int) return a.toJS;
            if (a is String) return a.toJS;
            if (a is double) return a.toJS;
            return a as JSAny;
          })
          .toList()
          .toJS,
    );
    if (returnType == 'number' && result != null) {
      return (result as JSNumber).toDartInt;
    }
    if (returnType == 'string' && result != null) {
      return (result as JSString).toDart;
    }
    return result;
  }

  static int _callMalloc(JSObject module, int size) {
    final mallocFn = module.getProperty('_malloc'.toJS) as JSFunction;
    return (mallocFn.callAsFunction(null, size.toJS) as JSNumber).toDartInt;
  }

  static void _callFree(JSObject module, int ptr) {
    final freeFn = module.getProperty('_free'.toJS) as JSFunction;
    freeFn.callAsFunction(null, ptr.toJS);
  }
}
