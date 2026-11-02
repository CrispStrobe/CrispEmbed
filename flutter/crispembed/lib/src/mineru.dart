import 'dart:convert';
import 'dart:ffi';
import 'dart:typed_data';
import 'package:ffi/ffi.dart';
import 'crispembed_bindings.dart';

/// Resident MinerU2.5-Pro native GGML model. Calls are synchronous; use a worker
/// isolate for UI applications. Requires CRISPEMBED_MINERU25_PRO=1.
class CrispMinerU {
  Pointer<Void> _ctx = nullptr;
  late final MineruFree _free;
  late final MineruPredict _predict;
  late final MineruExtract _extract;
  late final MineruString _error;
  late final MineruDocument _document;
  late final MineruRender _render;

  CrispMinerU(DynamicLibrary lib, String modelPath, {int nThreads = 4}) {
    if (modelPath.contains('\u0000')) throw ArgumentError('NUL in model path');
    final init = lib.lookupFunction<MineruInitNative, MineruInit>('crispembed_mineru_init');
    _free = lib.lookupFunction<MineruFreeNative, MineruFree>('crispembed_mineru_free');
    _predict = lib.lookupFunction<MineruPredictNative, MineruPredict>('crispembed_mineru_predict');
    _extract = lib.lookupFunction<MineruExtractNative, MineruExtract>('crispembed_mineru_extract');
    _error = lib.lookupFunction<MineruStringNative, MineruString>('crispembed_mineru_last_error');
    _document = lib.lookupFunction<MineruDocumentNative, MineruDocument>('crispembed_mineru_extract_document');
    _render = lib.lookupFunction<MineruRenderNative, MineruRender>('crispembed_mineru_last_document');
    final path = modelPath.toNativeUtf8();
    try { _ctx = init(path, nThreads); } finally { calloc.free(path); }
    if (_ctx == nullptr) throw StateError('MinerU model/profile/gate initialization failed');
  }

  void _check(int maxTokens) {
    if (_ctx == nullptr) throw StateError('MinerU context is disposed');
    if (maxTokens < 0 || maxTokens > 8192) throw ArgumentError('maxTokens must be in [0,8192]');
  }

  Map<String, dynamic> _result(Pointer<Utf8> result) {
    if (result == nullptr) throw StateError(_error(_ctx).toDartString());
    // Copy before freeing temporaries or starting the next native call.
    return jsonDecode(result.toDartString()) as Map<String, dynamic>;
  }

  Map<String, dynamic> _pixels(Uint8List rgb, int width, int height,
      Map<String, dynamic> Function(Pointer<Uint8>) operation) {
    if (width <= 0 || height <= 0 || width > 2147483647 || height > 2147483647 ||
        width * height > 100000000 || rgb.length != width * height * 3) {
      throw ArgumentError('RGB byte length does not match valid image dimensions');
    }
    final pixels = calloc<Uint8>(rgb.length);
    try { pixels.asTypedList(rgb.length).setAll(0, rgb); return operation(pixels); }
    finally { calloc.free(pixels); }
  }

  /// One region request; layout/crop preparation is the caller's responsibility.
  Map<String, dynamic> predictRgb(Uint8List rgb, int width, int height, String prompt,
      {int maxTokens = 0}) {
    _check(maxTokens);
    if (prompt.contains('\u0000')) throw ArgumentError('NUL in prompt');
    final text = prompt.toNativeUtf8();
    try { return _pixels(rgb, width, height,
      (pixels) => _result(_predict(_ctx, pixels, width, height, 3, text, maxTokens))); }
    finally { calloc.free(text); }
  }

  Map<String, dynamic> predictText(String prompt, {int maxTokens = 0}) {
    _check(maxTokens);
    if (prompt.contains('\u0000')) throw ArgumentError('NUL in prompt');
    final text = prompt.toNativeUtf8();
    try { return _result(_predict(_ctx, nullptr, 0, 0, 0, text, maxTokens)); }
    finally { calloc.free(text); }
  }

  /// Native basic two-step page JSON. Full document postprocessing uses the
  /// Python CrispMinerU adapter, as documented in docs/mineru25-pro.md.
  Map<String, dynamic> extractNativeBasic(Uint8List rgb, int width, int height,
      {bool imageAnalysis = false, int maxTokens = 0}) {
    _check(maxTokens);
    return _pixels(rgb, width, height, (pixels) =>
      _result(_extract(_ctx, pixels, width, height, 3, imageAnalysis ? 1 : 0, maxTokens)));
  }

  /// Complete native document route (default on; CRISPEMBED_MINERU25_PRO_FULL_DOC=0 disables it).
  /// "blocks" and "markdown" are byte-identical to mineru-vl-utils output.
  Map<String, dynamic> extractDocument(Uint8List rgb, int width, int height,
      {Map<String, dynamic> options = const {}, int maxTokens = 0}) {
    _check(maxTokens);
    final opts = jsonEncode(options).toNativeUtf8();
    try {
      return _pixels(rgb, width, height,
          (pixels) => _result(_document(_ctx, pixels, width, height, 3, opts, maxTokens)));
    } finally { calloc.free(opts); }
  }

  /// Render the last extractDocument page: mineru-json, mineru-markdown,
  /// markdown, html, json or chunks (Marker renderers).
  String lastDocument(String format) {
    if (_ctx == nullptr) throw StateError('MinerU context is disposed');
    final f = format.toNativeUtf8();
    try {
      final r = _render(_ctx, f);
      if (r == nullptr) throw StateError(_error(_ctx).toDartString());
      return r.toDartString();
    } finally { calloc.free(f); }
  }

  void dispose() { if (_ctx != nullptr) { _free(_ctx); _ctx = nullptr; } }
}
