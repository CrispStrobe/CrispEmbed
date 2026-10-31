import 'dart:convert';
import 'dart:ffi';
import 'dart:typed_data';
import 'package:ffi/ffi.dart';
import 'crispembed_bindings.dart';

/// Resident OXR recognition/read-order model. Requires CRISPEMBED_OXR=1.
/// Calls are synchronous; use a worker isolate for UI applications. Layout
/// detection and full document postprocessing are separate operations.
class CrispOXR {
  Pointer<Void> _ctx = nullptr;
  late final OxrFree _free;
  late final OxrPredict _predict;
  late final OxrString _error;

  CrispOXR(DynamicLibrary lib, String modelPath, {int nThreads = 4}) {
    if (modelPath.isEmpty || modelPath.contains('\u0000')) {
      throw ArgumentError('Model path is empty or contains NUL');
    }
    if (nThreads < 0 || nThreads > 2147483647) throw ArgumentError('Invalid thread count');
    final init = lib.lookupFunction<OxrInitNative, OxrInit>('crispembed_oxr_init');
    _free = lib.lookupFunction<OxrFreeNative, OxrFree>('crispembed_oxr_free');
    _predict = lib.lookupFunction<OxrPredictNative, OxrPredict>('crispembed_oxr_predict');
    _error = lib.lookupFunction<OxrStringNative, OxrString>('crispembed_oxr_last_error');
    final path = modelPath.toNativeUtf8();
    try { _ctx = init(path, nThreads); } finally { calloc.free(path); }
    if (_ctx == nullptr) throw StateError('OXR model/profile/gate initialization failed');
  }

  void _check(String prompt, int maxTokens, double penalty) {
    if (_ctx == nullptr) throw StateError('OXR context is disposed');
    if (maxTokens < 0 || maxTokens > 32768 || !penalty.isFinite || penalty <= 0) {
      throw ArgumentError('Invalid token cap or repetition penalty');
    }
    if (prompt.contains('\u0000') || utf8.encode(prompt).length > 65536) {
      throw ArgumentError('Prompt contains NUL or exceeds 65536 bytes');
    }
  }

  Map<String, dynamic> _result(Pointer<Utf8> result) {
    if (result == nullptr) {
      final error = _error(_ctx);
      throw StateError(error == nullptr ? 'OXR inference failed' : error.toDartString());
    }
    return jsonDecode(result.toDartString()) as Map<String, dynamic>;
  }

  /// JSON retains EOS/cap and IDs. Capped text is not complete document output.
  Map<String, dynamic> predictRgb(Uint8List rgb, int width, int height, String prompt,
      {int maxTokens = 0, double repetitionPenalty = 1.0}) {
    _check(prompt, maxTokens, repetitionPenalty);
    if (width <= 0 || height <= 0 || width > 2147483647 || height > 2147483647 ||
        width * height > 100000000 || rgb.length != width * height * 3) {
      throw ArgumentError('RGB byte length does not match valid image dimensions');
    }
    final pixels = calloc<Uint8>(rgb.length);
    final text = prompt.toNativeUtf8();
    try {
      pixels.asTypedList(rgb.length).setAll(0, rgb);
      return _result(_predict(_ctx, pixels, width, height, 3, text, maxTokens, repetitionPenalty));
    } finally { calloc.free(text); calloc.free(pixels); }
  }

  Map<String, dynamic> predictText(String prompt,
      {int maxTokens = 0, double repetitionPenalty = 1.0}) {
    _check(prompt, maxTokens, repetitionPenalty);
    final text = prompt.toNativeUtf8();
    try { return _result(_predict(_ctx, nullptr, 0, 0, 0, text, maxTokens, repetitionPenalty)); }
    finally { calloc.free(text); }
  }

  /// Native C++ document route (requires CRISPEMBED_OXR_NATIVE_DOC=1): the
  /// translated pinned OXR document helper over a supplied D-FINE batch
  /// response. Each page is tightly packed RGB. Returns the upstream-shaped
  /// document (outputs.json, outputs.markdown, metadata).
  /// Attach the native OXR D-FINE GGUF as the default layout provider; then
  /// pass `layoutResponse: null` to [processPagesRgb].
  void setDocumentLayoutModel(DynamicLibrary lib, String modelPath, {int nThreads = 4}) {
    final attach = lib.lookupFunction<OxrAttachLayoutNative, OxrAttachLayout>('crispembed_oxr_document_layout_model');
    final p = modelPath.toNativeUtf8();
    try {
      if (attach(_ctx, p, nThreads) != 1) throw StateError(_error(_ctx).toDartString());
    } finally { calloc.free(p); }
  }

  Map<String, dynamic> processPagesRgb(DynamicLibrary lib, List<Uint8List> pages, List<int> widths,
      List<int> heights, Map<String, dynamic>? layoutResponse, {Map<String, dynamic>? options}) {
    if (_ctx == nullptr) throw StateError('OXR context is disposed');
    final n = pages.length;
    if (n == 0 || widths.length != n || heights.length != n) throw ArgumentError('Page lists must match');
    final process = lib.lookupFunction<OxrProcessPagesNative, OxrProcessPages>('crispembed_oxr_process_pages');
    final ptrs = calloc<Pointer<Uint8>>(n);
    final w = calloc<Int32>(n), h = calloc<Int32>(n), c = calloc<Int32>(n);
    final Pointer<Utf8> layout = layoutResponse == null ? nullptr : jsonEncode(layoutResponse).toNativeUtf8();
    final opts = jsonEncode(options ?? {}).toNativeUtf8();
    try {
      for (var i = 0; i < n; i++) {
        if (widths[i] <= 0 || heights[i] <= 0 || pages[i].length != widths[i] * heights[i] * 3) {
          throw ArgumentError('RGB byte length does not match page $i dimensions');
        }
        ptrs[i] = calloc<Uint8>(pages[i].length);
        ptrs[i].asTypedList(pages[i].length).setAll(0, pages[i]);
        w[i] = widths[i];
        h[i] = heights[i];
        c[i] = 3;
      }
      return _result(process(_ctx, ptrs, w, h, c, n, nullptr, nullptr, layout, opts));
    } finally {
      for (var i = 0; i < n; i++) {
        if (ptrs[i] != nullptr) calloc.free(ptrs[i]);
      }
      calloc.free(ptrs); calloc.free(w); calloc.free(h); calloc.free(c);
      if (layout != nullptr) calloc.free(layout);
      calloc.free(opts);
    }
  }

  /// Every native request of the last document call.
  List<dynamic> lastTrace(DynamicLibrary lib) {
    final trace = lib.lookupFunction<OxrStringNative, OxrString>('crispembed_oxr_last_trace');
    return jsonDecode(trace(_ctx).toDartString()) as List<dynamic>;
  }

  /// Render the last native document via the shared document model.
  String renderDocument(DynamicLibrary lib, String format) {
    final render = lib.lookupFunction<OxrRenderNative, OxrRender>('crispembed_oxr_render_document');
    final f = format.toNativeUtf8();
    try {
      final r = render(_ctx, f);
      if (r == nullptr) throw StateError(_error(_ctx).toDartString());
      return r.toDartString();
    } finally { calloc.free(f); }
  }

  void dispose() {
    if (_ctx != nullptr) { _free(_ctx); _ctx = nullptr; }
  }
}
