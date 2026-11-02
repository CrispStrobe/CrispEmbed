import 'dart:convert';
import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

final class _Config extends Struct {
  @UintPtr()
  external int structSize;
  external Pointer<Utf8> modelPath;
  external Pointer<Utf8> pdfiumPath;
  external Pointer<Utf8> password;
  @Int32()
  external int mode;
  @Int32()
  external int nThreads;
  @Int32()
  external int firstPage;
  @Int32()
  external int lastPage;
  @Int32()
  external int dpi;
  @Int32()
  external int maxTokens;
  @Int32()
  external int maxLayoutTokens;
  @Int32()
  external int maxBlocks;
  @Int32()
  external int maxPages;
  @Uint64()
  external int maxPixels;
  @Uint64()
  external int maxAssetBytes;
  @Int32()
  external int includePageHeaders;
  @Int32()
  external int extractImages;
}

typedef _InitN = Pointer<Void> Function(Pointer<_Config>);
typedef _InitD = Pointer<Void> Function(Pointer<_Config>);
typedef _FreeN = Void Function(Pointer<Void>);
typedef _FreeD = void Function(Pointer<Void>);
typedef _ConvertN = Int32 Function(Pointer<Void>, Pointer<Utf8>);
typedef _ConvertD = int Function(Pointer<Void>, Pointer<Utf8>);
typedef _RawN = Int32 Function(
  Pointer<Void>,
  Pointer<Uint8>,
  Int32,
  Int32,
  Int32,
);
typedef _RawD = int Function(Pointer<Void>, Pointer<Uint8>, int, int, int);
typedef _RenderN = Pointer<Utf8> Function(Pointer<Void>, Pointer<Utf8>);
typedef _RenderD = Pointer<Utf8> Function(Pointer<Void>, Pointer<Utf8>);
typedef _StringN = Pointer<Utf8> Function(Pointer<Void>);
typedef _StringD = Pointer<Utf8> Function(Pointer<Void>);
typedef _CountN = Int32 Function(Pointer<Void>);
typedef _CountD = int Function(Pointer<Void>);
typedef _NameN = Pointer<Utf8> Function(Pointer<Void>, Int32);
typedef _NameD = Pointer<Utf8> Function(Pointer<Void>, int);
typedef _DataN = Pointer<Uint8> Function(
  Pointer<Void>,
  Int32,
  Pointer<UintPtr>,
);
typedef _DataD = Pointer<Uint8> Function(Pointer<Void>, int, Pointer<UintPtr>);

enum MarkerMode { auto, text, ocr }

void _scalar(String value) {
  if (value.contains('\u0000')) throw ArgumentError('String contains NUL');
  final units = value.codeUnits;
  for (var i = 0; i < units.length; i++) {
    if (units[i] >= 0xd800 && units[i] <= 0xdbff) {
      if (++i >= units.length || units[i] < 0xdc00 || units[i] > 0xdfff) {
        throw ArgumentError('String contains an unpaired surrogate');
      }
    } else if (units[i] >= 0xdc00 && units[i] <= 0xdfff) {
      throw ArgumentError('String contains an unpaired surrogate');
    }
  }
}

/// Native Marker-style document processing, optional PDFium and TeleOCR GGUF.
/// Requires CRISPEMBED_MARKER=1. Auto/OCR also require CRISPEMBED_TELEOCR=1.
/// All paths are local. No Python runtime, model downloads or Surya parity claim.
class CrispMarker {
  late final DynamicLibrary _lib;
  late final _FreeD _free;
  late final _ConvertD _convert;
  late final _RawD _raw;
  late final _RenderD _render;
  late final _StringD _error, _metadata;
  late final _CountD _count;
  late final _NameD _name;
  late final _DataD _data;
  Pointer<Void> _ctx = nullptr;
  final int maxPixels;

  CrispMarker({
    String modelPath = '',
    String pdfiumPath = '',
    MarkerMode mode = MarkerMode.auto,
    int nThreads = 4,
    int firstPage = 0,
    int lastPage = -1,
    int dpi = 144,
    int maxTokens = 1024,
    int maxLayoutTokens = 2048,
    int maxBlocks = 512,
    int maxPages = 1000,
    this.maxPixels = 16000000,
    int maxAssetBytes = 128 * 1024 * 1024,
    bool includePageHeaders = false,
    bool extractImages = true,
    String password = '',
    String? libPath,
  }) {
    for (final text in [modelPath, pdfiumPath, password]) {
      _scalar(text);
    }
    for (final n in [
      nThreads,
      firstPage,
      lastPage,
      dpi,
      maxTokens,
      maxLayoutTokens,
      maxBlocks,
      maxPages,
    ]) {
      if (n < -2147483648 || n > 2147483647)
        throw ArgumentError('Integer exceeds C ABI range');
    }
    if (maxPixels < 1 ||
        maxPixels > 64000000 ||
        maxAssetBytes < 0 ||
        maxAssetBytes > 1024 * 1024 * 1024) {
      throw ArgumentError('Invalid pixel/asset limit');
    }
    _lib = libPath != null
        ? DynamicLibrary.open(libPath)
        : Platform.isIOS
        ? DynamicLibrary.process()
        : DynamicLibrary.open(
            Platform.isWindows
                ? 'crispembed.dll'
                : Platform.isMacOS
                ? 'libcrispembed.dylib'
                : 'libcrispembed.so',
          );
    final init = _lib.lookupFunction<_InitN, _InitD>('crispembed_marker_init');
    _free = _lib.lookupFunction<_FreeN, _FreeD>('crispembed_marker_free');
    _convert = _lib.lookupFunction<_ConvertN, _ConvertD>(
      'crispembed_marker_convert',
    );
    _raw = _lib.lookupFunction<_RawN, _RawD>('crispembed_marker_convert_raw');
    _render = _lib.lookupFunction<_RenderN, _RenderD>(
      'crispembed_marker_render',
    );
    _error = _lib.lookupFunction<_StringN, _StringD>('crispembed_marker_error');
    _metadata = _lib.lookupFunction<_StringN, _StringD>(
      'crispembed_marker_metadata',
    );
    _count = _lib.lookupFunction<_CountN, _CountD>(
      'crispembed_marker_asset_count',
    );
    _name = _lib.lookupFunction<_NameN, _NameD>('crispembed_marker_asset_name');
    _data = _lib.lookupFunction<_DataN, _DataD>('crispembed_marker_asset_data');
    final c = calloc<_Config>();
    final model = modelPath.toNativeUtf8(),
        pdfium = pdfiumPath.toNativeUtf8(),
        pass = password.toNativeUtf8();
    try {
      c.ref
        ..structSize = sizeOf<_Config>()
        ..modelPath = model
        ..pdfiumPath = pdfium
        ..password = pass
        ..mode = mode.index
        ..nThreads = nThreads
        ..firstPage = firstPage
        ..lastPage = lastPage
        ..dpi = dpi
        ..maxTokens = maxTokens
        ..maxLayoutTokens = maxLayoutTokens
        ..maxBlocks = maxBlocks
        ..maxPages = maxPages
        ..maxPixels = maxPixels
        ..maxAssetBytes = maxAssetBytes
        ..includePageHeaders = includePageHeaders ? 1 : 0
        ..extractImages = extractImages ? 1 : 0;
      _ctx = init(c);
      if (_ctx == nullptr) throw StateError(_errorText());
    } finally {
      calloc.free(c);
      calloc.free(model);
      calloc.free(pdfium);
      calloc.free(pass);
    }
  }

  String _errorText() {
    final p = _error(_ctx);
    return p == nullptr ? 'Marker operation failed' : p.toDartString();
  }

  void _open() {
    if (_ctx == nullptr) throw StateError('Marker context is disposed');
  }

  String convert(String path, {String format = 'markdown'}) {
    _open();
    _scalar(path);
    final p = path.toNativeUtf8();
    try {
      if (_convert(_ctx, p) == 0) throw StateError(_errorText());
    } finally {
      calloc.free(p);
    }
    return render(format);
  }

  String convertImage(
    Uint8List pixels,
    int width,
    int height,
    int channels, {
    String format = 'markdown',
  }) {
    _open();
    if (width < 1 ||
        height < 1 ||
        width > 32768 ||
        height > 32768 ||
        ![1, 3, 4].contains(channels) ||
        width * height > maxPixels ||
        width * height * channels != pixels.length) {
      throw ArgumentError('Invalid raw image dimensions or byte length');
    }
    final p = calloc<Uint8>(pixels.length);
    try {
      p.asTypedList(pixels.length).setAll(0, pixels);
      if (_raw(_ctx, p, width, height, channels) == 0)
        throw StateError(_errorText());
    } finally {
      calloc.free(p);
    }
    return render(format);
  }

  String render([String format = 'markdown']) {
    _open();
    _scalar(format);
    final f = format.toNativeUtf8();
    try {
      final p = _render(_ctx, f);
      if (p == nullptr) throw StateError(_errorText());
      return p.toDartString();
    } finally {
      calloc.free(f);
    }
  }

  Map<String, dynamic> get metadata {
    _open();
    final p = _metadata(_ctx);
    if (p == nullptr) throw StateError(_errorText());
    return jsonDecode(p.toDartString()) as Map<String, dynamic>;
  }

  Map<String, Uint8List> get assets {
    _open();
    final out = <String, Uint8List>{};
    final size = calloc<UintPtr>();
    try {
      for (var i = 0; i < _count(_ctx); i++) {
        final name = _name(_ctx, i), p = _data(_ctx, i, size);
        if (name == nullptr || (p == nullptr && size.value != 0))
          throw StateError('Invalid native asset');
        out[name.toDartString()] = size.value == 0
            ? Uint8List(0)
            : Uint8List.fromList(p.asTypedList(size.value));
      }
      return out;
    } finally {
      calloc.free(size);
    }
  }

  void dispose() {
    if (_ctx != nullptr) {
      _free(_ctx);
      _ctx = nullptr;
    }
  }
}
