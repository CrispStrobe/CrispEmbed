// CrispEmbed Scan Cleanup — document scan preprocessing.
//
// Tier 1 (classical, no model): deskew, border crop, background whitening.
// Tier 2 (learned, NAFNet GGUF): CNN denoising.
//
// Usage:
//   final cleanup = CrispScanCleanup();  // tier 1 only
//   final result = cleanup.process(imageBytes, width, height, channels: 3);
//   // result.pixels is Uint8List RGB, result.width/height are output dims
//   cleanup.dispose();

import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

// ---------------------------------------------------------------------------
// FFI function types
// ---------------------------------------------------------------------------

typedef _InitC = Pointer<Void> Function(Pointer<Utf8>, Int32);
typedef _InitDart = Pointer<Void> Function(Pointer<Utf8>, int);

typedef _FreeC = Void Function(Pointer<Void>);
typedef _FreeDart = void Function(Pointer<Void>);

typedef _ProcessSimpleC = Int32 Function(
    Pointer<Void>,
    Pointer<Uint8>,
    Int32,
    Int32,
    Int32,
    Int32,
    Int32,
    Int32,
    Int32,
    Pointer<Pointer<Uint8>>,
    Pointer<Int32>,
    Pointer<Int32>);
typedef _ProcessSimpleDart = int Function(
    Pointer<Void>,
    Pointer<Uint8>,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    Pointer<Pointer<Uint8>>,
    Pointer<Int32>,
    Pointer<Int32>);

typedef _FreeImageC = Void Function(Pointer<Uint8>);
typedef _FreeImageDart = void Function(Pointer<Uint8>);

typedef _DetectSplitC = Int32 Function(Pointer<Uint8>, Int32, Int32, Int32);
typedef _DetectSplitDart = int Function(Pointer<Uint8>, int, int, int);

typedef _ContentBboxC = Int32 Function(Pointer<Uint8>, Int32, Int32, Int32,
    Pointer<Int32>, Pointer<Int32>, Pointer<Int32>, Pointer<Int32>);
typedef _ContentBboxDart = int Function(Pointer<Uint8>, int, int, int,
    Pointer<Int32>, Pointer<Int32>, Pointer<Int32>, Pointer<Int32>);

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

/// Result of scan cleanup processing.
class ScanCleanupResult {
  final Uint8List pixels; // RGB uint8
  final int width;
  final int height;

  const ScanCleanupResult({
    required this.pixels,
    required this.width,
    required this.height,
  });
}

// ---------------------------------------------------------------------------
// Main class
// ---------------------------------------------------------------------------

/// Document scan preprocessing (deskew, crop, whiten, denoise).
///
/// Pass a model path for tier-2 learned denoising (NAFNet GGUF),
/// or null for tier-1 classical processing only.
class CrispScanCleanup {
  late final DynamicLibrary _lib;
  late final Pointer<Void> _ctx;
  late final _FreeDart _free;
  late final _FreeImageDart _freeImage;
  late final _ProcessSimpleDart _processSimple;
  late final _DetectSplitDart _detectSplit;
  late final _ContentBboxDart _contentBbox;

  CrispScanCleanup({String? modelPath, int nThreads = 4, String? libPath}) {
    _lib = DynamicLibrary.open(libPath ?? _defaultLibPath());

    final init =
        _lib.lookupFunction<_InitC, _InitDart>('crispembed_scan_cleanup_init');
    _free =
        _lib.lookupFunction<_FreeC, _FreeDart>('crispembed_scan_cleanup_free');
    _freeImage = _lib.lookupFunction<_FreeImageC, _FreeImageDart>(
        'crispembed_scan_cleanup_free_image');
    _processSimple = _lib.lookupFunction<_ProcessSimpleC, _ProcessSimpleDart>(
        'crispembed_scan_cleanup_process_simple');
    _detectSplit = _lib.lookupFunction<_DetectSplitC, _DetectSplitDart>(
        'crispembed_scan_cleanup_detect_page_split');
    _contentBbox = _lib.lookupFunction<_ContentBboxC, _ContentBboxDart>(
        'crispembed_scan_cleanup_content_bbox');

    final pathPtr = modelPath != null
        ? modelPath.toNativeUtf8()
        : Pointer<Utf8>.fromAddress(0);
    _ctx = init(pathPtr, nThreads);
    if (modelPath != null) calloc.free(pathPtr);
    if (_ctx.address == 0) {
      throw Exception('Failed to init scan cleanup');
    }
  }

  /// Process a scan image.
  ///
  /// [pixels] must be RGB (channels=3) or grayscale (channels=1) uint8 data.
  ScanCleanupResult process(
    Uint8List pixels,
    int width,
    int height, {
    int channels = 3,
    bool deskew = true,
    bool cropBorders = true,
    bool whitenBackground = true,
    bool binarize = false,
  }) {
    final pxNative = calloc<Uint8>(pixels.length);
    pxNative.asTypedList(pixels.length).setAll(0, pixels);

    final outPtr = calloc<Pointer<Uint8>>();
    final outW = calloc<Int32>();
    final outH = calloc<Int32>();

    final rc = _processSimple(
      _ctx,
      pxNative,
      width,
      height,
      channels,
      deskew ? 1 : 0,
      cropBorders ? 1 : 0,
      whitenBackground ? 1 : 0,
      binarize ? 1 : 0,
      outPtr,
      outW,
      outH,
    );

    calloc.free(pxNative);

    if (rc != 0 || outPtr.value.address == 0) {
      calloc.free(outPtr);
      calloc.free(outW);
      calloc.free(outH);
      throw Exception('Scan cleanup failed');
    }

    final ow = outW.value;
    final oh = outH.value;
    final resultPixels =
        Uint8List.fromList(outPtr.value.asTypedList(ow * oh * 3));
    _freeImage(outPtr.value);

    calloc.free(outPtr);
    calloc.free(outW);
    calloc.free(outH);

    return ScanCleanupResult(pixels: resultPixels, width: ow, height: oh);
  }

  /// Detect a two-up (double-page) book spread. Returns the gutter column to
  /// split at, or null for a single page.
  int? detectPageSplit(Uint8List pixels, int width, int height,
      {int channels = 3}) {
    final px = calloc<Uint8>(pixels.length);
    px.asTypedList(pixels.length).setAll(0, pixels);
    final sx = _detectSplit(px, width, height, channels);
    calloc.free(px);
    return sx < 0 ? null : sx;
  }

  /// Detect the printed content bounding box (trims blank margins). Returns
  /// [x0, y0, x1, y1] with x1/y1 exclusive, or null for a blank page.
  List<int>? contentBbox(Uint8List pixels, int width, int height,
      {int channels = 3}) {
    final px = calloc<Uint8>(pixels.length);
    px.asTypedList(pixels.length).setAll(0, pixels);
    final x0 = calloc<Int32>(), y0 = calloc<Int32>();
    final x1 = calloc<Int32>(), y1 = calloc<Int32>();
    final rc = _contentBbox(px, width, height, channels, x0, y0, x1, y1);
    final result = rc != 0 ? null : [x0.value, y0.value, x1.value, y1.value];
    calloc.free(px);
    calloc.free(x0);
    calloc.free(y0);
    calloc.free(x1);
    calloc.free(y1);
    return result;
  }

  void dispose() {
    _free(_ctx);
  }

  static String _defaultLibPath() {
    if (Platform.isLinux) return 'libcrispembed.so';
    if (Platform.isMacOS) return 'libcrispembed.dylib';
    if (Platform.isWindows) return 'crispembed.dll';
    return 'libcrispembed.so';
  }
}
