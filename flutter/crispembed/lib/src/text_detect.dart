// CrispEmbed Text Detection — text region detection via Surya/DBNet.
//
// Detects text line bounding boxes in document images using segmentation
// models (Surya-OCR-2 EfficientViT or DBNet ResNet-18).
//
// Usage:
//   final det = CrispTextDetect('surya-det-q8_0.gguf');
//   final regions = det.detect(imageBytes, width, height, channels: 3);
//   for (final r in regions) {
//     print('(${r.x0}, ${r.y0}) → (${r.x1}, ${r.y1})  conf=${r.confidence}');
//   }
//   det.dispose();

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

// crispembed_text_det(ctx, pixels, w, h, channels, text_thresh, low_thresh, out_n)
typedef _DetectC = Pointer<Float> Function(
    Pointer<Void>, Pointer<Uint8>, Int32, Int32, Int32, Float, Float,
    Pointer<Int32>);
typedef _DetectDart = Pointer<Float> Function(
    Pointer<Void>, Pointer<Uint8>, int, int, int, double, double,
    Pointer<Int32>);

// ---------------------------------------------------------------------------
// Dart result type
// ---------------------------------------------------------------------------

/// A single text detection result — bounding box in original image coordinates.
class TextDetRegion {
  final double x0;
  final double y0;
  final double x1;
  final double y1;
  final double confidence;

  const TextDetRegion({
    required this.x0,
    required this.y0,
    required this.x1,
    required this.y1,
    required this.confidence,
  });

  @override
  String toString() {
    final coords =
        '(${x0.toStringAsFixed(1)}, ${y0.toStringAsFixed(1)}) → '
        '(${x1.toStringAsFixed(1)}, ${y1.toStringAsFixed(1)})';
    return 'TextDetRegion($coords conf=${confidence.toStringAsFixed(3)})';
  }
}

// ---------------------------------------------------------------------------
// Library loader
// ---------------------------------------------------------------------------

DynamicLibrary _openLib([String? libPath]) {
  if (libPath != null) return DynamicLibrary.open(libPath);
  if (Platform.isIOS) return DynamicLibrary.process();
  if (Platform.isAndroid || Platform.isLinux) {
    return DynamicLibrary.open('libcrispembed.so');
  }
  if (Platform.isMacOS) return DynamicLibrary.open('libcrispembed.dylib');
  if (Platform.isWindows) return DynamicLibrary.open('crispembed.dll');
  return DynamicLibrary.open('libcrispembed.so');
}

// ---------------------------------------------------------------------------
// High-level wrapper
// ---------------------------------------------------------------------------

/// On-device text detection via CrispEmbed (Surya-OCR-2 or DBNet).
///
/// Detects text line bounding boxes in images. Backed by segmentation
/// models compiled into `libcrispembed`.
///
/// ```dart
/// final det = CrispTextDetect('surya-det-q8_0.gguf');
/// final regions = det.detect(imageBytes, 640, 480, channels: 3);
/// for (final r in regions) {
///   print(r); // TextDetRegion((12.0, 34.0) → (590.0, 98.0) conf=0.921)
/// }
/// det.dispose();
/// ```
class CrispTextDetect {
  late final DynamicLibrary _lib;
  late final Pointer<Void> _ctx;
  bool _disposed = false;

  late final _FreeDart _free;
  late final _DetectDart _detect;

  /// Load a text detection GGUF model.
  ///
  /// [modelPath] — path to the `.gguf` model file (Surya-det or DBNet).
  /// [nThreads] — CPU thread count (0 = auto-detect).
  /// [libPath] — optional path to the shared library.
  CrispTextDetect(String modelPath, {int nThreads = 0, String? libPath}) {
    _lib = _openLib(libPath);

    final init =
        _lib.lookupFunction<_InitC, _InitDart>('crispembed_text_det_init');
    _free =
        _lib.lookupFunction<_FreeC, _FreeDart>('crispembed_text_det_free');
    _detect =
        _lib.lookupFunction<_DetectC, _DetectDart>('crispembed_text_det');

    final pathPtr = modelPath.toNativeUtf8();
    _ctx = init(pathPtr, nThreads);
    calloc.free(pathPtr);

    if (_ctx == nullptr) {
      throw Exception('Failed to load text detection model: $modelPath');
    }
  }

  // ------------------------------------------------------------------
  // Inference
  // ------------------------------------------------------------------

  /// Detect text regions in raw pixel data.
  ///
  /// [pixels] — raw pixel bytes (RGB, RGBA, or grayscale).
  /// [width], [height] — image dimensions.
  /// [channels] — 1 (gray), 3 (RGB), or 4 (RGBA).
  /// [textThreshold] — confidence threshold for text regions (default 0.5).
  /// [lowThreshold] — low-confidence threshold for region expansion (default 0.3).
  ///
  /// Returns a list of [TextDetRegion]s sorted by confidence descending.
  List<TextDetRegion> detect(
    Uint8List pixels,
    int width,
    int height, {
    int channels = 3,
    double textThreshold = 0.5,
    double lowThreshold = 0.3,
  }) {
    _checkDisposed();
    final pixPtr = calloc<Uint8>(pixels.length);
    pixPtr.asTypedList(pixels.length).setAll(0, pixels);
    final countPtr = calloc<Int32>();
    try {
      final buf = _detect(
          _ctx, pixPtr, width, height, channels,
          textThreshold, lowThreshold, countPtr);
      final n = countPtr.value;
      if (buf == nullptr || n <= 0) return [];
      return _decodeResults(buf, n);
    } finally {
      calloc.free(pixPtr);
      calloc.free(countPtr);
    }
  }

  // ------------------------------------------------------------------
  // Lifecycle
  // ------------------------------------------------------------------

  /// Release all native resources.
  void dispose() {
    if (!_disposed) {
      _free(_ctx);
      _disposed = true;
    }
  }

  void _checkDisposed() {
    if (_disposed) throw StateError('CrispTextDetect has been disposed');
  }
}

// ---------------------------------------------------------------------------
// Internal decode helper
// ---------------------------------------------------------------------------

/// Decode the packed `crispembed_text_det_result[]` buffer.
/// Each result is 5 floats: x0, y0, x1, y1, confidence.
List<TextDetRegion> _decodeResults(Pointer<Float> buf, int n) {
  final results = <TextDetRegion>[];
  for (var i = 0; i < n; i++) {
    final base = i * 5;
    results.add(TextDetRegion(
      x0: buf[base],
      y0: buf[base + 1],
      x1: buf[base + 2],
      y1: buf[base + 3],
      confidence: buf[base + 4],
    ));
  }
  results.sort((a, b) => b.confidence.compareTo(a.confidence));
  return results;
}
