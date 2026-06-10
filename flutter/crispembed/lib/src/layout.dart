// CrispEmbed Layout Detection — document region detection.
//
// Detects layout regions (text, figure, table, formula, etc.) in document
// images and returns bounding boxes with class labels and confidence scores.
//
// Usage:
//   final layout = CrispLayout('layout-picodet.gguf');
//   final regions = layout.detect('page.jpg');
//   for (final r in regions) {
//     print('${r.labelName}: (${r.x1}, ${r.y1}) → (${r.x2}, ${r.y2})  score=${r.score}');
//   }
//   layout.dispose();

import 'dart:ffi';
import 'dart:io';

import 'package:ffi/ffi.dart';

// ---------------------------------------------------------------------------
// FFI struct
// ---------------------------------------------------------------------------

/// Mirror of the C struct `CrispLayoutRegion`.
///
/// All coordinate fields are absolute pixel values in the input image.
final class CrispLayoutRegion extends Struct {
  @Float()
  external double x1;

  @Float()
  external double y1;

  @Float()
  external double x2;

  @Float()
  external double y2;

  @Float()
  external double score;

  @Int32()
  external int label;

  external Pointer<Utf8> labelName;
}

// ---------------------------------------------------------------------------
// FFI function types
// ---------------------------------------------------------------------------

typedef _LayoutInitC = Pointer<Void> Function(Pointer<Utf8>, Int32);
typedef _LayoutInitDart = Pointer<Void> Function(Pointer<Utf8>, int);

typedef _LayoutFreeC = Void Function(Pointer<Void>);
typedef _LayoutFreeDart = void Function(Pointer<Void>);

typedef _LayoutDetectC = Pointer<CrispLayoutRegion> Function(
    Pointer<Void>, Pointer<Utf8>, Float, Pointer<Int32>);
typedef _LayoutDetectDart = Pointer<CrispLayoutRegion> Function(
    Pointer<Void>, Pointer<Utf8>, double, Pointer<Int32>);

// ---------------------------------------------------------------------------
// Dart helper types
// ---------------------------------------------------------------------------

/// A single layout region returned by [CrispLayout.detect].
///
/// Coordinates are in absolute pixels of the input image.
/// [label] is the integer class index; [labelName] is the human-readable
/// class name (e.g. `'text'`, `'figure'`, `'table'`, `'formula'`).
class LayoutRegion {
  final double x1;
  final double y1;
  final double x2;
  final double y2;
  final double score;
  final int label;
  final String labelName;

  const LayoutRegion({
    required this.x1,
    required this.y1,
    required this.x2,
    required this.y2,
    required this.score,
    required this.label,
    required this.labelName,
  });

  @override
  String toString() {
    final coords =
        '(${x1.toStringAsFixed(1)}, ${y1.toStringAsFixed(1)}) → (${x2.toStringAsFixed(1)}, ${y2.toStringAsFixed(1)})';
    return 'LayoutRegion($labelName [$label] $coords score=${score.toStringAsFixed(3)})';
  }
}

// ---------------------------------------------------------------------------
// Library loader
// ---------------------------------------------------------------------------

DynamicLibrary _openLayoutLib([String? libPath]) {
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

/// On-device document layout detection via CrispEmbed.
///
/// Detects regions such as text blocks, figures, tables, and formulae in
/// document page images. Backed by a PicoDet / RT-DETR style model compiled
/// into `libcrispembed`.
///
/// ```dart
/// final layout = CrispLayout('layout-picodet.gguf');
/// final regions = layout.detect('page.jpg', threshold: 0.4);
/// for (final r in regions) {
///   print(r); // LayoutRegion(text [0] (12.0, 34.0) → (590.0, 98.0) score=0.921)
/// }
/// layout.dispose();
/// ```
class CrispLayout {
  late final DynamicLibrary _lib;
  late final Pointer<Void> _ctx;
  bool _disposed = false;

  late final _LayoutFreeDart _free;
  late final _LayoutDetectDart _detect;

  /// Load a layout detection GGUF model.
  ///
  /// [modelPath] — path to the `.gguf` model file.
  /// [nThreads] — CPU thread count (0 = auto-detect).
  /// [libPath] — optional path to the shared library. If omitted, searches
  ///   standard platform locations.
  CrispLayout(String modelPath, {int nThreads = 0, String? libPath}) {
    _lib = _openLayoutLib(libPath);

    final init = _lib.lookupFunction<_LayoutInitC, _LayoutInitDart>(
        'crispembed_layout_init');
    _free = _lib.lookupFunction<_LayoutFreeC, _LayoutFreeDart>(
        'crispembed_layout_free');
    _detect = _lib.lookupFunction<_LayoutDetectC, _LayoutDetectDart>(
        'crispembed_layout_detect');

    final pathPtr = modelPath.toNativeUtf8();
    _ctx = init(pathPtr, nThreads);
    calloc.free(pathPtr);

    if (_ctx == nullptr) {
      throw Exception('Failed to load layout model: $modelPath');
    }
  }

  // ------------------------------------------------------------------
  // Inference
  // ------------------------------------------------------------------

  /// Detect layout regions in an image file.
  ///
  /// [imagePath] — path to a JPEG/PNG file readable by the native library.
  /// [threshold] — minimum confidence threshold (default 0.3).
  ///
  /// Returns a list of [LayoutRegion]s sorted by confidence descending.
  /// Returns an empty list if no regions exceed [threshold] or on error.
  List<LayoutRegion> detect(String imagePath, {double threshold = 0.3}) {
    _checkDisposed();
    final pathPtr = imagePath.toNativeUtf8();
    final countPtr = calloc<Int32>();
    try {
      final buf = _detect(_ctx, pathPtr, threshold, countPtr);
      final n = countPtr.value;
      if (buf == nullptr || n <= 0) return [];
      return _decodeLayoutBuffer(buf, n);
    } finally {
      calloc.free(pathPtr);
      calloc.free(countPtr);
    }
  }

  // ------------------------------------------------------------------
  // Lifecycle
  // ------------------------------------------------------------------

  /// Release all native resources. Must be called when done.
  void dispose() {
    if (!_disposed) {
      _free(_ctx);
      _disposed = true;
    }
  }

  void _checkDisposed() {
    if (_disposed) throw StateError('CrispLayout has been disposed');
  }
}

// ---------------------------------------------------------------------------
// Internal decode helper
// ---------------------------------------------------------------------------

/// Decode the packed `CrispLayoutRegion[]` buffer returned by
/// `crispembed_layout_detect`.
List<LayoutRegion> _decodeLayoutBuffer(
    Pointer<CrispLayoutRegion> buf, int n) {
  final results = <LayoutRegion>[];
  for (var i = 0; i < n; i++) {
    final r = buf[i];
    results.add(LayoutRegion(
      x1: r.x1,
      y1: r.y1,
      x2: r.x2,
      y2: r.y2,
      score: r.score,
      label: r.label,
      labelName: r.labelName == nullptr ? '' : r.labelName.toDartString(),
    ));
  }
  results.sort((a, b) => b.score.compareTo(a.score));
  return results;
}
