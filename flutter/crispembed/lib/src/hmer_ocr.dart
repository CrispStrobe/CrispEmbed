// CrispEmbed HMER — Handwritten Math Expression Recognition.
//
// DenseNet-121 encoder + GRU attention decoder, trained on CROHME 2016.
// 112 LaTeX tokens, ~6.8M params, 26 MB F32 / ~4 MB Q4_K.
//
// Usage:
//   final ocr = CrispEmbedHmerOcr('hmer-hw-f32.gguf');
//   final latex = ocr.recognizeGray(grayPixels, width, height);
//   print(latex); // "\log _ { 2 }"
//   ocr.dispose();

import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

// FFI function types
typedef _HmerInitC = Pointer<Void> Function(Pointer<Utf8>, Int32);
typedef _HmerInitDart = Pointer<Void> Function(Pointer<Utf8>, int);
typedef _HmerFreeC = Void Function(Pointer<Void>);
typedef _HmerFreeDart = void Function(Pointer<Void>);
typedef _HmerRecognizeGrayC = Pointer<Utf8> Function(
    Pointer<Void>, Pointer<Float>, Int32, Int32, Pointer<Int32>);
typedef _HmerRecognizeGrayDart = Pointer<Utf8> Function(
    Pointer<Void>, Pointer<Float>, int, int, Pointer<Int32>);
typedef _HmerRecognizeRawC = Pointer<Utf8> Function(
    Pointer<Void>, Pointer<Uint8>, Int32, Int32, Int32, Pointer<Int32>);
typedef _HmerRecognizeRawDart = Pointer<Utf8> Function(
    Pointer<Void>, Pointer<Uint8>, int, int, int, Pointer<Int32>);

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

/// On-device handwritten math OCR via CrispEmbed's HMER model.
class CrispEmbedHmerOcr {
  late final DynamicLibrary _lib;
  late final Pointer<Void> _ctx;
  bool _disposed = false;

  late final _HmerFreeDart _free;
  late final _HmerRecognizeGrayDart _recognizeGray;
  late final _HmerRecognizeRawDart _recognizeRaw;

  /// Load an HMER GGUF model for handwritten math OCR.
  ///
  /// [modelPath] — path to the `.gguf` file.
  /// [nThreads] — CPU thread count (default 4).
  /// [libPath] — optional path to the shared library.
  CrispEmbedHmerOcr(String modelPath, {int nThreads = 4, String? libPath}) {
    _lib = _openLib(libPath);

    final init = _lib.lookupFunction<_HmerInitC, _HmerInitDart>(
        'crispembed_hmer_ocr_init');
    _free = _lib.lookupFunction<_HmerFreeC, _HmerFreeDart>(
        'crispembed_hmer_ocr_free');
    _recognizeGray =
        _lib.lookupFunction<_HmerRecognizeGrayC, _HmerRecognizeGrayDart>(
            'crispembed_hmer_ocr_recognize_gray');
    _recognizeRaw =
        _lib.lookupFunction<_HmerRecognizeRawC, _HmerRecognizeRawDart>(
            'crispembed_hmer_ocr_recognize');

    final pathPtr = modelPath.toNativeUtf8();
    _ctx = init(pathPtr, nThreads);
    calloc.free(pathPtr);

    if (_ctx == nullptr) {
      throw Exception('Failed to load HMER model: $modelPath');
    }
  }

  /// Recognize handwritten math from a grayscale float image.
  /// [pixels] — row-major grayscale floats [0..1], size = width × height.
  String? recognizeGray(Float32List pixels, int width, int height) {
    if (_disposed) return null;
    final ptr = calloc<Float>(pixels.length);
    ptr.asTypedList(pixels.length).setAll(0, pixels);
    final outLen = calloc<Int32>();

    final result = _recognizeGray(_ctx, ptr, width, height, outLen);

    final len = outLen.value;
    calloc.free(ptr);
    calloc.free(outLen);

    if (result == nullptr) return null;
    return result.toDartString(length: len);
  }

  /// Recognize handwritten math from raw RGB/RGBA pixel bytes.
  /// [bytes] — raw pixel data, [channels] = 1/3/4.
  String? recognizeRaw(Uint8List bytes, int width, int height, int channels) {
    if (_disposed) return null;
    final ptr = calloc<Uint8>(bytes.length);
    ptr.asTypedList(bytes.length).setAll(0, bytes);
    final outLen = calloc<Int32>();

    final result = _recognizeRaw(_ctx, ptr, width, height, channels, outLen);

    final len = outLen.value;
    calloc.free(ptr);
    calloc.free(outLen);

    if (result == nullptr) return null;
    return result.toDartString(length: len);
  }

  void dispose() {
    if (!_disposed) {
      _free(_ctx);
      _disposed = true;
    }
  }
}
