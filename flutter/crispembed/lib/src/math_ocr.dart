// CrispEmbed Math OCR — DeiT+TrOCR on-device equation recognition.
//
// Usage:
//   final ocr = CrispEmbedOcr('pix2tex-mfr-q4_k.gguf');
//   final latex = ocr.recognizeGray(grayPixels, width, height);
//   print(latex); // "x^{2} + 1"
//   ocr.dispose();

import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

/// On-device math OCR via CrispEmbed's ggml inference.
class CrispEmbedOcr {
  late final DynamicLibrary _lib;
  late final Pointer<Void> _ctx;
  bool _disposed = false;

  // FFI function types
  typedef _InitC = Pointer<Void> Function(Pointer<Utf8>, Int32);
  typedef _InitDart = Pointer<Void> Function(Pointer<Utf8>, int);
  typedef _FreeC = Void Function(Pointer<Void>);
  typedef _FreeDart = void Function(Pointer<Void>);
  typedef _RecognizeC = Pointer<Utf8> Function(
      Pointer<Void>, Pointer<Float>, Int32, Int32, Pointer<Int32>);
  typedef _RecognizeDart = Pointer<Utf8> Function(
      Pointer<Void>, Pointer<Float>, int, int, Pointer<Int32>);
  typedef _RecognizeRawC = Pointer<Utf8> Function(
      Pointer<Void>, Pointer<Uint8>, Int32, Int32, Int32, Pointer<Int32>);
  typedef _RecognizeRawDart = Pointer<Utf8> Function(
      Pointer<Void>, Pointer<Uint8>, int, int, int, Pointer<Int32>);

  late final _FreeDart _free;
  late final _RecognizeDart _recognize;
  late final _RecognizeRawDart _recognizeRaw;

  /// Load a pix2tex GGUF model for math OCR.
  ///
  /// [modelPath] — path to the `.gguf` file.
  /// [nThreads] — CPU thread count (default 4).
  /// [libPath] — optional path to the shared library.
  CrispEmbedOcr(String modelPath, {int nThreads = 4, String? libPath}) {
    _lib = DynamicLibrary.open(libPath ?? _findLib());

    final init = _lib.lookupFunction<_InitC, _InitDart>(
        'crispembed_math_ocr_init');
    _free = _lib.lookupFunction<_FreeC, _FreeDart>(
        'crispembed_math_ocr_free');
    _recognize = _lib.lookupFunction<_RecognizeC, _RecognizeDart>(
        'math_ocr_recognize');
    _recognizeRaw = _lib.lookupFunction<_RecognizeRawC, _RecognizeRawDart>(
        'math_ocr_recognize_raw');

    final pathPtr = modelPath.toNativeUtf8();
    _ctx = init(pathPtr, nThreads);
    calloc.free(pathPtr);

    if (_ctx == nullptr) {
      throw Exception('Failed to load OCR model: $modelPath');
    }
  }

  /// Recognize math from a grayscale float image.
  /// [pixels] — row-major grayscale floats [0..1], size = width × height.
  String? recognizeGray(Float32List pixels, int width, int height) {
    if (_disposed) return null;
    final ptr = calloc<Float>(pixels.length);
    ptr.asTypedList(pixels.length).setAll(0, pixels);
    final outLen = calloc<Int32>();

    final result = _recognize(_ctx, ptr, width, height, outLen);

    final len = outLen.value;
    calloc.free(ptr);
    calloc.free(outLen);

    if (result == nullptr) return null;
    return result.toDartString(length: len);
  }

  /// Recognize math from raw RGB/RGBA pixel bytes.
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

  static String _findLib() {
    if (Platform.isAndroid || Platform.isLinux) return 'libcrispembed.so';
    if (Platform.isIOS || Platform.isMacOS) return 'crispembed.framework/crispembed';
    if (Platform.isWindows) return 'crispembed.dll';
    throw UnsupportedError('Unsupported platform');
  }
}
