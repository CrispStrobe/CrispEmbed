// CrispEmbed OMR — on-device Optical Music Recognition (sheet music → tokens).
//
// Wraps the same auto-detecting OCR dispatcher as the other recognizers, but
// named/documented for OMR. Works with any OMR GGUF whose `general.architecture`
// the engine recognizes (`smt_ocr` — Sheet Music Transformer → bekern;
// `tromr_ocr` — Polyphonic-TrOMR → rhythm/pitch/lift streams; `flova_ocr` —
// handwritten music → LilyPond; `transcoda_ocr` — zero-shot full-page score →
// Humdrum `**kern`). The model is auto-detected from the GGUF, so the same
// class handles every OMR engine.
//
// Usage (Flutter):
//   final omr = CrispEmbedOmr('smt-grandstaff-q8_0.gguf');
//   // decode your PNG/JPEG to raw RGB bytes (e.g. via the `image` package):
//   final tokens = omr.recognize(rgbBytes, width, height, 3);
//   print(tokens); // "**ekern_1.0 <t> **ekern_1.0 <b> *clefG2 ..."
//   omr.dispose();

import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

typedef _OmrInitC = Pointer<Void> Function(Pointer<Utf8>, Int32);
typedef _OmrInitDart = Pointer<Void> Function(Pointer<Utf8>, int);
typedef _OmrFreeC = Void Function(Pointer<Void>);
typedef _OmrFreeDart = void Function(Pointer<Void>);
typedef _OmrRecognizeGrayC = Pointer<Utf8> Function(
    Pointer<Void>, Pointer<Float>, Int32, Int32, Pointer<Int32>);
typedef _OmrRecognizeGrayDart = Pointer<Utf8> Function(
    Pointer<Void>, Pointer<Float>, int, int, Pointer<Int32>);
typedef _OmrRecognizeRawC = Pointer<Utf8> Function(
    Pointer<Void>, Pointer<Uint8>, Int32, Int32, Int32, Pointer<Int32>);
typedef _OmrRecognizeRawDart = Pointer<Utf8> Function(
    Pointer<Void>, Pointer<Uint8>, int, int, int, Pointer<Int32>);

DynamicLibrary _openOmrLib([String? libPath]) {
  if (libPath != null) return DynamicLibrary.open(libPath);
  if (Platform.isIOS) return DynamicLibrary.process();
  if (Platform.isAndroid || Platform.isLinux) return DynamicLibrary.open('libcrispembed.so');
  if (Platform.isMacOS) return DynamicLibrary.open('libcrispembed.dylib');
  if (Platform.isWindows) return DynamicLibrary.open('crispembed.dll');
  return DynamicLibrary.open('libcrispembed.so');
}

/// On-device Optical Music Recognition via CrispEmbed's ggml inference.
///
/// The engine (SMT, TrOMR, …) is auto-detected from the GGUF architecture,
/// so one class covers every OMR model. Output is the model's raw token
/// string (parse downstream into MusicXML/MIDI/LilyPond/etc.).
class CrispEmbedOmr {
  late final DynamicLibrary _lib;
  late final Pointer<Void> _ctx;
  bool _disposed = false;

  late final _OmrFreeDart _free;
  late final _OmrRecognizeGrayDart _recognizeGray;
  late final _OmrRecognizeRawDart _recognizeRaw;

  /// Load an OMR GGUF model (e.g. `smt-grandstaff-q8_0.gguf`).
  ///
  /// [modelPath] — path to the `.gguf` file (arch auto-detected).
  /// [nThreads] — CPU thread count (default 4).
  /// [libPath] — optional explicit path to the shared library.
  CrispEmbedOmr(String modelPath, {int nThreads = 4, String? libPath}) {
    _lib = _openOmrLib(libPath);
    final init = _lib.lookupFunction<_OmrInitC, _OmrInitDart>('crispembed_ocr_model_init');
    _free = _lib.lookupFunction<_OmrFreeC, _OmrFreeDart>('crispembed_ocr_model_free');
    _recognizeGray = _lib.lookupFunction<_OmrRecognizeGrayC, _OmrRecognizeGrayDart>(
        'crispembed_ocr_model_recognize_gray');
    _recognizeRaw = _lib.lookupFunction<_OmrRecognizeRawC, _OmrRecognizeRawDart>(
        'crispembed_ocr_model_recognize');

    final pathPtr = modelPath.toNativeUtf8();
    _ctx = init(pathPtr, nThreads);
    calloc.free(pathPtr);
    if (_ctx == nullptr) {
      throw Exception('Failed to load OMR model: $modelPath');
    }
  }

  /// Recognize music from raw RGB/RGBA/grayscale pixel bytes.
  /// [bytes] — row-major pixels, [channels] = 1/3/4. Preprocessing (resize,
  /// grayscale, per-model normalization) is applied internally.
  String? recognize(Uint8List bytes, int width, int height, int channels) {
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

  /// Recognize music from a grayscale float image ([0..1], row-major).
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

  void dispose() {
    if (!_disposed) {
      _free(_ctx);
      _disposed = true;
    }
  }
}
