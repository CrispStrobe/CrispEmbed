import 'dart:convert';
import 'dart:ffi';
import 'dart:io';
import 'package:ffi/ffi.dart';

// Independent lazy bindings: importing the package with an older native library
// does not look up new exports unless this new capability is constructed.
typedef _InitNative = Pointer<Void> Function(Pointer<Utf8>, Int32);
typedef _Init = Pointer<Void> Function(Pointer<Utf8>, int);
typedef _FreeNative = Void Function(Pointer<Void>);
typedef _Free = void Function(Pointer<Void>);
typedef _RunNative = Pointer<Utf8> Function(Pointer<Void>, Pointer<Utf8>);
typedef _Run = Pointer<Utf8> Function(Pointer<Void>, Pointer<Utf8>);
typedef _TextNative = Pointer<Utf8> Function(Pointer<Void>);
typedef _Text = Pointer<Utf8> Function(Pointer<Void>);
typedef _ResolveNative = Pointer<Utf8> Function(Pointer<Utf8>, Int32);
typedef _Resolve = Pointer<Utf8> Function(Pointer<Utf8>, int);
typedef _OfflineNative = Void Function(Int32);
typedef _Offline = void Function(int);

// Dart strings can contain unpaired UTF-16 surrogates. Reject them before FFI
// encoding rather than silently changing the raw request to replacement chars.
void _requireScalarString(String value) {
  final units = value.codeUnits;
  for (var i = 0; i < units.length; i++) {
    final u = units[i];
    if (u >= 0xd800 && u <= 0xdbff) {
      if (++i >= units.length || units[i] < 0xdc00 || units[i] > 0xdfff) {
        throw ArgumentError('SystemOne requires Unicode scalar strings');
      }
    } else if (u >= 0xdc00 && u <= 0xdfff) {
      throw ArgumentError('SystemOne requires Unicode scalar strings');
    }
  }
}

/// Native CPU/F32 SystemOne joint-schema decisions, full model/answers/usage.
class CrispSystemOne {
  late final DynamicLibrary _lib;
  late final _Free _free;
  late final _Run _run;
  late final _Text _error;
  late final _Text _profile;
  Pointer<Void> _ctx = nullptr;

  CrispSystemOne(String modelPath, {int nThreads = 2, String? libPath,
      bool? autoDownload, bool? offline}) {
    _requireScalarString(modelPath);
    if (modelPath.contains('\u0000')) throw ArgumentError('SystemOne model path contains NUL');
    if (nThreads < 0 || nThreads > 64) throw ArgumentError('SystemOne threads must be0..64');
    _lib = libPath != null ? DynamicLibrary.open(libPath) : Platform.isIOS
        ? DynamicLibrary.process() : DynamicLibrary.open(Platform.isWindows
            ? 'crispembed.dll' : Platform.isMacOS ? 'libcrispembed.dylib' : 'libcrispembed.so');
    for (final name in ['crispembed_systemone_init', 'crispembed_systemone_free',
      'crispembed_systemone_run_json', 'crispembed_systemone_error', 'crispembed_systemone_profile']) {
      if (!_lib.providesSymbol(name)) throw StateError('Native library does not provide SystemOne support');
    }
    final init = _lib.lookupFunction<_InitNative, _Init>('crispembed_systemone_init');
    _free = _lib.lookupFunction<_FreeNative, _Free>('crispembed_systemone_free');
    _run = _lib.lookupFunction<_RunNative, _Run>('crispembed_systemone_run_json');
    _error = _lib.lookupFunction<_TextNative, _Text>('crispembed_systemone_error');
    _profile = _lib.lookupFunction<_TextNative, _Text>('crispembed_systemone_profile');
    if (offline != null) _lib.lookupFunction<_OfflineNative, _Offline>('crispembed_set_offline')(offline ? 1 : 0);
    final resolver = _lib.lookupFunction<_ResolveNative, _Resolve>('crispembed_resolve_model');
    final arg = modelPath.toNativeUtf8();
    late String resolved;
    try {
      final value = resolver(arg, (autoDownload ?? (!modelPath.contains('.gguf') &&
          !modelPath.contains('/') && !modelPath.contains('\\'))) ? 1 : 0);
      if (value == nullptr) throw StateError('Could not resolve SystemOne model');
      resolved = value.toDartString();
      if (resolved.isEmpty) throw StateError('Could not resolve SystemOne model');
    } finally { calloc.free(arg); }
    final path = resolved.toNativeUtf8();
    try { _ctx = init(path, nThreads); } finally { calloc.free(path); }
    if (_ctx == nullptr) throw StateError(_message());
  }
  String _message() {
    final value = _error(_ctx);
    return value == nullptr ? 'SystemOne operation failed' : value.toDartString();
  }
  void _requireOpen() { if (_ctx == nullptr) throw StateError('SystemOne context is closed'); }
  String get profile { _requireOpen(); return _profile(_ctx).toDartString(); }
  String runJson(String request) {
    _requireOpen();
    _requireScalarString(request);
    if (request.contains('\u0000') || utf8.encode(request).length > 65536) {
      throw ArgumentError('SystemOne requires NUL-free JSON within64KiB');
    }
    final input = request.toNativeUtf8();
    try {
      final output = _run(_ctx, input);
      if (output == nullptr) throw StateError(_message());
      return output.toDartString();
    } finally { calloc.free(input); }
  }
  Map<String, dynamic> run(Map<String, dynamic> request) =>
      jsonDecode(runJson(jsonEncode(request))) as Map<String, dynamic>;
  void dispose() { if (_ctx != nullptr) { _free(_ctx); _ctx = nullptr; } }
}
