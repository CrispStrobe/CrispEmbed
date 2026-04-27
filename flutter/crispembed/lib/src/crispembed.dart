import 'dart:ffi';
import 'dart:io';
import 'dart:math';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import 'crispembed_bindings.dart';

/// Result from bi-encoder reranking.
class RerankResult {
  final int index;
  final double score;
  final String? document;

  RerankResult({required this.index, required this.score, this.document});

  @override
  String toString() => 'RerankResult(index=$index, score=${score.toStringAsFixed(4)})';
}

class ModelInfo {
  final String name;
  final String desc;
  final String filename;
  final String size;

  const ModelInfo({
    required this.name,
    required this.desc,
    required this.filename,
    required this.size,
  });
}

/// Text embedding model using ggml inference.
///
/// Supports dense embeddings, sparse retrieval (BGE-M3/SPLADE), ColBERT
/// multi-vector, cross-encoder reranking, and bi-encoder reranking.
///
/// ```dart
/// final model = CrispEmbed('all-MiniLM-L6-v2.gguf');
/// final vec = model.encode('Hello world');  // Float32List(384)
///
/// // Batch
/// final vecs = model.encodeBatch(['Hello', 'World']);  // List<Float32List>
///
/// // Bi-encoder reranking
/// final ranked = model.rerankBiencoder('query', ['doc1', 'doc2']);
///
/// model.dispose();
/// ```
class CrispEmbed {
  late final DynamicLibrary _lib;
  late final Pointer<CrispembedContext> _ctx;
  bool _disposed = false;

  // Cached function lookups
  late final CrispembedEncode _encode;
  late final CrispembedEncodeBatch _encodeBatch;
  late final CrispembedFree _free;
  late final CrispembedSetDim _setDimFn;
  late final CrispembedSetPrefix _setPrefixFn;
  late final CrispembedGetPrefix _getPrefixFn;
  late final CrispembedHasSparse _hasSparseCheck;
  late final CrispembedHasColbert _hasColbertCheck;
  late final CrispembedIsReranker _isRerankerCheck;
  late final CrispembedEncodeSparse _encodeSparse;
  late final CrispembedEncodeMultivec _encodeMultivec;
  late final CrispembedRerank _rerankFn;
  // Audio (BidirLM-Omni etc.) — optional, missing on builds without crisp_audio.
  CrispembedHasAudio? _hasAudioCheck;
  CrispembedEncodeAudio? _encodeAudioFn;
  // Vision (BidirLM-Omni etc.) — always present, but the GGUF may lack vision tensors.
  CrispembedHasVision? _hasVisionCheck;
  CrispembedEncodeImage? _encodeImageFn;
  CrispembedEncodeImageRaw? _encodeImageRawFn;

  /// Load a GGUF model file.
  ///
  /// [modelPath] — path to the `.gguf` file.
  /// [nThreads] — CPU thread count (0 = auto-detect).
  /// [libPath] — optional path to the shared library. If omitted, searches
  ///   standard platform locations.
  CrispEmbed(String modelPath, {int nThreads = 0, String? libPath, bool? autoDownload}) {
    _lib = DynamicLibrary.open(libPath ?? _findLib());
    _bindFunctions();

    final resolved = resolveModel(modelPath, libPath: libPath, autoDownload: autoDownload);
    final pathPtr = resolved.toNativeUtf8();
    _ctx = _lib
        .lookupFunction<CrispembedInitNative, CrispembedInit>('crispembed_init')
        .call(pathPtr, nThreads);
    calloc.free(pathPtr);

    if (_ctx == nullptr) {
      throw Exception('Failed to load model: $modelPath');
    }
  }

  void _bindFunctions() {
    _encode = _lib.lookupFunction<CrispembedEncodeNative, CrispembedEncode>(
        'crispembed_encode');
    _encodeBatch = _lib
        .lookupFunction<CrispembedEncodeBatchNative, CrispembedEncodeBatch>(
            'crispembed_encode_batch');
    _free = _lib.lookupFunction<CrispembedFreeNative, CrispembedFree>(
        'crispembed_free');
    _setDimFn = _lib.lookupFunction<CrispembedSetDimNative, CrispembedSetDim>(
        'crispembed_set_dim');
    _setPrefixFn = _lib
        .lookupFunction<CrispembedSetPrefixNative, CrispembedSetPrefix>(
            'crispembed_set_prefix');
    _getPrefixFn = _lib
        .lookupFunction<CrispembedGetPrefixNative, CrispembedGetPrefix>(
            'crispembed_get_prefix');
    _hasSparseCheck = _lib
        .lookupFunction<CrispembedHasSparseNative, CrispembedHasSparse>(
            'crispembed_has_sparse');
    _hasColbertCheck = _lib
        .lookupFunction<CrispembedHasColbertNative, CrispembedHasColbert>(
            'crispembed_has_colbert');
    _isRerankerCheck = _lib
        .lookupFunction<CrispembedIsRerankerNative, CrispembedIsReranker>(
            'crispembed_is_reranker');
    _encodeSparse = _lib
        .lookupFunction<CrispembedEncodeSparseNative, CrispembedEncodeSparse>(
            'crispembed_encode_sparse');
    _encodeMultivec = _lib.lookupFunction<CrispembedEncodeMultivecNative,
        CrispembedEncodeMultivec>('crispembed_encode_multivec');
    _rerankFn = _lib.lookupFunction<CrispembedRerankNative, CrispembedRerank>(
        'crispembed_rerank');
    // Audio symbols are absent in builds without crisp_audio — bind lazily.
    try {
      _hasAudioCheck = _lib.lookupFunction<CrispembedHasAudioNative, CrispembedHasAudio>(
          'crispembed_has_audio');
      _encodeAudioFn = _lib.lookupFunction<CrispembedEncodeAudioNative, CrispembedEncodeAudio>(
          'crispembed_encode_audio');
    } catch (_) {
      _hasAudioCheck = null;
      _encodeAudioFn = null;
    }
    try {
      _hasVisionCheck = _lib.lookupFunction<CrispembedHasVisionNative, CrispembedHasVision>(
          'crispembed_has_vision');
      _encodeImageFn = _lib.lookupFunction<CrispembedEncodeImageNative, CrispembedEncodeImage>(
          'crispembed_encode_image');
      _encodeImageRawFn = _lib.lookupFunction<CrispembedEncodeImageRawNative, CrispembedEncodeImageRaw>(
          'crispembed_encode_image_raw');
    } catch (_) {
      _hasVisionCheck = null;
      _encodeImageFn = null;
      _encodeImageRawFn = null;
    }
  }

  // ------------------------------------------------------------------
  // Audio encoding (BidirLM-Omni and similar)
  // ------------------------------------------------------------------

  /// Whether this build of CrispEmbed has audio support compiled in.
  bool get hasAudio => _hasAudioCheck != null && _hasAudioCheck!(_ctx) != 0;

  /// Encode raw 16 kHz mono float32 PCM into the model's shared embedding
  /// space. Returns an empty list if the build lacks audio support or the
  /// model has no audio tower.
  Float32List encodeAudio(Float32List pcm) {
    _checkDisposed();
    if (_encodeAudioFn == null) return Float32List(0);
    final pcmPtr = calloc<Float>(pcm.length);
    pcmPtr.asTypedList(pcm.length).setAll(0, pcm);
    final dimPtr = calloc<Int32>();
    try {
      final ptr = _encodeAudioFn!(_ctx, pcmPtr, pcm.length, dimPtr);
      if (ptr == nullptr || dimPtr.value <= 0) return Float32List(0);
      return Float32List.fromList(ptr.asTypedList(dimPtr.value));
    } finally {
      calloc.free(pcmPtr);
      calloc.free(dimPtr);
    }
  }

  // ------------------------------------------------------------------
  // Image encoding (BidirLM-Omni vision tower)
  // ------------------------------------------------------------------

  /// Whether this build has vision support compiled in. The loaded GGUF
  /// may still lack vision tensors — `encodeImage` returns empty in that
  /// case.
  bool get hasVision => _hasVisionCheck != null && _hasVisionCheck!(_ctx) != 0;

  /// Encode pre-flattened pixel patches into the model's shared embedding
  /// space (mean-pooled, L2-normalized).
  ///
  /// [pixelPatches] is `(n_patches, 1536)` row-major float32, produced by
  /// the Python preprocessor (or a future Dart port of `Qwen2VLImageProcessor`).
  /// [gridThw] is `(n_images, 3)` int32 with `(t, h, w)` per image.
  Float32List encodeImage(Float32List pixelPatches, Int32List gridThw) {
    _checkDisposed();
    if (_encodeImageFn == null) return Float32List(0);
    if (gridThw.length % 3 != 0) return Float32List(0);
    final nImages = gridThw.length ~/ 3;
    var nPatches = 0;
    for (var i = 0; i < nImages; i++) {
      nPatches += gridThw[i * 3] * gridThw[i * 3 + 1] * gridThw[i * 3 + 2];
    }
    if (nPatches <= 0) return Float32List(0);

    final pxPtr = calloc<Float>(pixelPatches.length);
    pxPtr.asTypedList(pixelPatches.length).setAll(0, pixelPatches);
    final gridPtr = calloc<Int32>(gridThw.length);
    gridPtr.asTypedList(gridThw.length).setAll(0, gridThw);
    final dimPtr = calloc<Int32>();
    try {
      final ptr = _encodeImageFn!(_ctx, pxPtr, nPatches, gridPtr, nImages, dimPtr);
      if (ptr == nullptr || dimPtr.value <= 0) return Float32List(0);
      return Float32List.fromList(ptr.asTypedList(dimPtr.value));
    } finally {
      calloc.free(pxPtr);
      calloc.free(gridPtr);
      calloc.free(dimPtr);
    }
  }

  /// Raw vision tower output (un-pooled, un-normalized). Returns
  /// `(imageEmbeds, deepstackFeatures)` — each entry is a flat row-major
  /// `(n_merged * dim)` Float32List.
  ///
  /// Used for parity testing against HF's BidirLMOmniVisionModel.
  (Float32List, List<Float32List>) encodeImageRaw(
      Float32List pixelPatches, Int32List gridThw) {
    _checkDisposed();
    if (_encodeImageRawFn == null) return (Float32List(0), []);
    if (gridThw.length % 3 != 0) return (Float32List(0), []);
    final nImages = gridThw.length ~/ 3;
    var nPatches = 0;
    for (var i = 0; i < nImages; i++) {
      nPatches += gridThw[i * 3] * gridThw[i * 3 + 1] * gridThw[i * 3 + 2];
    }
    if (nPatches <= 0) return (Float32List(0), []);

    final pxPtr = calloc<Float>(pixelPatches.length);
    pxPtr.asTypedList(pixelPatches.length).setAll(0, pixelPatches);
    final gridPtr = calloc<Int32>(gridThw.length);
    gridPtr.asTypedList(gridThw.length).setAll(0, gridThw);
    final nMergedPtr = calloc<Int32>();
    final dimPtr = calloc<Int32>();
    final nDsPtr = calloc<Int32>();
    try {
      final ptr = _encodeImageRawFn!(
          _ctx, pxPtr, nPatches, gridPtr, nImages, nMergedPtr, dimPtr, nDsPtr);
      if (ptr == nullptr || nMergedPtr.value <= 0) {
        return (Float32List(0), []);
      }
      final perSlab = nMergedPtr.value * dimPtr.value;
      final flat = ptr.asTypedList((1 + nDsPtr.value) * perSlab);
      final img = Float32List.fromList(flat.sublist(0, perSlab));
      final ds = <Float32List>[];
      for (var k = 0; k < nDsPtr.value; k++) {
        final beg = (1 + k) * perSlab;
        ds.add(Float32List.fromList(flat.sublist(beg, beg + perSlab)));
      }
      return (img, ds);
    } finally {
      calloc.free(pxPtr);
      calloc.free(gridPtr);
      calloc.free(nMergedPtr);
      calloc.free(dimPtr);
      calloc.free(nDsPtr);
    }
  }

  // ------------------------------------------------------------------
  // Dense encoding
  // ------------------------------------------------------------------

  /// Encode a single text to an L2-normalized embedding vector.
  Float32List encode(String text) {
    _checkDisposed();
    final textPtr = text.toNativeUtf8();
    final dimPtr = calloc<Int32>();
    try {
      final ptr = _encode(_ctx, textPtr, dimPtr);
      if (ptr == nullptr) {
        throw Exception('Encoding failed for: ${text.substring(0, min(50, text.length))}');
      }
      final dim = dimPtr.value;
      return Float32List.fromList(ptr.asTypedList(dim));
    } finally {
      calloc.free(textPtr);
      calloc.free(dimPtr);
    }
  }

  /// Encode multiple texts in a single GPU graph pass.
  List<Float32List> encodeBatch(List<String> texts) {
    _checkDisposed();
    if (texts.isEmpty) return [];

    final n = texts.length;
    final textPtrs = calloc<Pointer<Utf8>>(n);
    final nativeTexts = <Pointer<Utf8>>[];
    try {
      for (var i = 0; i < n; i++) {
        final p = texts[i].toNativeUtf8();
        nativeTexts.add(p);
        textPtrs[i] = p;
      }

      final dimPtr = calloc<Int32>();
      try {
        final ptr = _encodeBatch(_ctx, textPtrs, n, dimPtr);
        if (ptr == nullptr) throw Exception('Batch encoding failed');
        final dim = dimPtr.value;
        final flat = ptr.asTypedList(n * dim);
        return List.generate(
            n, (i) => Float32List.fromList(flat.sublist(i * dim, (i + 1) * dim)));
      } finally {
        calloc.free(dimPtr);
      }
    } finally {
      for (final p in nativeTexts) {
        calloc.free(p);
      }
      calloc.free(textPtrs);
    }
  }

  // ------------------------------------------------------------------
  // Sparse retrieval (BGE-M3 / SPLADE)
  // ------------------------------------------------------------------

  /// Encode text to sparse term-weight vector.
  ///
  /// Returns a map of vocab token IDs to positive weights.
  Map<int, double> encodeSparse(String text) {
    _checkDisposed();
    final textPtr = text.toNativeUtf8();
    final indicesPtr = calloc<Pointer<Int32>>();
    final valuesPtr = calloc<Pointer<Float>>();
    try {
      final n = _encodeSparse(_ctx, textPtr, indicesPtr, valuesPtr);
      if (n <= 0) return {};
      final indices = indicesPtr.value.asTypedList(n);
      final values = valuesPtr.value.asTypedList(n);
      return {for (var i = 0; i < n; i++) indices[i]: values[i].toDouble()};
    } finally {
      calloc.free(textPtr);
      calloc.free(indicesPtr);
      calloc.free(valuesPtr);
    }
  }

  // ------------------------------------------------------------------
  // ColBERT multi-vector
  // ------------------------------------------------------------------

  /// Encode text to per-token L2-normalized embeddings (ColBERT).
  ///
  /// Returns a list of token embeddings, each of length [colbertDim].
  List<Float32List> encodeMultivec(String text) {
    _checkDisposed();
    final textPtr = text.toNativeUtf8();
    final nTokensPtr = calloc<Int32>();
    final dimPtr = calloc<Int32>();
    try {
      final ptr = _encodeMultivec(_ctx, textPtr, nTokensPtr, dimPtr);
      if (ptr == nullptr || nTokensPtr.value <= 0) return [];
      final nTokens = nTokensPtr.value;
      final dim = dimPtr.value;
      final flat = ptr.asTypedList(nTokens * dim);
      return List.generate(
          nTokens, (i) => Float32List.fromList(flat.sublist(i * dim, (i + 1) * dim)));
    } finally {
      calloc.free(textPtr);
      calloc.free(nTokensPtr);
      calloc.free(dimPtr);
    }
  }

  // ------------------------------------------------------------------
  // Cross-encoder reranking
  // ------------------------------------------------------------------

  /// Score a (query, document) pair with a cross-encoder.
  ///
  /// Returns raw logit (higher = more relevant).
  double rerank(String query, String document) {
    _checkDisposed();
    if (!isReranker) throw Exception('Model is not a reranker');
    final qPtr = query.toNativeUtf8();
    final dPtr = document.toNativeUtf8();
    try {
      return _rerankFn(_ctx, qPtr, dPtr);
    } finally {
      calloc.free(qPtr);
      calloc.free(dPtr);
    }
  }

  // ------------------------------------------------------------------
  // Bi-encoder reranking
  // ------------------------------------------------------------------

  /// Rank documents by cosine similarity to query embedding.
  ///
  /// Encodes query + all documents in one batch, computes dot products
  /// of L2-normalized embeddings (= cosine similarity).
  List<RerankResult> rerankBiencoder(
    String query,
    List<String> documents, {
    int? topN,
    bool returnDocuments = true,
  }) {
    final allTexts = [query, ...documents];
    final embeddings = encodeBatch(allTexts);
    if (embeddings.isEmpty) return [];

    final queryVec = embeddings[0];
    final results = <RerankResult>[];
    for (var i = 1; i < embeddings.length; i++) {
      var dot = 0.0;
      for (var j = 0; j < queryVec.length; j++) {
        dot += queryVec[j] * embeddings[i][j];
      }
      results.add(RerankResult(
        index: i - 1,
        score: dot,
        document: returnDocuments ? documents[i - 1] : null,
      ));
    }
    results.sort((a, b) => b.score.compareTo(a.score));
    if (topN != null && topN < results.length) {
      return results.sublist(0, topN);
    }
    return results;
  }

  // ------------------------------------------------------------------
  // Configuration
  // ------------------------------------------------------------------

  /// Set Matryoshka output dimension. Pass 0 to restore model default.
  void setDim(int dim) {
    _checkDisposed();
    _setDimFn(_ctx, dim);
  }

  /// Set a text prefix prepended to all inputs before tokenization.
  ///
  /// Typical values: `"query: "`, `"search_query: "`, `"Represent this sentence: "`.
  /// Pass empty string to clear.
  void setPrefix(String prefix) {
    _checkDisposed();
    final p = prefix.toNativeUtf8();
    try {
      _setPrefixFn(_ctx, p);
    } finally {
      calloc.free(p);
    }
  }

  /// Get the current prefix (empty string if none).
  String get prefix {
    _checkDisposed();
    final p = _getPrefixFn(_ctx);
    return p == nullptr ? '' : p.toDartString();
  }

  /// True if the model has a sparse retrieval head (BGE-M3/SPLADE).
  bool get hasSparse => _hasSparseCheck(_ctx) != 0;

  /// True if the model has a ColBERT multi-vector head.
  bool get hasColbert => _hasColbertCheck(_ctx) != 0;

  /// True if the model is a cross-encoder reranker.
  bool get isReranker => _isRerankerCheck(_ctx) != 0;

  // ------------------------------------------------------------------
  // Lifecycle
  // ------------------------------------------------------------------

  /// Release all resources. Must be called when done.
  void dispose() {
    if (!_disposed) {
      _free(_ctx);
      _disposed = true;
    }
  }

  void _checkDisposed() {
    if (_disposed) throw StateError('CrispEmbed has been disposed');
  }

  static String _findLib() {
    if (Platform.isAndroid || Platform.isLinux) return 'libcrispembed.so';
    if (Platform.isIOS || Platform.isMacOS) return 'crispembed.framework/crispembed';
    if (Platform.isWindows) return 'crispembed.dll';
    return 'libcrispembed.so';
  }

  static String cacheDir({String? libPath}) {
    final lib = DynamicLibrary.open(libPath ?? _findLib());
    final fn = lib.lookupFunction<CrispembedCacheDirNative, CrispembedCacheDir>(
        'crispembed_cache_dir');
    final p = fn();
    return p == nullptr ? '' : p.toDartString();
  }

  static String resolveModel(String modelPath, {String? libPath, bool? autoDownload}) {
    final lib = DynamicLibrary.open(libPath ?? _findLib());
    final fn = lib.lookupFunction<CrispembedResolveModelNative,
        CrispembedResolveModel>('crispembed_resolve_model');
    final shouldDownload = autoDownload ??
        (!modelPath.contains('.gguf') &&
            !modelPath.contains('/') &&
            !modelPath.contains('\\'));
    final arg = modelPath.toNativeUtf8();
    try {
      final out = fn(arg, shouldDownload ? 1 : 0);
      final resolved = out == nullptr ? '' : out.toDartString();
      if (resolved.isEmpty) {
        throw Exception('Could not resolve model: $modelPath');
      }
      return resolved;
    } finally {
      calloc.free(arg);
    }
  }

  static List<ModelInfo> listModels({String? libPath}) {
    final lib = DynamicLibrary.open(libPath ?? _findLib());
    final nModels = lib.lookupFunction<CrispembedNModelsNative, CrispembedNModels>(
        'crispembed_n_models');
    final modelName = lib.lookupFunction<CrispembedModelStringNative,
        CrispembedModelString>('crispembed_model_name');
    final modelDesc = lib.lookupFunction<CrispembedModelStringNative,
        CrispembedModelString>('crispembed_model_desc');
    final modelFilename = lib.lookupFunction<CrispembedModelStringNative,
        CrispembedModelString>('crispembed_model_filename');
    final modelSize = lib.lookupFunction<CrispembedModelStringNative,
        CrispembedModelString>('crispembed_model_size');
    final models = <ModelInfo>[];
    for (var i = 0; i < nModels(); i++) {
      models.add(ModelInfo(
        name: modelName(i).toDartString(),
        desc: modelDesc(i).toDartString(),
        filename: modelFilename(i).toDartString(),
        size: modelSize(i).toDartString(),
      ));
    }
    return models;
  }
}
