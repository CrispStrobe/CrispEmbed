import 'dart:ffi';
import 'dart:io';
import 'dart:math';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import 'crispembed_bindings.dart';

/// Open the CrispEmbed native library for the current platform.
///
/// iOS uses static linking (symbols in the main binary), so we return
/// [DynamicLibrary.process()]. All other platforms load a shared library.
DynamicLibrary _openNativeLib([String? libPath]) {
  if (libPath != null) return DynamicLibrary.open(libPath);
  if (Platform.isIOS) return DynamicLibrary.process();
  if (Platform.isAndroid || Platform.isLinux) return DynamicLibrary.open('libcrispembed.so');
  if (Platform.isMacOS) return DynamicLibrary.open('libcrispembed.dylib');
  if (Platform.isWindows) return DynamicLibrary.open('crispembed.dll');
  return DynamicLibrary.open('libcrispembed.so');
}

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
  // Ctx prefix (GGUF metadata) — optional, missing on older builds.
  CrispembedCtxQueryPrefixDart? _ctxQueryPrefixFn;
  CrispembedCtxPassagePrefixDart? _ctxPassagePrefixFn;
  // Audio (BidirLM-Omni etc.) — optional, missing on builds without crisp_audio.
  CrispembedHasAudio? _hasAudioCheck;
  CrispembedEncodeAudio? _encodeAudioFn;
  // Vision (BidirLM-Omni etc.) — always present, but the GGUF may lack vision tensors.
  CrispembedHasVision? _hasVisionCheck;
  CrispembedEncodeImage? _encodeImageFn;
  CrispembedEncodeImageRaw? _encodeImageRawFn;
  // File-based vision helpers — optional, absent on older builds.
  CrispembedEncodeImageFileDart? _encodeImageFileFn;
  CrispembedEncodeTextWithImageFileDart? _encodeTextWithImageFileFn;

  /// Load a GGUF model file.
  ///
  /// [modelPath] — path to the `.gguf` file.
  /// [nThreads] — CPU thread count (0 = auto-detect).
  /// [libPath] — optional path to the shared library. If omitted, searches
  ///   standard platform locations.
  CrispEmbed(String modelPath, {int nThreads = 0, String? libPath, bool? autoDownload}) {
    _lib = _openNativeLib(libPath);
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
    // Ctx prefix symbols — optional, missing on older builds.
    try {
      _ctxQueryPrefixFn = _lib.lookupFunction<CrispembedCtxQueryPrefixNative,
          CrispembedCtxQueryPrefixDart>('crispembed_ctx_query_prefix');
      _ctxPassagePrefixFn = _lib.lookupFunction<CrispembedCtxPassagePrefixNative,
          CrispembedCtxPassagePrefixDart>('crispembed_ctx_passage_prefix');
    } catch (_) {
      _ctxQueryPrefixFn = null;
      _ctxPassagePrefixFn = null;
    }
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
    try {
      _encodeImageFileFn = _lib.lookupFunction<CrispembedEncodeImageFileNative,
          CrispembedEncodeImageFileDart>('crispembed_encode_image_file');
      _encodeTextWithImageFileFn = _lib.lookupFunction<
          CrispembedEncodeTextWithImageFileNative,
          CrispembedEncodeTextWithImageFileDart>('crispembed_encode_text_with_image_file');
    } catch (_) {
      _encodeImageFileFn = null;
      _encodeTextWithImageFileFn = null;
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
  // File-based vision encoding
  // ------------------------------------------------------------------

  /// Load an image from disk and encode it into the model's shared embedding
  /// space (mean-pooled, L2-normalized), without requiring a Python preprocessor.
  ///
  /// Supports JPEG, PNG, BMP, and other formats readable by stb_image.
  /// Returns an empty list if the build lacks vision support, the model has
  /// no vision tower, or the file cannot be read.
  ///
  /// Note: the native preprocessor uses bilinear resize; expect cosine ≈ 0.95–0.98
  /// compared to the HF reference path (bicubic + antialias).
  Float32List encodeImageFile(String path) {
    _checkDisposed();
    if (_encodeImageFileFn == null) return Float32List(0);
    final pathPtr = path.toNativeUtf8();
    final dimPtr = calloc<Int32>();
    try {
      final ptr = _encodeImageFileFn!(_ctx, pathPtr, dimPtr);
      if (ptr == nullptr || dimPtr.value <= 0) return Float32List(0);
      return Float32List.fromList(ptr.asTypedList(dimPtr.value));
    } finally {
      calloc.free(pathPtr);
      calloc.free(dimPtr);
    }
  }

  /// Load an image from disk and produce a text-conditioned multimodal embedding.
  ///
  /// [text] must contain the appropriate number of image-token placeholders for
  /// the grid that the native smart_resize will produce. See the Python helpers
  /// in `python/crispembed/image.py` for how to build the text template.
  ///
  /// Returns an empty list on failure (no vision tower, bad file, etc.).
  Float32List encodeTextWithImageFile(String text, String imagePath) {
    _checkDisposed();
    if (_encodeTextWithImageFileFn == null) return Float32List(0);
    final textPtr = text.toNativeUtf8();
    final pathPtr = imagePath.toNativeUtf8();
    final dimPtr = calloc<Int32>();
    try {
      final ptr = _encodeTextWithImageFileFn!(_ctx, textPtr, pathPtr, dimPtr);
      if (ptr == nullptr || dimPtr.value <= 0) return Float32List(0);
      return Float32List.fromList(ptr.asTypedList(dimPtr.value));
    } finally {
      calloc.free(textPtr);
      calloc.free(pathPtr);
      calloc.free(dimPtr);
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
  // ColBERT MaxSim scoring
  // ------------------------------------------------------------------

  late final CrispembedColbertScore _colbertScore = _lib
      .lookupFunction<CrispembedColbertScoreNative, CrispembedColbertScore>(
          'crispembed_colbert_score');

  /// Compute ColBERT MaxSim score between query and document token vectors.
  double colbertScore(Float32List queryVecs, int nQuery,
      Float32List docVecs, int nDoc, int dim) {
    final qPtr = calloc<Float>(queryVecs.length);
    final dPtr = calloc<Float>(docVecs.length);
    qPtr.asTypedList(queryVecs.length).setAll(0, queryVecs);
    dPtr.asTypedList(docVecs.length).setAll(0, docVecs);
    try {
      return _colbertScore(qPtr, nQuery, dPtr, nDoc, dim);
    } finally {
      calloc.free(qPtr);
      calloc.free(dPtr);
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

  /// Query prefix from GGUF metadata (`colbert.query_prefix`), or empty string.
  String get ctxQueryPrefix {
    _checkDisposed();
    if (_ctxQueryPrefixFn == null) return '';
    final p = _ctxQueryPrefixFn!(_ctx);
    return p == nullptr ? '' : p.toDartString();
  }

  /// Passage/document prefix from GGUF metadata, or empty string.
  String get ctxPassagePrefix {
    _checkDisposed();
    if (_ctxPassagePrefixFn == null) return '';
    final p = _ctxPassagePrefixFn!(_ctx);
    return p == nullptr ? '' : p.toDartString();
  }

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

  static String cacheDir({String? libPath}) {
    final lib = _openNativeLib(libPath);
    final fn = lib.lookupFunction<CrispembedCacheDirNative, CrispembedCacheDir>(
        'crispembed_cache_dir');
    final p = fn();
    return p == nullptr ? '' : p.toDartString();
  }

  static String resolveModel(String modelPath, {String? libPath, bool? autoDownload}) {
    final lib = _openNativeLib(libPath);
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
    final lib = _openNativeLib(libPath);
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

// ---------------------------------------------------------------------------
// Face detection & recognition
// ---------------------------------------------------------------------------

/// A single detected face returned by [CrispFace.detect] or
/// [CrispFacePipeline.run].
///
/// [bbox] is `[x, y, w, h]` in pixel coordinates.
/// [landmarks] is a flat `[x0, y0, x1, y1, …]` array of keypoint coordinates
/// (5 points × 2 floats = 10 values for SCRFD/RetinaFace).
/// [score] is the detector confidence in [0, 1].
/// [embedding] is set by [CrispFacePipeline.run]; null from [CrispFace.detect].
class FaceResult {
  final List<double> bbox;
  final List<double> landmarks;
  final double score;
  final Float32List? embedding;

  const FaceResult({
    required this.bbox,
    required this.landmarks,
    required this.score,
    this.embedding,
  });

  @override
  String toString() {
    final b = bbox.map((v) => v.toStringAsFixed(1)).join(', ');
    final s = score.toStringAsFixed(3);
    final dim = embedding?.length ?? 0;
    return 'FaceResult(bbox=[$b], score=$s, embDim=$dim)';
  }
}

/// Wraps a single face model (detector *or* recognizer) loaded via
/// `crispembed_face_init`.
///
/// ```dart
/// final det = CrispFace('scrfd_10g.onnx');
/// final faces = det.detect('photo.jpg', conf: 0.6);
/// print(det.modelType);   // 'scrfd' / 'retinaface' / 'sface' / …
/// det.dispose();
/// ```
class CrispFace {
  late final DynamicLibrary _lib;
  late final Pointer<Void> _ctx;
  bool _disposed = false;

  // Cached function lookups
  late final CrispembedFaceDimDart _dimFn;
  late final CrispembedFaceTypeDart _typeFn;
  late final CrispembedDetectFacesDart _detectFn;
  late final CrispembedEncodeFaceDart _encodeFn;
  late final CrispembedFaceFreeDart _freeFn;

  /// Load a face model (ONNX or GGUF path).
  ///
  /// [modelPath] — path to the model file or a short alias (e.g. `'scrfd_10g'`).
  /// [nThreads] — CPU thread count (0 = auto).
  /// [libPath] — optional path to `libcrispembed.so` / `crispembed.dll`.
  CrispFace(String modelPath, {int nThreads = 0, String? libPath}) {
    _lib = _openNativeLib(libPath);
    _bindFunctions();

    final pathPtr = modelPath.toNativeUtf8();
    _ctx = _lib
        .lookupFunction<CrispembedFaceInitNative, CrispembedFaceInitDart>(
            'crispembed_face_init')
        .call(pathPtr, nThreads);
    calloc.free(pathPtr);

    if (_ctx == nullptr) {
      throw Exception('Failed to load face model: $modelPath');
    }
  }

  void _bindFunctions() {
    _dimFn = _lib.lookupFunction<CrispembedFaceDimNative, CrispembedFaceDimDart>(
        'crispembed_face_dim');
    _typeFn = _lib
        .lookupFunction<CrispembedFaceTypeNative, CrispembedFaceTypeDart>(
            'crispembed_face_type');
    _detectFn = _lib
        .lookupFunction<CrispembedDetectFacesNative, CrispembedDetectFacesDart>(
            'crispembed_detect_faces');
    _encodeFn = _lib
        .lookupFunction<CrispembedEncodeFaceNative, CrispembedEncodeFaceDart>(
            'crispembed_encode_face');
    _freeFn = _lib
        .lookupFunction<CrispembedFaceFreeNative, CrispembedFaceFreeDart>(
            'crispembed_face_free');
  }

  // ------------------------------------------------------------------
  // Queries
  // ------------------------------------------------------------------

  /// Embedding dimension reported by this model (0 for detection-only models).
  int get dim {
    _checkDisposed();
    return _dimFn(_ctx);
  }

  /// Short model-type string, e.g. `'scrfd'`, `'sface'`, `'arcface'`.
  String get modelType {
    _checkDisposed();
    final p = _typeFn(_ctx);
    return p == nullptr ? '' : p.toDartString();
  }

  // ------------------------------------------------------------------
  // Detection
  // ------------------------------------------------------------------

  /// Detect faces in an image file.
  ///
  /// [imagePath] — path to a JPEG/PNG file readable by the native library.
  /// [conf] — minimum detector confidence threshold (default 0.5).
  ///
  /// Returns a list of detected faces. Each map contains:
  /// - `'bbox'`: `List<double>` — `[x, y, w, h]`
  /// - `'landmarks'`: `List<double>` — flat keypoints `[x0, y0, …]`
  /// - `'score'`: `double` — confidence
  ///
  /// The native side returns a packed Void buffer; layout is documented in
  /// `src/crispembed_face.h`. We decode it here so callers never touch raw
  /// pointers.
  List<Map<String, dynamic>> detect(String imagePath, {double conf = 0.5, int detSize = 0}) {
    _checkDisposed();
    final pathPtr = imagePath.toNativeUtf8();
    final countPtr = calloc<Int32>();
    try {
      final buf = _detectFn(_ctx, pathPtr, conf, detSize, countPtr);
      final n = countPtr.value;
      if (buf == nullptr || n <= 0) return [];
      return _decodeFaceBuffer(buf, n);
    } finally {
      calloc.free(pathPtr);
      calloc.free(countPtr);
    }
  }

  // ------------------------------------------------------------------
  // Recognition
  // ------------------------------------------------------------------

  /// Encode a single face crop to an L2-normalized embedding.
  ///
  /// [imagePath] — path to the (already-cropped or full) image.
  /// [landmarks] — flat 10-element keypoint array `[x0, y0, …, x4, y4]`
  ///   used to align the face before encoding. Pass an empty list to skip
  ///   alignment (model-specific behaviour).
  ///
  /// Returns a [Float32List] of length [dim], or an empty list on failure.
  Float32List encodeFace(String imagePath, List<double> landmarks) {
    _checkDisposed();
    final pathPtr = imagePath.toNativeUtf8();
    final dimPtr = calloc<Int32>();
    Pointer<Float> lmPtr = nullptr;
    try {
      if (landmarks.isNotEmpty) {
        lmPtr = calloc<Float>(landmarks.length);
        lmPtr.asTypedList(landmarks.length).setAll(0, landmarks);
      }
      final out = _encodeFn(_ctx, pathPtr, lmPtr, dimPtr);
      if (out == nullptr || dimPtr.value <= 0) return Float32List(0);
      return Float32List.fromList(out.asTypedList(dimPtr.value));
    } finally {
      calloc.free(pathPtr);
      calloc.free(dimPtr);
      if (lmPtr != nullptr) calloc.free(lmPtr);
    }
  }

  // ------------------------------------------------------------------
  // Lifecycle
  // ------------------------------------------------------------------

  /// Release all native resources. Must be called when done.
  void dispose() {
    if (!_disposed) {
      _freeFn(_ctx);
      _disposed = true;
    }
  }

  void _checkDisposed() {
    if (_disposed) throw StateError('CrispFace has been disposed');
  }

}

/// Combines a detector [CrispFace] and a recognizer [CrispFace] into a
/// single end-to-end face pipeline.
///
/// ```dart
/// final pipeline = CrispFacePipeline(
///   detectorPath: 'scrfd_10g.onnx',
///   recognizerPath: 'sface_sf_lfw.onnx',
/// );
///
/// final faces = pipeline.run('photo.jpg', conf: 0.5);
/// for (final f in faces) {
///   print(f);   // FaceResult(bbox=[…], score=0.97, embDim=512)
/// }
///
/// final sim = pipeline.match(faces[0].embedding!, faces[1].embedding!);
/// print('cosine similarity: $sim');
///
/// pipeline.dispose();
/// ```
class CrispFacePipeline {
  final CrispFace detector;
  final CrispFace recognizer;
  late final DynamicLibrary _lib;
  late final CrispembedFacePipelineDart _pipelineFn;
  late final CrispembedFaceFreeDart _freeFn;
  bool _disposed = false;

  /// Create a pipeline from pre-constructed [CrispFace] instances.
  ///
  /// The pipeline takes *ownership* of [detector] and [recognizer]: calling
  /// [dispose] on the pipeline also disposes both models.
  ///
  /// [libPath] — optional path to the shared library. If omitted, deduced
  ///   from the platform the same way [CrispFace] does.
  CrispFacePipeline({
    required this.detector,
    required this.recognizer,
    String? libPath,
  }) {
    _lib = _openNativeLib(libPath);
    _pipelineFn = _lib.lookupFunction<CrispembedFacePipelineNative,
        CrispembedFacePipelineDart>('crispembed_face_pipeline');
    _freeFn = _lib.lookupFunction<CrispembedFaceFreeNative,
        CrispembedFaceFreeDart>('crispembed_face_free');
  }

  /// Convenience constructor that loads both models from paths.
  ///
  /// [detectorPath] / [recognizerPath] — model paths or aliases.
  /// [nThreads] — shared thread budget (0 = auto).
  /// [libPath] — optional shared library path.
  factory CrispFacePipeline.fromPaths({
    required String detectorPath,
    required String recognizerPath,
    int nThreads = 0,
    String? libPath,
  }) {
    final det = CrispFace(detectorPath, nThreads: nThreads, libPath: libPath);
    final rec = CrispFace(recognizerPath, nThreads: nThreads, libPath: libPath);
    return CrispFacePipeline(detector: det, recognizer: rec, libPath: libPath);
  }

  // ------------------------------------------------------------------
  // Pipeline inference
  // ------------------------------------------------------------------

  /// Detect all faces in [imagePath] and return their embeddings.
  ///
  /// Internally calls `crispembed_face_pipeline` which runs detection →
  /// alignment → recognition in one pass, avoiding redundant image decoding.
  ///
  /// [conf] — detection confidence threshold (default 0.5).
  ///
  /// Returns a [List<FaceResult>] sorted by detection score descending.
  List<FaceResult> run(String imagePath, {double conf = 0.5, int detSize = 0}) {
    _checkDisposed();
    final pathPtr = imagePath.toNativeUtf8();
    final countPtr = calloc<Int32>();
    try {
      final buf = _pipelineFn(
          detector._ctx, recognizer._ctx, pathPtr, conf, detSize, countPtr);
      final n = countPtr.value;
      if (buf == nullptr || n <= 0) return [];
      return _decodePipelineBuffer(buf, n, recognizer.dim);
    } finally {
      calloc.free(pathPtr);
      calloc.free(countPtr);
    }
  }

  // ------------------------------------------------------------------
  // Similarity
  // ------------------------------------------------------------------

  /// Compute cosine similarity between two face embeddings.
  ///
  /// Both embeddings must be L2-normalized (which [run] guarantees).
  /// Returns a value in [-1, 1]; values ≥ 0.3 typically indicate the same
  /// identity (threshold is model-dependent).
  double match(Float32List emb1, Float32List emb2) {
    if (emb1.length != emb2.length || emb1.isEmpty) return 0.0;
    var dot = 0.0;
    for (var i = 0; i < emb1.length; i++) {
      dot += emb1[i] * emb2[i];
    }
    return dot;
  }

  // ------------------------------------------------------------------
  // Lifecycle
  // ------------------------------------------------------------------

  /// Release all native resources, including the underlying detector and
  /// recognizer models.
  void dispose() {
    if (!_disposed) {
      detector.dispose();
      recognizer.dispose();
      _disposed = true;
    }
  }

  void _checkDisposed() {
    if (_disposed) throw StateError('CrispFacePipeline has been disposed');
  }

}

// ---------------------------------------------------------------------------
// Internal helpers shared by CrispFace and CrispFacePipeline
// ---------------------------------------------------------------------------

/// Decode the packed detection-only buffer returned by
/// `crispembed_detect_faces`.
///
/// Layout per face (all float32):
///   [x, y, w, h, score, lm_x0, lm_y0, …, lm_x4, lm_y4]  = 15 floats
List<Map<String, dynamic>> _decodeFaceBuffer(Pointer<Void> buf, int n) {
  const _floatsPerFace = 15; // 4 bbox + 1 score + 10 landmarks
  final flat = buf.cast<Float>().asTypedList(n * _floatsPerFace);
  final results = <Map<String, dynamic>>[];
  for (var i = 0; i < n; i++) {
    final base = i * _floatsPerFace;
    results.add({
      'bbox': flat.sublist(base, base + 4).map((v) => v.toDouble()).toList(),
      'score': flat[base + 4].toDouble(),
      'landmarks': flat.sublist(base + 5, base + 15).map((v) => v.toDouble()).toList(),
    });
  }
  return results;
}

/// Decode the packed pipeline buffer returned by `crispembed_face_pipeline`.
///
/// Layout per face (all float32):
///   [x, y, w, h, score, lm_x0, lm_y0, …, lm_x4, lm_y4, emb_0, …, emb_(dim-1)]
List<FaceResult> _decodePipelineBuffer(Pointer<Void> buf, int n, int embDim) {
  final floatsPerFace = 15 + embDim; // 4 bbox + 1 score + 10 landmarks + embedding
  final flat = buf.cast<Float>().asTypedList(n * floatsPerFace);
  final results = <FaceResult>[];
  for (var i = 0; i < n; i++) {
    final base = i * floatsPerFace;
    final bbox = flat.sublist(base, base + 4).map((v) => v.toDouble()).toList();
    final score = flat[base + 4].toDouble();
    final landmarks =
        flat.sublist(base + 5, base + 15).map((v) => v.toDouble()).toList();
    final embedding = embDim > 0
        ? Float32List.fromList(flat.sublist(base + 15, base + 15 + embDim))
        : null;
    results.add(FaceResult(
      bbox: bbox,
      landmarks: landmarks,
      score: score,
      embedding: embedding,
    ));
  }
  // Sort by confidence descending, matching native detector output convention.
  results.sort((a, b) => b.score.compareTo(a.score));
  return results;
}

// ---------------------------------------------------------------------------
// Math OCR (pix2tex) — image → LaTeX
// ---------------------------------------------------------------------------

/// Wraps a pix2tex math OCR model loaded via `crispembed_math_ocr_init`.
///
/// Converts grayscale image pixel data into LaTeX strings using a ViT
/// encoder + transformer decoder architecture.
///
/// ```dart
/// final ocr = CrispMathOcr('pix2tex.gguf');
/// final latex = ocr.recognize(grayBytes, width: 320, height: 64);
/// print(latex);  // e.g. '\frac{a}{b} + c^2'
/// ocr.dispose();
/// ```
class CrispMathOcr {
  late final DynamicLibrary _lib;
  late final Pointer<Void> _ctx;
  bool _disposed = false;

  // Cached function lookups
  late final CrispembedMathOcrRecognizeDart _recognizeFn;
  late final CrispembedMathOcrFreeDart _freeFn;

  /// Load a math OCR GGUF model (auto-detects architecture: pix2tex,
  /// PP-FormulaNet, PP-FormulaNet-L, Texo, HMER, BTTR, PosFormer).
  ///
  /// [modelPath] — path to the `.gguf` model file.
  /// [nThreads] — CPU thread count (0 = auto-detect).
  /// [libPath] — optional path to the shared library. If omitted, searches
  ///   standard platform locations.
  CrispMathOcr(String modelPath, {int nThreads = 0, String? libPath}) {
    _lib = _openNativeLib(libPath);
    _bindFunctions();

    final pathPtr = modelPath.toNativeUtf8();
    _ctx = _lib
        .lookupFunction<CrispembedMathOcrInitNative, CrispembedMathOcrInitDart>(
            'crispembed_math_ocr_init')
        .call(pathPtr, nThreads);
    calloc.free(pathPtr);

    if (_ctx == nullptr) {
      throw Exception('Failed to load math OCR model: $modelPath');
    }
  }

  void _bindFunctions() {
    _recognizeFn = _lib.lookupFunction<CrispembedMathOcrRecognizeNative,
        CrispembedMathOcrRecognizeDart>('crispembed_math_ocr_recognize');
    _freeFn = _lib.lookupFunction<CrispembedMathOcrFreeNative,
        CrispembedMathOcrFreeDart>('crispembed_math_ocr_free');
  }

  // ------------------------------------------------------------------
  // Inference
  // ------------------------------------------------------------------

  /// Recognize math in a grayscale image and return a LaTeX string.
  ///
  /// [grayPixels] — raw single-channel (grayscale) pixel bytes, row-major,
  ///   length must equal [width] × [height].
  /// [width] / [height] — image dimensions in pixels.
  ///
  /// Returns the recognized LaTeX string, or an empty string on failure.
  String recognize(Uint8List grayPixels, int width, int height) {
    _checkDisposed();
    if (grayPixels.length != width * height) {
      throw ArgumentError(
          'grayPixels.length (${grayPixels.length}) must equal width * height (${width * height})');
    }

    final pixPtr = calloc<Uint8>(grayPixels.length);
    pixPtr.asTypedList(grayPixels.length).setAll(0, grayPixels);
    final lenPtr = calloc<Int32>();
    try {
      // channels = 1 for grayscale input
      final result = _recognizeFn(_ctx, pixPtr, width, height, 1, lenPtr);
      if (result == nullptr) return '';
      return result.toDartString();
    } finally {
      calloc.free(pixPtr);
      calloc.free(lenPtr);
    }
  }

  // ------------------------------------------------------------------
  // Lifecycle
  // ------------------------------------------------------------------

  /// Release all native resources. Must be called when done.
  void dispose() {
    if (!_disposed) {
      _freeFn(_ctx);
      _disposed = true;
    }
  }

  void _checkDisposed() {
    if (_disposed) throw StateError('CrispMathOcr has been disposed');
  }
}

// ---------------------------------------------------------------------------
// Standalone ViT image embedding (SigLIP, CLIP)
// ---------------------------------------------------------------------------

/// Wraps a standalone ViT model loaded via `crispembed_vit_init`.
///
/// Encodes images into dense embedding vectors using SigLIP, CLIP, or similar
/// ViT-based architectures.
///
/// ```dart
/// final vit = CrispVit('siglip-base.gguf');
/// print('dim = ${vit.dim}');
/// final emb = vit.encodeFile('photo.jpg');
/// print('embedding length: ${emb.length}');
/// vit.dispose();
/// ```
class CrispVit {
  late final DynamicLibrary _lib;
  late final Pointer<Void> _ctx;
  bool _disposed = false;

  // Cached function lookups
  late final CrispembedVitDimDart _dimFn;
  late final CrispembedVitEncodeFileDart _encodeFileFn;
  late final CrispembedVitFreeDart _freeFn;

  /// Load a SigLIP/CLIP GGUF model for image embedding.
  ///
  /// [modelPath] — path to the `.gguf` model file.
  /// [nThreads] — CPU thread count (0 = auto-detect).
  /// [libPath] — optional path to the shared library. If omitted, searches
  ///   standard platform locations.
  CrispVit(String modelPath, {int nThreads = 0, String? libPath}) {
    _lib = _openNativeLib(libPath);
    _bindFunctions();

    final pathPtr = modelPath.toNativeUtf8();
    _ctx = _lib
        .lookupFunction<CrispembedVitInitNative, CrispembedVitInitDart>(
            'crispembed_vit_init')
        .call(pathPtr, nThreads);
    calloc.free(pathPtr);

    if (_ctx == nullptr) {
      throw Exception('Failed to load ViT model: $modelPath');
    }
  }

  void _bindFunctions() {
    _dimFn = _lib.lookupFunction<CrispembedVitDimNative, CrispembedVitDimDart>(
        'crispembed_vit_dim');
    _encodeFileFn = _lib.lookupFunction<CrispembedVitEncodeFileNative,
        CrispembedVitEncodeFileDart>('crispembed_vit_encode_file');
    _freeFn = _lib.lookupFunction<CrispembedVitFreeNative,
        CrispembedVitFreeDart>('crispembed_vit_free');
  }

  // ------------------------------------------------------------------
  // Queries
  // ------------------------------------------------------------------

  /// Embedding dimension produced by the ViT model.
  int get dim {
    _checkDisposed();
    return _dimFn(_ctx);
  }

  // ------------------------------------------------------------------
  // Inference
  // ------------------------------------------------------------------

  /// Encode an image file to a dense L2-normalized embedding.
  ///
  /// Supports JPEG, PNG, BMP, and other formats readable by stb_image.
  /// Returns an empty list on failure (bad path, unsupported format, etc.).
  Float32List encodeFile(String imagePath) {
    _checkDisposed();
    final pathPtr = imagePath.toNativeUtf8();
    final dimPtr = calloc<Int32>();
    try {
      final ptr = _encodeFileFn(_ctx, pathPtr, dimPtr);
      if (ptr == nullptr || dimPtr.value <= 0) return Float32List(0);
      return Float32List.fromList(ptr.asTypedList(dimPtr.value));
    } finally {
      calloc.free(pathPtr);
      calloc.free(dimPtr);
    }
  }

  // ------------------------------------------------------------------
  // Lifecycle
  // ------------------------------------------------------------------

  /// Release all native resources. Must be called when done.
  void dispose() {
    if (!_disposed) {
      _freeFn(_ctx);
      _disposed = true;
    }
  }

  void _checkDisposed() {
    if (_disposed) throw StateError('CrispVit has been disposed');
  }
}

// ---------------------------------------------------------------------------
// Pix2Struct — document understanding (image → text)
// ---------------------------------------------------------------------------

/// Wraps a Pix2Struct model loaded via `crispembed_pix2struct_init`.
///
/// Variable-resolution ViT encoder + T5 decoder (282M params).
/// Generates text from document/chart/screen images.
///
/// ```dart
/// final p2s = CrispPix2Struct('pix2struct-base.gguf');
/// final text = p2s.generate(imageBytes, 800, 600);
/// print(text);
/// p2s.dispose();
/// ```
class CrispPix2Struct {
  late final DynamicLibrary _lib;
  late final Pointer<Void> _ctx;
  bool _disposed = false;

  // Cached function lookups
  late final CrispembedPix2StructGenerateDart _generateFn;
  late final CrispembedPix2StructFreeTextDart _freeTextFn;
  late final CrispembedPix2StructFreeDart _freeFn;

  /// Load a Pix2Struct GGUF model for document understanding.
  ///
  /// [modelPath] — path to the `.gguf` model file.
  /// [nThreads] — CPU thread count (0 = auto-detect).
  /// [libPath] — optional path to the shared library.
  CrispPix2Struct(String modelPath, {int nThreads = 0, String? libPath}) {
    _lib = _openNativeLib(libPath);
    _bindFunctions();

    final pathPtr = modelPath.toNativeUtf8();
    _ctx = _lib
        .lookupFunction<CrispembedPix2StructInitNative,
            CrispembedPix2StructInitDart>('crispembed_pix2struct_init')
        .call(pathPtr, nThreads);
    calloc.free(pathPtr);

    if (_ctx == nullptr) {
      throw Exception('Failed to load Pix2Struct model: $modelPath');
    }
  }

  void _bindFunctions() {
    _generateFn = _lib.lookupFunction<CrispembedPix2StructGenerateNative,
        CrispembedPix2StructGenerateDart>('crispembed_pix2struct_generate');
    _freeTextFn = _lib.lookupFunction<CrispembedPix2StructFreeTextNative,
        CrispembedPix2StructFreeTextDart>('crispembed_pix2struct_free_text');
    _freeFn = _lib.lookupFunction<CrispembedPix2StructFreeNative,
        CrispembedPix2StructFreeDart>('crispembed_pix2struct_free');
  }

  // ------------------------------------------------------------------
  // Inference
  // ------------------------------------------------------------------

  /// Generate text from a raw RGB image.
  ///
  /// [imageBytes] — raw RGB pixel bytes, row-major (length = width * height * 3).
  /// [width] / [height] — image dimensions.
  /// [maxTokens] — maximum number of tokens to generate (default 256).
  ///
  /// Returns the generated text string, or empty string on failure.
  String generate(Uint8List imageBytes, int width, int height,
      {int maxTokens = 256}) {
    _checkDisposed();
    final pixPtr = calloc<Uint8>(imageBytes.length);
    pixPtr.asTypedList(imageBytes.length).setAll(0, imageBytes);

    try {
      final result = _generateFn(_ctx, pixPtr, width, height, maxTokens);
      if (result == nullptr) return '';
      final text = result.toDartString();
      _freeTextFn(result);
      return text;
    } finally {
      calloc.free(pixPtr);
    }
  }

  // ------------------------------------------------------------------
  // Lifecycle
  // ------------------------------------------------------------------

  /// Release all native resources. Must be called when done.
  void dispose() {
    if (!_disposed) {
      _freeFn(_ctx);
      _disposed = true;
    }
  }

  void _checkDisposed() {
    if (_disposed) throw StateError('CrispPix2Struct has been disposed');
  }
}

// ---------------------------------------------------------------------------
// General OCR Pipeline (text detection + recognition)
// ---------------------------------------------------------------------------

/// Result from the general OCR pipeline (one detected text region).
class OcrResult {
  final String text;
  final double x, y, w, h;
  final double confidence;

  const OcrResult({
    required this.text,
    required this.x,
    required this.y,
    required this.w,
    required this.h,
    required this.confidence,
  });

  @override
  String toString() => 'OcrResult("$text" @ ($x,$y,$w,$h) conf=$confidence)';
}

/// General OCR pipeline — detects text regions (DBNet) then recognizes each
/// crop (TrOCR). Wraps `crispembed_ocr_init` / `crispembed_ocr`.
class CrispOcrPipeline {
  late final DynamicLibrary _lib;
  late final Pointer<Void> _ctx;
  bool _disposed = false;

  late final CrispembedOcrFreeDart _freeFn;
  late final CrispembedOcrRunDart _runFn;
  late final CrispembedOcrRecognizeDart _recognizeFn;

  /// Load detection + recognition models for general OCR.
  ///
  /// [detModelPath] — path to the detection GGUF (e.g. DBNet).
  /// [recModelPath] — path to the recognition GGUF (e.g. TrOCR).
  /// [nThreads] — CPU thread count (0 = auto-detect).
  CrispOcrPipeline(String detModelPath, String recModelPath,
      {int nThreads = 4, String? libPath}) {
    _lib = _openNativeLib(libPath);

    final init = _lib.lookupFunction<CrispembedOcrInitNative,
        CrispembedOcrInitDart>('crispembed_ocr_init');
    _freeFn = _lib.lookupFunction<CrispembedOcrFreeNative,
        CrispembedOcrFreeDart>('crispembed_ocr_free');
    _runFn = _lib.lookupFunction<CrispembedOcrRunNative,
        CrispembedOcrRunDart>('crispembed_ocr');
    _recognizeFn = _lib.lookupFunction<CrispembedOcrRecognizeNative,
        CrispembedOcrRecognizeDart>('crispembed_ocr_recognize');

    final detPtr = detModelPath.toNativeUtf8();
    final recPtr = recModelPath.toNativeUtf8();
    _ctx = init(detPtr, recPtr, nThreads);
    calloc.free(detPtr);
    calloc.free(recPtr);

    if (_ctx == nullptr) {
      throw Exception('Failed to load OCR pipeline: $detModelPath + $recModelPath');
    }
  }

  /// Detect and recognize text in an image file.
  /// Returns a list of [OcrResult] with bounding boxes and recognized text.
  List<OcrResult> run(String imagePath) {
    if (_disposed) return [];
    final pathPtr = imagePath.toNativeUtf8();
    final outN = calloc<Int32>();

    final ptr = _runFn(_ctx, pathPtr, outN);
    final n = outN.value;
    calloc.free(pathPtr);
    calloc.free(outN);

    if (ptr == nullptr || n <= 0) return [];

    // The C API returns crispembed_ocr_result structs:
    //   float x, y, w, h, confidence; const char* text; int text_len;
    // Total struct size: 5 floats + 1 pointer + 1 int = 28 bytes on 32-bit, 32 on 64-bit
    final results = <OcrResult>[];
    final structSize = sizeOf<Float>() * 5 + sizeOf<Pointer>() + sizeOf<Int32>();
    for (var i = 0; i < n; i++) {
      final base = ptr.cast<Uint8>().elementAt(i * structSize);
      final floats = base.cast<Float>();
      final x = floats[0];
      final y = floats[1];
      final w = floats[2];
      final h = floats[3];
      final confidence = floats[4];
      // text pointer is at offset 5 * sizeof(float)
      final textPtrAddr = base.elementAt(5 * sizeOf<Float>()).cast<Pointer<Utf8>>();
      final textPtr = textPtrAddr.value;
      final text = textPtr != nullptr ? textPtr.toDartString() : '';
      results.add(OcrResult(
        text: text,
        x: x, y: y, w: w, h: h,
        confidence: confidence,
      ));
    }
    return results;
  }

  /// Recognize text from a single image crop (no detection).
  String? recognize(String imagePath) {
    if (_disposed) return null;
    final pathPtr = imagePath.toNativeUtf8();
    final outLen = calloc<Int32>();
    final result = _recognizeFn(_ctx, pathPtr, outLen);
    final len = outLen.value;
    calloc.free(pathPtr);
    calloc.free(outLen);
    if (result == nullptr) return null;
    return result.toDartString(length: len);
  }

  void dispose() {
    if (!_disposed) {
      _freeFn(_ctx);
      _disposed = true;
    }
  }
}

// ---------------------------------------------------------------------------
// Named Entity Recognition (GLiNER)
// ---------------------------------------------------------------------------

/// A single named entity extracted by CrispNER.
class NerEntity {
  final String text;
  final String label;
  final int start;
  final int end;
  final double score;

  const NerEntity({
    required this.text,
    required this.label,
    required this.start,
    required this.end,
    required this.score,
  });

  @override
  String toString() =>
      'NerEntity("$text", label=$label, [$start, $end), score=${score.toStringAsFixed(4)})';
}

/// GLiNER zero-shot named entity recognition.
///
/// ```dart
/// final ner = CrispNER('gliner-lfm.gguf');
/// final entities = ner.extract(
///   'John works at Google in New York.',
///   labels: ['person', 'organization', 'location'],
/// );
/// for (final e in entities) {
///   print(e);
/// }
/// ner.dispose();
/// ```
class CrispNER {
  late final DynamicLibrary _lib;
  late final Pointer<Void> _ctx;
  bool _disposed = false;

  late final CrispembedNerFreeDart _freeFn;
  late final CrispembedNerExtractDart _extractFn;

  /// Load a NER GGUF model (auto-detects architecture).
  ///
  /// [modelPath] -- path to the `.gguf` model file.
  /// [nThreads] -- CPU thread count (0 = auto-detect).
  /// [libPath] -- optional path to the shared library.
  CrispNER(String modelPath, {int nThreads = 0, String? libPath}) {
    _lib = _openNativeLib(libPath);

    final init = _lib.lookupFunction<CrispembedNerInitNative,
        CrispembedNerInitDart>('crispembed_ner_init');
    _freeFn = _lib.lookupFunction<CrispembedNerFreeNative,
        CrispembedNerFreeDart>('crispembed_ner_free');
    _extractFn = _lib.lookupFunction<CrispembedNerExtractNative,
        CrispembedNerExtractDart>('crispembed_ner_extract');

    final pathPtr = modelPath.toNativeUtf8();
    _ctx = init(pathPtr, nThreads);
    calloc.free(pathPtr);

    if (_ctx == nullptr) {
      throw Exception('Failed to load NER model: $modelPath');
    }
  }

  /// Extract named entities from [text] using zero-shot [labels].
  ///
  /// [labels] -- entity types to detect (e.g. `['person', 'organization']`).
  /// [threshold] -- confidence threshold in [0, 1] (default 0.5).
  ///
  /// Returns a list of [NerEntity] results.
  List<NerEntity> extract(String text, {
    List<String> labels = const ['person', 'organization', 'location'],
    double threshold = 0.5,
  }) {
    _checkDisposed();

    final textPtr = text.toNativeUtf8();
    final nLabels = labels.length;
    final labelPtrs = calloc<Pointer<Utf8>>(nLabels);
    final nativeLabelPtrs = <Pointer<Utf8>>[];
    for (var i = 0; i < nLabels; i++) {
      final p = labels[i].toNativeUtf8();
      nativeLabelPtrs.add(p);
      labelPtrs[i] = p;
    }
    final outEntitiesPtr = calloc<Pointer<Void>>();

    try {
      final n = _extractFn(
          _ctx, textPtr, labelPtrs, nLabels, threshold, outEntitiesPtr);
      if (n <= 0) return [];

      final entPtr = outEntitiesPtr.value;
      if (entPtr == nullptr) return [];

      // crispembed_ner_entity layout:
      //   int start_char, int end_char, const char* text, const char* label, float score
      final structSize = 2 * sizeOf<Int32>() + 2 * sizeOf<Pointer>() + sizeOf<Float>();
      final results = <NerEntity>[];
      for (var i = 0; i < n; i++) {
        final base = entPtr.cast<Uint8>().elementAt(i * structSize);
        final ints = base.cast<Int32>();
        final startChar = ints[0];
        final endChar = ints[1];
        // text pointer at offset 2 * sizeof(int32)
        final textPtrAddr = base.elementAt(2 * sizeOf<Int32>()).cast<Pointer<Utf8>>();
        final entityText = textPtrAddr.value != nullptr
            ? textPtrAddr.value.toDartString()
            : '';
        // label pointer at offset 2 * sizeof(int32) + sizeof(pointer)
        final labelPtrAddr = base
            .elementAt(2 * sizeOf<Int32>() + sizeOf<Pointer>())
            .cast<Pointer<Utf8>>();
        final entityLabel = labelPtrAddr.value != nullptr
            ? labelPtrAddr.value.toDartString()
            : '';
        // score at offset 2 * sizeof(int32) + 2 * sizeof(pointer)
        final scorePtr = base
            .elementAt(2 * sizeOf<Int32>() + 2 * sizeOf<Pointer>())
            .cast<Float>();
        final score = scorePtr.value;

        results.add(NerEntity(
          text: entityText,
          label: entityLabel,
          start: startChar,
          end: endChar,
          score: score,
        ));
      }
      return results;
    } finally {
      calloc.free(textPtr);
      for (final p in nativeLabelPtrs) {
        calloc.free(p);
      }
      calloc.free(labelPtrs);
      calloc.free(outEntitiesPtr);
    }
  }

  /// Release all native resources. Must be called when done.
  void dispose() {
    if (!_disposed) {
      _freeFn(_ctx);
      _disposed = true;
    }
  }

  void _checkDisposed() {
    if (_disposed) throw StateError('CrispNER has been disposed');
  }
}

// ---------------------------------------------------------------------------
// LiLT — Language-independent Layout Transformer
// ---------------------------------------------------------------------------

/// Token classification result from LiLT.
class LiltToken {
  final int tokenId;
  final int labelId;
  final String label;
  final double score;

  const LiltToken({
    required this.tokenId,
    required this.labelId,
    required this.label,
    required this.score,
  });

  @override
  String toString() => 'LiltToken($tokenId → $label, score=${score.toStringAsFixed(3)})';
}

/// LiLT dual-stream encoder for layout-aware document understanding.
///
/// ```dart
/// final lilt = CrispLiLT('lilt-funsd-f32.gguf');
/// final tokens = lilt.classify([0, 10566, 35, 2],
///     bbox: [[0,0,0,0], [10,50,90,80], [90,50,110,80], [0,0,0,0]]);
/// for (final t in tokens) print(t);
/// lilt.dispose();
/// ```
class CrispLiLT {
  late final DynamicLibrary _lib;
  late final Pointer<Void> _ctx;
  bool _disposed = false;

  late final CrispembedLiltFreeDart _freeFn;
  late final CrispembedLiltClassifyDart _classifyFn;
  late final CrispembedLiltNumLabelsDart _numLabelsFn;

  CrispLiLT(String modelPath, {int nThreads = 0, String? libPath}) {
    _lib = _openNativeLib(libPath);

    final init = _lib.lookupFunction<CrispembedLiltInitNative,
        CrispembedLiltInitDart>('crispembed_lilt_init');
    _freeFn = _lib.lookupFunction<CrispembedLiltFreeNative,
        CrispembedLiltFreeDart>('crispembed_lilt_free');
    _classifyFn = _lib.lookupFunction<CrispembedLiltClassifyNative,
        CrispembedLiltClassifyDart>('crispembed_lilt_classify');
    _numLabelsFn = _lib.lookupFunction<CrispembedLiltNumLabelsNative,
        CrispembedLiltNumLabelsDart>('crispembed_lilt_num_labels');

    final pathPtr = modelPath.toNativeUtf8();
    _ctx = init(pathPtr, nThreads);
    calloc.free(pathPtr);

    if (_ctx == nullptr) {
      throw Exception('Failed to load LiLT model: $modelPath');
    }
  }

  int get numLabels => _numLabelsFn(_ctx);

  List<LiltToken> classify(List<int> inputIds, {required List<List<int>> bbox}) {
    _checkDisposed();
    final T = inputIds.length;
    final idsPtr = calloc<Int32>(T);
    for (var i = 0; i < T; i++) idsPtr[i] = inputIds[i];
    final bboxPtr = calloc<Int32>(T * 4);
    for (var i = 0; i < T; i++) {
      for (var j = 0; j < 4; j++) {
        bboxPtr[i * 4 + j] = (i < bbox.length && j < bbox[i].length) ? bbox[i][j] : 0;
      }
    }
    final outN = calloc<Int32>();

    try {
      final resultPtr = _classifyFn(_ctx, idsPtr, bboxPtr, T, outN);
      if (resultPtr == nullptr || outN.value <= 0) return [];

      // crispembed_lilt_token: int token_id, int label_id, char* label, float score
      final structSize = 2 * sizeOf<Int32>() + sizeOf<Pointer>() + sizeOf<Float>();
      final results = <LiltToken>[];
      for (var i = 0; i < outN.value; i++) {
        final base = resultPtr.cast<Uint8>().elementAt(i * structSize);
        final ints = base.cast<Int32>();
        final tokenId = ints[0];
        final labelId = ints[1];
        final labelPtr = base.elementAt(2 * sizeOf<Int32>()).cast<Pointer<Utf8>>();
        final label = labelPtr.value != nullptr ? labelPtr.value.toDartString() : '';
        final scorePtr = base.elementAt(2 * sizeOf<Int32>() + sizeOf<Pointer>()).cast<Float>();
        results.add(LiltToken(
          tokenId: tokenId, labelId: labelId,
          label: label, score: scorePtr.value,
        ));
      }
      return results;
    } finally {
      calloc.free(idsPtr);
      calloc.free(bboxPtr);
      calloc.free(outN);
    }
  }

  void dispose() {
    if (!_disposed) {
      _freeFn(_ctx);
      _disposed = true;
    }
  }

  void _checkDisposed() {
    if (_disposed) throw StateError('CrispLiLT has been disposed');
  }
}

// ---------------------------------------------------------------------------
// Text LID — Language Identification
// ---------------------------------------------------------------------------

/// Text-based language identification (CLD3 / GlotLID).
class CrispTextLID {
  late final DynamicLibrary _lib;
  late final Pointer<Void> _ctx;
  bool _disposed = false;

  late final TextLidFreeDart _freeFn;
  late final TextLidPredictDart _predictFn;
  late final TextLidNLabelsDart _nLabelsFn;

  CrispTextLID(String modelPath, {int nThreads = 1, String? libPath}) {
    _lib = _openNativeLib(libPath);
    final init = _lib.lookupFunction<TextLidInitNative, TextLidInitDart>(
        'text_lid_init_from_file');
    _freeFn = _lib.lookupFunction<TextLidFreeNative, TextLidFreeDart>(
        'text_lid_free');
    _predictFn = _lib.lookupFunction<TextLidPredictNative, TextLidPredictDart>(
        'text_lid_predict');
    _nLabelsFn = _lib.lookupFunction<TextLidNLabelsNative, TextLidNLabelsDart>(
        'text_lid_n_labels');
    final pathPtr = modelPath.toNativeUtf8();
    _ctx = init(pathPtr, nThreads);
    calloc.free(pathPtr);
    if (_ctx == nullptr) throw Exception('Failed to load LID model: $modelPath');
  }

  int get nLabels => _nLabelsFn(_ctx);

  /// Predict language. Returns (isoCode, confidence).
  (String, double) predict(String text) {
    if (_disposed) throw StateError('CrispTextLID has been disposed');
    final textPtr = text.toNativeUtf8();
    final confPtr = calloc<Float>();
    try {
      final langPtr = _predictFn(_ctx, textPtr, confPtr);
      final lang = langPtr != nullptr ? langPtr.toDartString() : '';
      return (lang, confPtr.value);
    } finally {
      calloc.free(textPtr);
      calloc.free(confPtr);
    }
  }

  void dispose() {
    if (!_disposed) { _freeFn(_ctx); _disposed = true; }
  }
}

// ---------------------------------------------------------------------------
// Truecaser — BiLSTM character-level
// ---------------------------------------------------------------------------

/// BiLSTM character-level truecaser.
class CrispTruecaser {
  late final DynamicLibrary _lib;
  late final Pointer<Void> _ctx;
  bool _disposed = false;

  late final TruecaserLstmFreeDart _freeFn;
  late final TruecaserLstmProcessDart _processFn;

  CrispTruecaser(String modelPath, {String? libPath}) {
    _lib = _openNativeLib(libPath);
    final init = _lib.lookupFunction<TruecaserLstmInitNative,
        TruecaserLstmInitDart>('truecaser_lstm_init');
    _freeFn = _lib.lookupFunction<TruecaserLstmFreeNative,
        TruecaserLstmFreeDart>('truecaser_lstm_free');
    _processFn = _lib.lookupFunction<TruecaserLstmProcessNative,
        TruecaserLstmProcessDart>('truecaser_lstm_process');
    final pathPtr = modelPath.toNativeUtf8();
    _ctx = init(pathPtr);
    calloc.free(pathPtr);
    if (_ctx == nullptr) throw Exception('Failed to load truecaser: $modelPath');
  }

  /// Apply truecasing. Returns truecased text.
  String process(String text) {
    if (_disposed) throw StateError('CrispTruecaser has been disposed');
    final textPtr = text.toNativeUtf8();
    try {
      final resultPtr = _processFn(_ctx, textPtr);
      return resultPtr != nullptr ? resultPtr.toDartString() : text;
    } finally {
      calloc.free(textPtr);
    }
  }

  void dispose() {
    if (!_disposed) { _freeFn(_ctx); _disposed = true; }
  }
}

// ---------------------------------------------------------------------------
// Key Information Extraction (KIE) — OCR + NER pipeline
// ---------------------------------------------------------------------------

/// A single extracted key-value field with spatial position.
class KieField {
  final String label;
  final String value;
  final double score;
  final double x, y, w, h;

  const KieField({
    required this.label,
    required this.value,
    required this.score,
    required this.x,
    required this.y,
    required this.w,
    required this.h,
  });

  @override
  String toString() =>
      'KieField($label="$value", score=${score.toStringAsFixed(3)}, '
      'bbox=[$x,$y,$w,$h])';
}

/// Result of KIE extraction.
class KieResult {
  final List<KieField> fields;
  final String ocrText;
  final double ocrConfidence;
  final int nOcrRegions;

  const KieResult({
    required this.fields,
    required this.ocrText,
    required this.ocrConfidence,
    required this.nOcrRegions,
  });
}

/// Key Information Extraction — chains OCR + NER to extract structured
/// key-value fields from document images (receipts, invoices, forms).
class CrispKIE {
  late final DynamicLibrary _lib;
  late final Pointer<Void> _ctx;
  bool _disposed = false;

  late final CrispembedKieFreeDart _freeFn;
  late final CrispembedKieExtractDart _extractFn;

  CrispKIE(String ocrDetModel, String ocrRecModel, String nerModel,
      {int nThreads = 0, String? libPath}) {
    _lib = _openNativeLib(libPath);

    final init = _lib.lookupFunction<CrispembedKieInitNative,
        CrispembedKieInitDart>('crispembed_kie_init');
    _freeFn = _lib.lookupFunction<CrispembedKieFreeNative,
        CrispembedKieFreeDart>('crispembed_kie_free');
    _extractFn = _lib.lookupFunction<CrispembedKieExtractNative,
        CrispembedKieExtractDart>('crispembed_kie_extract');

    final detPtr = ocrDetModel.toNativeUtf8();
    final recPtr = ocrRecModel.toNativeUtf8();
    final nerPtr = nerModel.toNativeUtf8();
    _ctx = init(detPtr, recPtr, nerPtr, nThreads);
    calloc.free(detPtr);
    calloc.free(recPtr);
    calloc.free(nerPtr);

    if (_ctx == nullptr) {
      throw Exception('Failed to init KIE pipeline');
    }
  }

  KieResult extract(String imagePath, {
    required List<String> labels,
    double threshold = 0.5,
  }) {
    _checkDisposed();

    final imgPtr = imagePath.toNativeUtf8();
    final nLabels = labels.length;
    final labelPtrs = calloc<Pointer<Utf8>>(nLabels);
    final nativeLabelPtrs = <Pointer<Utf8>>[];
    for (var i = 0; i < nLabels; i++) {
      final p = labels[i].toNativeUtf8();
      nativeLabelPtrs.add(p);
      labelPtrs[i] = p;
    }

    try {
      final res = _extractFn(_ctx, imgPtr, labelPtrs, nLabels, threshold);

      final fields = <KieField>[];
      if (res.nFields > 0 && res.fields != nullptr) {
        final fieldSize = 2 * sizeOf<Pointer>() + 5 * sizeOf<Float>();
        for (var i = 0; i < res.nFields; i++) {
          final base = res.fields.cast<Uint8>().elementAt(i * fieldSize);
          final labelPtr = base.cast<Pointer<Utf8>>().value;
          final valuePtr = base.elementAt(sizeOf<Pointer>()).cast<Pointer<Utf8>>().value;
          final floats = base.elementAt(2 * sizeOf<Pointer>()).cast<Float>();

          fields.add(KieField(
            label: labelPtr != nullptr ? labelPtr.toDartString() : '',
            value: valuePtr != nullptr ? valuePtr.toDartString() : '',
            score: floats[0],
            x: floats[1],
            y: floats[2],
            w: floats[3],
            h: floats[4],
          ));
        }
      }

      return KieResult(
        fields: fields,
        ocrText: res.ocrText != nullptr ? res.ocrText.toDartString() : '',
        ocrConfidence: res.ocrConfidence,
        nOcrRegions: res.nOcrRegions,
      );
    } finally {
      calloc.free(imgPtr);
      for (final p in nativeLabelPtrs) {
        calloc.free(p);
      }
      calloc.free(labelPtrs);
    }
  }

  void dispose() {
    if (!_disposed) {
      _freeFn(_ctx);
      _disposed = true;
    }
  }

  void _checkDisposed() {
    if (_disposed) throw StateError('CrispKIE has been disposed');
  }
}

// ---------------------------------------------------------------------------
// Text Super-Resolution
// ---------------------------------------------------------------------------

/// Result from text super-resolution upscaling.
class TextSrResult {
  final Uint8List pixels;
  final int width;
  final int height;

  const TextSrResult({
    required this.pixels,
    required this.width,
    required this.height,
  });
}

/// Text super-resolution model — upscales low-DPI document images.
class CrispTextSr {
  late final DynamicLibrary _lib;
  late final Pointer<Void> _ctx;
  bool _disposed = false;

  late final CrispembedTextSrFreeDart _freeFn;
  late final CrispembedTextSrUpscaleFactorDart _upscaleFactorFn;
  late final CrispembedTextSrProcessDart _processFn;
  late final CrispembedTextSrFreeImageDart _freeImageFn;

  CrispTextSr(String modelPath, {int nThreads = 0, String? libPath}) {
    _lib = _openNativeLib(libPath);
    _bindFunctions();

    final pathPtr = modelPath.toNativeUtf8();
    _ctx = _lib
        .lookupFunction<CrispembedTextSrInitNative, CrispembedTextSrInitDart>(
            'crispembed_text_sr_init')
        .call(pathPtr, nThreads);
    calloc.free(pathPtr);

    if (_ctx == nullptr) {
      throw Exception('Failed to load text SR model: $modelPath');
    }
  }

  void _bindFunctions() {
    _freeFn = _lib.lookupFunction<CrispembedTextSrFreeNative,
        CrispembedTextSrFreeDart>('crispembed_text_sr_free');
    _upscaleFactorFn = _lib.lookupFunction<CrispembedTextSrUpscaleFactorNative,
        CrispembedTextSrUpscaleFactorDart>('crispembed_text_sr_upscale_factor');
    _processFn = _lib.lookupFunction<CrispembedTextSrProcessNative,
        CrispembedTextSrProcessDart>('crispembed_text_sr_process');
    _freeImageFn = _lib.lookupFunction<CrispembedTextSrFreeImageNative,
        CrispembedTextSrFreeImageDart>('crispembed_text_sr_free_image');
  }

  int get upscaleFactor {
    _checkDisposed();
    return _upscaleFactorFn(_ctx);
  }

  TextSrResult process(
    Uint8List pixels,
    int width,
    int height, {
    int tileSize = 0,
    int tileOverlap = 0,
  }) {
    _checkDisposed();
    if (pixels.length != width * height * 3) {
      throw ArgumentError(
          'pixels.length (${pixels.length}) must equal width * height * 3 (${width * height * 3})');
    }

    final pxNative = calloc<Uint8>(pixels.length);
    pxNative.asTypedList(pixels.length).setAll(0, pixels);

    final outPxPtr = calloc<Pointer<Uint8>>();
    final outW = calloc<Int32>();
    final outH = calloc<Int32>();

    try {
      final rc = _processFn(
          _ctx, pxNative, width, height, tileSize, tileOverlap,
          outPxPtr, outW, outH);

      if (rc != 0 || outPxPtr.value.address == 0) {
        throw Exception('Text SR process failed (rc=$rc)');
      }

      final ow = outW.value;
      final oh = outH.value;
      final resultPixels =
          Uint8List.fromList(outPxPtr.value.asTypedList(ow * oh * 3));
      _freeImageFn(outPxPtr.value);

      return TextSrResult(pixels: resultPixels, width: ow, height: oh);
    } finally {
      calloc.free(pxNative);
      calloc.free(outPxPtr);
      calloc.free(outW);
      calloc.free(outH);
    }
  }

  void dispose() {
    if (!_disposed) {
      _freeFn(_ctx);
      _disposed = true;
    }
  }

  void _checkDisposed() {
    if (_disposed) throw StateError('CrispTextSr has been disposed');
  }
}

// ---------------------------------------------------------------------------
// TBSRN text-line Super-Resolution
// ---------------------------------------------------------------------------

/// Result from TBSRN text-line super-resolution upscaling.
class TbsrnSrResult {
  final Uint8List pixels;
  final int width;
  final int height;

  const TbsrnSrResult({
    required this.pixels,
    required this.width,
    required this.height,
  });
}

/// TBSRN text-line super-resolution model (PaddleOCR Telescope, 1.1M params).
///
/// Designed for text-line crops rather than whole images.
class CrispTbsrnSr {
  late final DynamicLibrary _lib;
  late final Pointer<Void> _ctx;
  bool _disposed = false;

  late final CrispembedTbsrnSrFreeDart _freeFn;
  late final CrispembedTbsrnSrProcessDart _processFn;
  late final CrispembedTbsrnSrFreeImageDart _freeImageFn;

  CrispTbsrnSr(String modelPath, {int nThreads = 0, String? libPath}) {
    _lib = _openNativeLib(libPath);
    _bindFunctions();

    final pathPtr = modelPath.toNativeUtf8();
    _ctx = _lib
        .lookupFunction<CrispembedTbsrnSrInitNative, CrispembedTbsrnSrInitDart>(
            'crispembed_tbsrn_sr_init')
        .call(pathPtr, nThreads);
    calloc.free(pathPtr);

    if (_ctx == nullptr) {
      throw Exception('Failed to load TBSRN SR model: $modelPath');
    }
  }

  void _bindFunctions() {
    _freeFn = _lib.lookupFunction<CrispembedTbsrnSrFreeNative,
        CrispembedTbsrnSrFreeDart>('crispembed_tbsrn_sr_free');
    _processFn = _lib.lookupFunction<CrispembedTbsrnSrProcessNative,
        CrispembedTbsrnSrProcessDart>('crispembed_tbsrn_sr_process');
    _freeImageFn = _lib.lookupFunction<CrispembedTbsrnSrFreeImageNative,
        CrispembedTbsrnSrFreeImageDart>('crispembed_tbsrn_sr_free_image');
  }

  TbsrnSrResult process(
    Uint8List pixels,
    int width,
    int height,
  ) {
    _checkDisposed();
    if (pixels.length != width * height * 3) {
      throw ArgumentError(
          'pixels.length (${pixels.length}) must equal width * height * 3 (${width * height * 3})');
    }

    final pxNative = calloc<Uint8>(pixels.length);
    pxNative.asTypedList(pixels.length).setAll(0, pixels);

    final outPxPtr = calloc<Pointer<Uint8>>();
    final outW = calloc<Int32>();
    final outH = calloc<Int32>();

    try {
      final rc = _processFn(
          _ctx, pxNative, width, height,
          outPxPtr, outW, outH);

      if (rc != 0 || outPxPtr.value.address == 0) {
        throw Exception('TBSRN SR process failed (rc=$rc)');
      }

      final ow = outW.value;
      final oh = outH.value;
      final resultPixels =
          Uint8List.fromList(outPxPtr.value.asTypedList(ow * oh * 3));
      _freeImageFn(outPxPtr.value);

      return TbsrnSrResult(pixels: resultPixels, width: ow, height: oh);
    } finally {
      calloc.free(pxNative);
      calloc.free(outPxPtr);
      calloc.free(outW);
      calloc.free(outH);
    }
  }

  void dispose() {
    if (!_disposed) {
      _freeFn(_ctx);
      _disposed = true;
    }
  }

  void _checkDisposed() {
    if (_disposed) throw StateError('CrispTbsrnSr has been disposed');
  }
}

// ---------------------------------------------------------------------------
// PAN Super-Resolution
// ---------------------------------------------------------------------------

/// Result from PAN super-resolution upscaling.
class PanSrResult {
  final Uint8List pixels;
  final int width;
  final int height;

  const PanSrResult({
    required this.pixels,
    required this.width,
    required this.height,
  });
}

/// PAN super-resolution model — upscales low-DPI document images.
class CrispPanSr {
  late final DynamicLibrary _lib;
  late final Pointer<Void> _ctx;
  bool _disposed = false;

  late final CrispembedPanSrFreeDart _freeFn;
  late final CrispembedPanSrScaleDart _scaleFn;
  late final CrispembedPanSrProcessDart _processFn;
  late final CrispembedPanSrFreeImageDart _freeImageFn;

  CrispPanSr(String modelPath, {int nThreads = 0, String? libPath}) {
    _lib = _openNativeLib(libPath);
    _bindFunctions();

    final pathPtr = modelPath.toNativeUtf8();
    _ctx = _lib
        .lookupFunction<CrispembedPanSrInitNative, CrispembedPanSrInitDart>(
            'crispembed_pan_sr_init')
        .call(pathPtr, nThreads);
    calloc.free(pathPtr);

    if (_ctx == nullptr) {
      throw Exception('Failed to load PAN SR model: $modelPath');
    }
  }

  void _bindFunctions() {
    _freeFn = _lib.lookupFunction<CrispembedPanSrFreeNative,
        CrispembedPanSrFreeDart>('crispembed_pan_sr_free');
    _scaleFn = _lib.lookupFunction<CrispembedPanSrScaleNative,
        CrispembedPanSrScaleDart>('crispembed_pan_sr_scale');
    _processFn = _lib.lookupFunction<CrispembedPanSrProcessNative,
        CrispembedPanSrProcessDart>('crispembed_pan_sr_process');
    _freeImageFn = _lib.lookupFunction<CrispembedPanSrFreeImageNative,
        CrispembedPanSrFreeImageDart>('crispembed_pan_sr_free_image');
  }

  int get scale {
    _checkDisposed();
    return _scaleFn(_ctx);
  }

  PanSrResult process(
    Uint8List pixels,
    int width,
    int height, {
    int tileSize = 0,
    int tileOverlap = 0,
  }) {
    _checkDisposed();
    if (pixels.length != width * height * 3) {
      throw ArgumentError(
          'pixels.length (${pixels.length}) must equal width * height * 3 (${width * height * 3})');
    }

    final pxNative = calloc<Uint8>(pixels.length);
    pxNative.asTypedList(pixels.length).setAll(0, pixels);

    final outPxPtr = calloc<Pointer<Uint8>>();
    final outW = calloc<Int32>();
    final outH = calloc<Int32>();

    try {
      final rc = _processFn(
          _ctx, pxNative, width, height, tileSize, tileOverlap,
          outPxPtr, outW, outH);

      if (rc != 0 || outPxPtr.value.address == 0) {
        throw Exception('PAN SR process failed (rc=$rc)');
      }

      final ow = outW.value;
      final oh = outH.value;
      final resultPixels =
          Uint8List.fromList(outPxPtr.value.asTypedList(ow * oh * 3));
      _freeImageFn(outPxPtr.value);

      return PanSrResult(pixels: resultPixels, width: ow, height: oh);
    } finally {
      calloc.free(pxNative);
      calloc.free(outPxPtr);
      calloc.free(outW);
      calloc.free(outH);
    }
  }

  void dispose() {
    if (!_disposed) {
      _freeFn(_ctx);
      _disposed = true;
    }
  }

  void _checkDisposed() {
    if (_disposed) throw StateError('CrispPanSr has been disposed');
  }
}

// ---------------------------------------------------------------------------
// HAT Super-Resolution (Hybrid Attention Transformer, CVPR 2023)
// ---------------------------------------------------------------------------

/// Result from HAT super-resolution upscaling.
class HatSrResult {
  final Uint8List pixels;
  final int width;
  final int height;

  const HatSrResult({
    required this.pixels,
    required this.width,
    required this.height,
  });
}

/// HAT super-resolution model — upscales low-DPI document images (CVPR 2023 SOTA, 21M params).
class CrispHatSr {
  late final DynamicLibrary _lib;
  late final Pointer<Void> _ctx;
  bool _disposed = false;

  late final CrispembedHatSrFreeDart _freeFn;
  late final CrispembedHatSrScaleDart _scaleFn;
  late final CrispembedHatSrProcessDart _processFn;
  late final CrispembedHatSrFreeImageDart _freeImageFn;

  CrispHatSr(String modelPath, {int nThreads = 0, String? libPath}) {
    _lib = _openNativeLib(libPath);
    _bindFunctions();

    final pathPtr = modelPath.toNativeUtf8();
    _ctx = _lib
        .lookupFunction<CrispembedHatSrInitNative, CrispembedHatSrInitDart>(
            'crispembed_hat_sr_init')
        .call(pathPtr, nThreads);
    calloc.free(pathPtr);

    if (_ctx == nullptr) {
      throw Exception('Failed to load HAT SR model: $modelPath');
    }
  }

  void _bindFunctions() {
    _freeFn = _lib.lookupFunction<CrispembedHatSrFreeNative,
        CrispembedHatSrFreeDart>('crispembed_hat_sr_free');
    _scaleFn = _lib.lookupFunction<CrispembedHatSrScaleNative,
        CrispembedHatSrScaleDart>('crispembed_hat_sr_scale');
    _processFn = _lib.lookupFunction<CrispembedHatSrProcessNative,
        CrispembedHatSrProcessDart>('crispembed_hat_sr_process');
    _freeImageFn = _lib.lookupFunction<CrispembedHatSrFreeImageNative,
        CrispembedHatSrFreeImageDart>('crispembed_hat_sr_free_image');
  }

  int get scale {
    _checkDisposed();
    return _scaleFn(_ctx);
  }

  HatSrResult process(
    Uint8List pixels,
    int width,
    int height, {
    int tileSize = 64,
    int tileOverlap = 8,
  }) {
    _checkDisposed();
    if (pixels.length != width * height * 3) {
      throw ArgumentError(
          'pixels.length (${pixels.length}) must equal width * height * 3 (${width * height * 3})');
    }

    final pxNative = calloc<Uint8>(pixels.length);
    pxNative.asTypedList(pixels.length).setAll(0, pixels);

    final outPxPtr = calloc<Pointer<Uint8>>();
    final outW = calloc<Int32>();
    final outH = calloc<Int32>();

    try {
      final rc = _processFn(
          _ctx, pxNative, width, height, tileSize, tileOverlap,
          outPxPtr, outW, outH);

      if (rc != 0 || outPxPtr.value.address == 0) {
        throw Exception('HAT SR process failed (rc=$rc)');
      }

      final ow = outW.value;
      final oh = outH.value;
      final resultPixels =
          Uint8List.fromList(outPxPtr.value.asTypedList(ow * oh * 3));
      _freeImageFn(outPxPtr.value);

      return HatSrResult(pixels: resultPixels, width: ow, height: oh);
    } finally {
      calloc.free(pxNative);
      calloc.free(outPxPtr);
      calloc.free(outW);
      calloc.free(outH);
    }
  }

  void dispose() {
    if (!_disposed) {
      _freeFn(_ctx);
      _disposed = true;
    }
  }

  void _checkDisposed() {
    if (_disposed) throw StateError('CrispHatSr has been disposed');
  }
}

// ---------------------------------------------------------------------------
// DAT Super-Resolution (Dual Aggregation Transformer, ICCV 2023)
// ---------------------------------------------------------------------------

class DatSrResult {
  final Uint8List pixels;
  final int width;
  final int height;
  const DatSrResult({required this.pixels, required this.width, required this.height});
}

class CrispDatSr {
  late final DynamicLibrary _lib;
  late final Pointer<Void> _ctx;
  bool _disposed = false;

  late final CrispembedDatSrFreeDart _freeFn;
  late final CrispembedDatSrProcessDart _processFn;
  late final CrispembedDatSrFreeImageDart _freeImageFn;

  CrispDatSr(String modelPath, {int nThreads = 0, String? libPath}) {
    _lib = _openNativeLib(libPath);
    _freeFn = _lib.lookupFunction<CrispembedDatSrFreeNative, CrispembedDatSrFreeDart>('crispembed_dat_sr_free');
    _processFn = _lib.lookupFunction<CrispembedDatSrProcessNative, CrispembedDatSrProcessDart>('crispembed_dat_sr_process');
    _freeImageFn = _lib.lookupFunction<CrispembedDatSrFreeImageNative, CrispembedDatSrFreeImageDart>('crispembed_dat_sr_free_image');

    final pathPtr = modelPath.toNativeUtf8();
    _ctx = _lib.lookupFunction<CrispembedDatSrInitNative, CrispembedDatSrInitDart>('crispembed_dat_sr_init').call(pathPtr, nThreads);
    calloc.free(pathPtr);
    if (_ctx == nullptr) throw Exception('Failed to load DAT SR model: $modelPath');
  }

  DatSrResult process(Uint8List pixels, int width, int height, {int tileW = 0, int tileH = 0}) {
    _checkDisposed();
    final pxNative = calloc<Uint8>(pixels.length);
    pxNative.asTypedList(pixels.length).setAll(0, pixels);
    final outPxPtr = calloc<Pointer<Uint8>>();
    final outW = calloc<Int32>();
    final outH = calloc<Int32>();
    try {
      final rc = _processFn(_ctx, pxNative, width, height, tileW, tileH, outPxPtr, outW, outH);
      if (rc != 0 || outPxPtr.value.address == 0) throw Exception('DAT SR process failed');
      final ow = outW.value, oh = outH.value;
      final result = Uint8List.fromList(outPxPtr.value.asTypedList(ow * oh * 3));
      _freeImageFn(outPxPtr.value);
      return DatSrResult(pixels: result, width: ow, height: oh);
    } finally {
      calloc.free(pxNative);
      calloc.free(outPxPtr);
      calloc.free(outW);
      calloc.free(outH);
    }
  }

  void dispose() {
    if (!_disposed) { _freeFn(_ctx); _disposed = true; }
  }

  void _checkDisposed() {
    if (_disposed) throw StateError('CrispDatSr has been disposed');
  }
}

/// Result from SAFMN super-resolution.
class SafmnSrResult {
  final Uint8List pixels;
  final int width;
  final int height;

  const SafmnSrResult({
    required this.pixels,
    required this.width,
    required this.height,
  });
}

/// SAFMN super-resolution model — lightweight upscaling with SAFM+CCM AttBlocks.
class CrispSafmnSr {
  late final DynamicLibrary _lib;
  late final Pointer<Void> _ctx;
  bool _disposed = false;

  late final CrispembedSafmnSrFreeDart _freeFn;
  late final CrispembedSafmnSrScaleDart _scaleFn;
  late final CrispembedSafmnSrProcessDart _processFn;
  late final CrispembedSafmnSrFreeImageDart _freeImageFn;

  CrispSafmnSr(String modelPath, {int nThreads = 0, String? libPath}) {
    _lib = _openNativeLib(libPath);
    _bindFunctions();

    final pathPtr = modelPath.toNativeUtf8();
    _ctx = _lib
        .lookupFunction<CrispembedSafmnSrInitNative, CrispembedSafmnSrInitDart>(
            'crispembed_safmn_sr_init')
        .call(pathPtr, nThreads);
    calloc.free(pathPtr);

    if (_ctx == nullptr) {
      throw Exception('Failed to load SAFMN SR model: $modelPath');
    }
  }

  void _bindFunctions() {
    _freeFn = _lib.lookupFunction<CrispembedSafmnSrFreeNative,
        CrispembedSafmnSrFreeDart>('crispembed_safmn_sr_free');
    _scaleFn = _lib.lookupFunction<CrispembedSafmnSrScaleNative,
        CrispembedSafmnSrScaleDart>('crispembed_safmn_sr_scale');
    _processFn = _lib.lookupFunction<CrispembedSafmnSrProcessNative,
        CrispembedSafmnSrProcessDart>('crispembed_safmn_sr_process');
    _freeImageFn = _lib.lookupFunction<CrispembedSafmnSrFreeImageNative,
        CrispembedSafmnSrFreeImageDart>('crispembed_safmn_sr_free_image');
  }

  int get scale {
    _checkDisposed();
    return _scaleFn(_ctx);
  }

  SafmnSrResult process(
    Uint8List pixels,
    int width,
    int height, {
    int tileSize = 0,
    int tileOverlap = 0,
  }) {
    _checkDisposed();
    if (pixels.length != width * height * 3) {
      throw ArgumentError(
          'pixels.length (${pixels.length}) must equal width * height * 3 (${width * height * 3})');
    }

    final pxNative = calloc<Uint8>(pixels.length);
    pxNative.asTypedList(pixels.length).setAll(0, pixels);

    final outPxPtr = calloc<Pointer<Uint8>>();
    final outW = calloc<Int32>();
    final outH = calloc<Int32>();

    try {
      final rc = _processFn(
          _ctx, pxNative, width, height, tileSize, tileOverlap,
          outPxPtr, outW, outH);

      if (rc != 0 || outPxPtr.value.address == 0) {
        throw Exception('SAFMN SR process failed (rc=$rc)');
      }

      final ow = outW.value;
      final oh = outH.value;
      final resultPixels =
          Uint8List.fromList(outPxPtr.value.asTypedList(ow * oh * 3));
      _freeImageFn(outPxPtr.value);

      return SafmnSrResult(pixels: resultPixels, width: ow, height: oh);
    } finally {
      calloc.free(pxNative);
      calloc.free(outPxPtr);
      calloc.free(outW);
      calloc.free(outH);
    }
  }

  void dispose() {
    if (!_disposed) {
      _freeFn(_ctx);
      _disposed = true;
    }
  }

  void _checkDisposed() {
    if (_disposed) throw StateError('CrispSafmnSr has been disposed');
  }
}

// ---------------------------------------------------------------------------
// Real-ESRGAN Super-Resolution (SRVGGNetCompact, 620K params)
// ---------------------------------------------------------------------------

/// Result from Real-ESRGAN super-resolution.
class EsrganSrResult {
  final Uint8List pixels;
  final int width;
  final int height;

  const EsrganSrResult({
    required this.pixels,
    required this.width,
    required this.height,
  });
}

/// Real-ESRGAN super-resolution model — SRVGGNetCompact whole-image upscaling.
class CrispEsrganSr {
  late final DynamicLibrary _lib;
  late final Pointer<Void> _ctx;
  bool _disposed = false;

  late final CrispembedEsrganSrFreeDart _freeFn;
  late final CrispembedEsrganSrScaleDart _scaleFn;
  late final CrispembedEsrganSrProcessDart _processFn;
  late final CrispembedEsrganSrFreeImageDart _freeImageFn;

  CrispEsrganSr(String modelPath, {int nThreads = 0, String? libPath}) {
    _lib = _openNativeLib(libPath);
    _bindFunctions();

    final pathPtr = modelPath.toNativeUtf8();
    _ctx = _lib
        .lookupFunction<CrispembedEsrganSrInitNative, CrispembedEsrganSrInitDart>(
            'crispembed_esrgan_sr_init')
        .call(pathPtr, nThreads);
    calloc.free(pathPtr);

    if (_ctx == nullptr) {
      throw Exception('Failed to load Real-ESRGAN SR model: $modelPath');
    }
  }

  void _bindFunctions() {
    _freeFn = _lib.lookupFunction<CrispembedEsrganSrFreeNative,
        CrispembedEsrganSrFreeDart>('crispembed_esrgan_sr_free');
    _scaleFn = _lib.lookupFunction<CrispembedEsrganSrScaleNative,
        CrispembedEsrganSrScaleDart>('crispembed_esrgan_sr_scale');
    _processFn = _lib.lookupFunction<CrispembedEsrganSrProcessNative,
        CrispembedEsrganSrProcessDart>('crispembed_esrgan_sr_process');
    _freeImageFn = _lib.lookupFunction<CrispembedEsrganSrFreeImageNative,
        CrispembedEsrganSrFreeImageDart>('crispembed_esrgan_sr_free_image');
  }

  int get scale {
    _checkDisposed();
    return _scaleFn(_ctx);
  }

  EsrganSrResult process(
    Uint8List pixels,
    int width,
    int height, {
    int tileSize = 0,
    int tileOverlap = 0,
  }) {
    _checkDisposed();
    if (pixels.length != width * height * 3) {
      throw ArgumentError(
          'pixels.length (${pixels.length}) must equal width * height * 3 (${width * height * 3})');
    }

    final pxNative = calloc<Uint8>(pixels.length);
    pxNative.asTypedList(pixels.length).setAll(0, pixels);

    final outPxPtr = calloc<Pointer<Uint8>>();
    final outW = calloc<Int32>();
    final outH = calloc<Int32>();

    try {
      final rc = _processFn(
          _ctx, pxNative, width, height, tileSize, tileOverlap,
          outPxPtr, outW, outH);

      if (rc != 0 || outPxPtr.value.address == 0) {
        throw Exception('Real-ESRGAN SR process failed (rc=$rc)');
      }

      final ow = outW.value;
      final oh = outH.value;
      final resultPixels =
          Uint8List.fromList(outPxPtr.value.asTypedList(ow * oh * 3));
      _freeImageFn(outPxPtr.value);

      return EsrganSrResult(pixels: resultPixels, width: ow, height: oh);
    } finally {
      calloc.free(pxNative);
      calloc.free(outPxPtr);
      calloc.free(outW);
      calloc.free(outH);
    }
  }

  void dispose() {
    if (!_disposed) {
      _freeFn(_ctx);
      _disposed = true;
    }
  }

  void _checkDisposed() {
    if (_disposed) throw StateError('CrispEsrganSr has been disposed');
  }
}

// ---------------------------------------------------------------------------
// SwinIR-light Super-Resolution (Swin Transformer, Apache-2.0)
// ---------------------------------------------------------------------------

/// Result from SwinIR super-resolution.
class SwinirSrResult {
  final Uint8List pixels;
  final int width;
  final int height;

  const SwinirSrResult({
    required this.pixels,
    required this.width,
    required this.height,
  });
}

/// SwinIR-light super-resolution model — Swin Transformer whole-image upscaling.
class CrispSwinirSr {
// SCUNet Denoising (Swin-Conv-UNet, 18M params)
// ---------------------------------------------------------------------------

/// SCUNet image denoising model — Swin-Conv-UNet hybrid blocks.
class CrispScunet {
  late final DynamicLibrary _lib;
  late final Pointer<Void> _ctx;
  bool _disposed = false;

  late final CrispembedSwinirSrFreeDart _freeFn;
  late final CrispembedSwinirSrScaleDart _scaleFn;
  late final CrispembedSwinirSrProcessDart _processFn;
  late final CrispembedSwinirSrFreeImageDart _freeImageFn;

  CrispSwinirSr(String modelPath, {int nThreads = 0, String? libPath}) {
  late final CrispembedScunetFreeDart _freeFn;
  late final CrispembedScunetProcessDart _processFn;
  late final CrispembedScunetFreeImageDart _freeImageFn;

  CrispScunet(String modelPath, {int nThreads = 0, String? libPath}) {
    _lib = _openNativeLib(libPath);
    _bindFunctions();

    final pathPtr = modelPath.toNativeUtf8();
    _ctx = _lib
        .lookupFunction<CrispembedSwinirSrInitNative, CrispembedSwinirSrInitDart>(
            'crispembed_swinir_sr_init')
        .lookupFunction<CrispembedScunetInitNative, CrispembedScunetInitDart>(
            'crispembed_scunet_init')
        .call(pathPtr, nThreads);
    calloc.free(pathPtr);

    if (_ctx == nullptr) {
      throw Exception('Failed to load SwinIR SR model: $modelPath');
      throw Exception('Failed to load SCUNet model: $modelPath');
    }
  }

  void _bindFunctions() {
    _freeFn = _lib.lookupFunction<CrispembedSwinirSrFreeNative,
        CrispembedSwinirSrFreeDart>('crispembed_swinir_sr_free');
    _scaleFn = _lib.lookupFunction<CrispembedSwinirSrScaleNative,
        CrispembedSwinirSrScaleDart>('crispembed_swinir_sr_scale');
    _processFn = _lib.lookupFunction<CrispembedSwinirSrProcessNative,
        CrispembedSwinirSrProcessDart>('crispembed_swinir_sr_process');
    _freeImageFn = _lib.lookupFunction<CrispembedSwinirSrFreeImageNative,
        CrispembedSwinirSrFreeImageDart>('crispembed_swinir_sr_free_image');
  }

  int get scale {
    _checkDisposed();
    return _scaleFn(_ctx);
  }

  SwinirSrResult process(
    Uint8List pixels,
    int width,
    int height, {
    int tileSize = 0,
    int tileOverlap = 0,
  }) {
    _freeFn = _lib.lookupFunction<CrispembedScunetFreeNative,
        CrispembedScunetFreeDart>('crispembed_scunet_free');
    _processFn = _lib.lookupFunction<CrispembedScunetProcessNative,
        CrispembedScunetProcessDart>('crispembed_scunet_process');
    _freeImageFn = _lib.lookupFunction<CrispembedScunetFreeImageNative,
        CrispembedScunetFreeImageDart>('crispembed_scunet_free_image');
  }

  /// Denoise an image. Output has the same dimensions as input.
  Uint8List process(Uint8List pixels, int width, int height) {
    _checkDisposed();
    if (pixels.length != width * height * 3) {
      throw ArgumentError(
          'pixels.length (${pixels.length}) must equal width * height * 3 (${width * height * 3})');
    }

    final pxNative = calloc<Uint8>(pixels.length);
    pxNative.asTypedList(pixels.length).setAll(0, pixels);

    final outPxPtr = calloc<Pointer<Uint8>>();
    final outW = calloc<Int32>();
    final outH = calloc<Int32>();

    try {
      final rc = _processFn(
          _ctx, pxNative, width, height, tileSize, tileOverlap,
          outPxPtr, outW, outH);

      if (rc != 0 || outPxPtr.value.address == 0) {
        throw Exception('SwinIR SR process failed (rc=$rc)');
      }

      final ow = outW.value;
      final oh = outH.value;
      final resultPixels =
          Uint8List.fromList(outPxPtr.value.asTypedList(ow * oh * 3));
      _freeImageFn(outPxPtr.value);

      return SwinirSrResult(pixels: resultPixels, width: ow, height: oh);
    } finally {
      calloc.free(pxNative);
      calloc.free(outPxPtr);
      calloc.free(outW);
      calloc.free(outH);

    try {
      final rc = _processFn(_ctx, pxNative, width, height, outPxPtr);

      if (rc != 0 || outPxPtr.value.address == 0) {
        throw Exception('SCUNet denoising failed (rc=$rc)');
      }

      final resultPixels =
          Uint8List.fromList(outPxPtr.value.asTypedList(width * height * 3));
      _freeImageFn(outPxPtr.value);

      return resultPixels;
    } finally {
      calloc.free(pxNative);
      calloc.free(outPxPtr);
    }
  }

  void dispose() {
    if (!_disposed) {
      _freeFn(_ctx);
      _disposed = true;
    }
  }

  void _checkDisposed() {
    if (_disposed) throw StateError('CrispSwinirSr has been disposed');
    if (_disposed) throw StateError('CrispScunet has been disposed');
  }
}

// ---------------------------------------------------------------------------
// InstructIR All-in-One Restoration (NAFNet+ICB, 16M params, MIT)
// ---------------------------------------------------------------------------

/// InstructIR all-in-one image restoration model — 7 tasks.
/// Tasks: 0=denoise, 1=deblur, 2=dehaze, 3=derain,
///        4=super_resolution, 5=low_light, 6=enhance.
class CrispInstructIR {
  late final DynamicLibrary _lib;
  late final Pointer<Void> _ctx;
  bool _disposed = false;

  late final CrispembedInstructirFreeDart _freeFn;
  late final CrispembedInstructirNTasksDart _nTasksFn;
  late final CrispembedInstructirProcessDart _processFn;
  late final CrispembedInstructirFreeImageDart _freeImageFn;

  CrispInstructIR(String modelPath, {int nThreads = 0, String? libPath}) {
    _lib = _openNativeLib(libPath);
    _bindFunctions();

    final pathPtr = modelPath.toNativeUtf8();
    _ctx = _lib
        .lookupFunction<CrispembedInstructirInitNative, CrispembedInstructirInitDart>(
            'crispembed_instructir_init')
        .call(pathPtr, nThreads);
    calloc.free(pathPtr);

    if (_ctx == nullptr) {
      throw Exception('Failed to load InstructIR model: $modelPath');
    }
  }

  void _bindFunctions() {
    _freeFn = _lib.lookupFunction<CrispembedInstructirFreeNative,
        CrispembedInstructirFreeDart>('crispembed_instructir_free');
    _nTasksFn = _lib.lookupFunction<CrispembedInstructirNTasksNative,
        CrispembedInstructirNTasksDart>('crispembed_instructir_n_tasks');
    _processFn = _lib.lookupFunction<CrispembedInstructirProcessNative,
        CrispembedInstructirProcessDart>('crispembed_instructir_process');
    _freeImageFn = _lib.lookupFunction<CrispembedInstructirFreeImageNative,
        CrispembedInstructirFreeImageDart>('crispembed_instructir_free_image');
  }

  int get nTasks {
    _checkDisposed();
    return _nTasksFn(_ctx);
  }

  /// Restore an image with the specified task. Output has same dimensions.
  Uint8List process(Uint8List pixels, int width, int height, {int task = 0}) {
    _checkDisposed();
    if (pixels.length != width * height * 3) {
      throw ArgumentError(
          'pixels.length (${pixels.length}) must equal width * height * 3 (${width * height * 3})');
    }

    final pxNative = calloc<Uint8>(pixels.length);
    pxNative.asTypedList(pixels.length).setAll(0, pixels);

    final outPxPtr = calloc<Pointer<Uint8>>();

    try {
      final rc = _processFn(_ctx, task, pxNative, width, height, outPxPtr);

      if (rc != 0 || outPxPtr.value.address == 0) {
        throw Exception('InstructIR processing failed (rc=$rc, task=$task)');
      }

      final resultPixels =
          Uint8List.fromList(outPxPtr.value.asTypedList(width * height * 3));
      _freeImageFn(outPxPtr.value);

      return resultPixels;
    } finally {
      calloc.free(pxNative);
      calloc.free(outPxPtr);
    }
  }

  void dispose() {
    if (!_disposed) {
      _freeFn(_ctx);
      _disposed = true;
    }
  }

  void _checkDisposed() {
    if (_disposed) throw StateError('CrispInstructIR has been disposed');
  }
}

// ---------------------------------------------------------------------------
// AdaIR All-in-One Restoration (Restormer+AFLB+FFT, 28.8M params, MIT)
// ---------------------------------------------------------------------------

/// AdaIR all-in-one image restoration model — 5 tasks.
/// Tasks: denoise, derain, dehaze, deblur, low-light.
class CrispAdaIR {
  late final DynamicLibrary _lib;
  late final Pointer<Void> _ctx;
  bool _disposed = false;

  late final CrispembedAdairFreeDart _freeFn;
  late final CrispembedAdairProcessDart _processFn;
  late final CrispembedAdairFreeImageDart _freeImageFn;

  CrispAdaIR(String modelPath, {int nThreads = 0, String? libPath}) {
    _lib = _openNativeLib(libPath);
    _bindFunctions();

    final pathPtr = modelPath.toNativeUtf8();
    _ctx = _lib
        .lookupFunction<CrispembedAdairInitNative, CrispembedAdairInitDart>(
            'crispembed_adair_init')
        .call(pathPtr, nThreads);
    calloc.free(pathPtr);

    if (_ctx == nullptr) {
      throw Exception('Failed to load AdaIR model: $modelPath');
    }
  }

  void _bindFunctions() {
    _freeFn = _lib.lookupFunction<CrispembedAdairFreeNative,
        CrispembedAdairFreeDart>('crispembed_adair_free');
    _processFn = _lib.lookupFunction<CrispembedAdairProcessNative,
        CrispembedAdairProcessDart>('crispembed_adair_process');
    _freeImageFn = _lib.lookupFunction<CrispembedAdairFreeImageNative,
        CrispembedAdairFreeImageDart>('crispembed_adair_free_image');
  }

  /// Restore an image. Output has the same dimensions as input.
  Uint8List process(Uint8List pixels, int width, int height) {
    _checkDisposed();
    if (pixels.length != width * height * 3) {
      throw ArgumentError(
          'pixels.length (${pixels.length}) must equal width * height * 3 (${width * height * 3})');
    }

    final pxNative = calloc<Uint8>(pixels.length);
    pxNative.asTypedList(pixels.length).setAll(0, pixels);

    final outPxPtr = calloc<Pointer<Uint8>>();

    try {
      final rc = _processFn(_ctx, pxNative, width, height, outPxPtr);

      if (rc != 0 || outPxPtr.value.address == 0) {
        throw Exception('AdaIR restoration failed (rc=$rc)');
      }

      final resultPixels =
          Uint8List.fromList(outPxPtr.value.asTypedList(width * height * 3));
      _freeImageFn(outPxPtr.value);

      return resultPixels;
    } finally {
      calloc.free(pxNative);
      calloc.free(outPxPtr);
    }
  }

  void dispose() {
    if (!_disposed) {
      _freeFn(_ctx);
      _disposed = true;
    }
  }

  void _checkDisposed() {
    if (_disposed) throw StateError('CrispAdaIR has been disposed');
  }
}

// ---------------------------------------------------------------------------
// TBSRN Super-Resolution (always 2×, 16×64 → 32×128)
// ---------------------------------------------------------------------------

/// Result from TBSRN super-resolution upscaling.
class TbsrnSrResult {
  final Uint8List pixels;
  final int width;
  final int height;

  const TbsrnSrResult({
    required this.pixels,
    required this.width,
    required this.height,
  });
}

/// TBSRN super-resolution model — upscales text-image crops (always 2×).
class CrispTbsrnSr {
  late final DynamicLibrary _lib;
  late final Pointer<Void> _ctx;
  bool _disposed = false;

  late final CrispembedTbsrnSrFreeDart _freeFn;
  late final CrispembedTbsrnSrProcessDart _processFn;
  late final CrispembedTbsrnSrFreeImageDart _freeImageFn;

  /// TBSRN is always 2×.
  static const int scale = 2;

  CrispTbsrnSr(String modelPath, {int nThreads = 0, String? libPath}) {
    _lib = _openNativeLib(libPath);
    _bindFunctions();

    final pathPtr = modelPath.toNativeUtf8();
    _ctx = _lib
        .lookupFunction<CrispembedTbsrnSrInitNative, CrispembedTbsrnSrInitDart>(
            'crispembed_tbsrn_sr_init')
        .call(pathPtr, nThreads);
    calloc.free(pathPtr);

    if (_ctx == nullptr) {
      throw Exception('Failed to load TBSRN SR model: $modelPath');
    }
  }

  void _bindFunctions() {
    _freeFn = _lib.lookupFunction<CrispembedTbsrnSrFreeNative,
        CrispembedTbsrnSrFreeDart>('crispembed_tbsrn_sr_free');
    _processFn = _lib.lookupFunction<CrispembedTbsrnSrProcessNative,
        CrispembedTbsrnSrProcessDart>('crispembed_tbsrn_sr_process');
    _freeImageFn = _lib.lookupFunction<CrispembedTbsrnSrFreeImageNative,
        CrispembedTbsrnSrFreeImageDart>('crispembed_tbsrn_sr_free_image');
  }

  TbsrnSrResult process(
    Uint8List pixels,
    int width,
    int height,
  ) {
    _checkDisposed();
    if (pixels.length != width * height * 3) {
      throw ArgumentError(
          'pixels.length (${pixels.length}) must equal width * height * 3 (${width * height * 3})');
    }

    final pxNative = calloc<Uint8>(pixels.length);
    pxNative.asTypedList(pixels.length).setAll(0, pixels);

    final outPxPtr = calloc<Pointer<Uint8>>();
    final outW = calloc<Int32>();
    final outH = calloc<Int32>();

    try {
      final rc = _processFn(
          _ctx, pxNative, width, height,
          outPxPtr, outW, outH);

      if (rc != 0 || outPxPtr.value.address == 0) {
        throw Exception('TBSRN SR process failed (rc=$rc)');
      }

      final ow = outW.value;
      final oh = outH.value;
      final resultPixels =
          Uint8List.fromList(outPxPtr.value.asTypedList(ow * oh * 3));
      _freeImageFn(outPxPtr.value);

      return TbsrnSrResult(pixels: resultPixels, width: ow, height: oh);
    } finally {
      calloc.free(pxNative);
      calloc.free(outPxPtr);
      calloc.free(outW);
      calloc.free(outH);
    }
  }

  void dispose() {
    if (!_disposed) {
      _freeFn(_ctx);
      _disposed = true;
    }
  }

  void _checkDisposed() {
    if (_disposed) throw StateError('CrispTbsrnSr has been disposed');
  }
}

// ---------------------------------------------------------------------------
// Table Structure Recognition
// ---------------------------------------------------------------------------

/// Table structure recognition — extracts an HTML table from a grayscale image.
///
/// Uses morphological line detection and grid intersection analysis. Optional
/// built-in cell OCR via a Tesseract LSTM GGUF model.
///
/// ```dart
/// final tp = CrispTableParse();                       // no OCR
/// final tp = CrispTableParse(ocrModelPath: 'tess.gguf');
/// final html = tp.toHtml(grayPixels, width, height);  // Uint8List, row-major
/// tp.dispose();
/// ```
class CrispTableParse {
  late final DynamicLibrary _lib;
  late final Pointer<Void> _ctx;
  bool _disposed = false;

  late final CrispembedTableParseFreeDart _freeFn;
  late final CrispembedTableParseToHtmlDart _toHtmlFn;
  late final CrispembedTableParseFreeStringDart _freeStringFn;

  /// Initialize the table parser.
  ///
  /// [ocrModelPath]: path to a Tesseract LSTM GGUF for built-in cell OCR.
  ///   Pass null to skip OCR (cells will be empty in the returned HTML).
  /// [nThreads]: CPU threads for OCR (0 = auto).
  CrispTableParse({String? ocrModelPath, int nThreads = 0, String? libPath}) {
    _lib = _openNativeLib(libPath);
    _bindFunctions();

    final pathPtr = ocrModelPath != null ? ocrModelPath.toNativeUtf8() : nullptr;
    _ctx = _lib
        .lookupFunction<CrispembedTableParseInitNative,
            CrispembedTableParseInitDart>('crispembed_table_parse_init')
        .call(pathPtr.cast<Utf8>(), nThreads);
    if (ocrModelPath != null) calloc.free(pathPtr);

    if (_ctx == nullptr) {
      throw Exception('Failed to init table parser'
          '${ocrModelPath != null ? " with OCR model: $ocrModelPath" : ""}');
    }
  }

  void _bindFunctions() {
    _freeFn = _lib.lookupFunction<CrispembedTableParseFreeNative,
        CrispembedTableParseFreeDart>('crispembed_table_parse_free');
    _toHtmlFn = _lib.lookupFunction<CrispembedTableParseToHtmlNative,
        CrispembedTableParseToHtmlDart>('crispembed_table_parse_to_html');
    _freeStringFn = _lib.lookupFunction<CrispembedTableParseFreeStringNative,
        CrispembedTableParseFreeStringDart>('crispembed_table_parse_free_string');
  }

  /// Parse a grayscale table image and return an HTML string.
  ///
  /// [gray]: uint8 grayscale pixels, row-major [height × width].
  /// Returns an HTML string containing a `<table>` element.
  String toHtml(Uint8List gray, int width, int height) {
    _checkDisposed();
    if (gray.length != width * height) {
      throw ArgumentError(
          'gray.length (${gray.length}) must equal width * height ($width * $height)');
    }

    final pxNative = calloc<Uint8>(gray.length);
    pxNative.asTypedList(gray.length).setAll(0, gray);

    try {
      final ptr = _toHtmlFn(_ctx, pxNative, width, height);
      if (ptr == nullptr) {
        throw Exception('Table parsing failed');
      }
      final result = ptr.toDartString();
      _freeStringFn(ptr);
      return result;
    } finally {
      calloc.free(pxNative);
    }
  }

  void dispose() {
    if (!_disposed) {
      _freeFn(_ctx);
      _disposed = true;
    }
  }

  void _checkDisposed() {
    if (_disposed) throw StateError('CrispTableParse has been disposed');
  }
}

// ---------------------------------------------------------------------------
// OCR Orchestrator (source-type routing + cleanup + accept-gate)
// ---------------------------------------------------------------------------

/// Full OCR orchestrator with source-type routing, scan cleanup,
/// multi-engine cascading, and accept-gate fallback.
///
/// Wraps `crispembed_ocr_pipeline_init` / `crispembed_ocr_pipeline_run`.
class CrispOcrOrchestrator {
  late final DynamicLibrary _lib;
  late final Pointer<Void> _ctx;
  bool _disposed = false;

  late final CrispembedOcrPipelineFreeDart _freeFn;
  late final CrispembedOcrPipelineRunDart _runFn;

  /// Load the orchestrator with detection + recognition models.
  ///
  /// [detModelPath] — text detection GGUF (DBNet / Surya).
  /// [recModelPath] — text recognition GGUF (TrOCR / VLM).
  /// [router] — enable source-type classification (screenshot/scan/photo).
  /// [cleanup] — enable scan cleanup (deskew, crop, whiten).
  /// [minChars] — accept-gate minimum text yield.
  /// [minConfidence] — accept-gate confidence floor.
  /// [nafnetModel] — optional NAFNet GGUF for tier-2 learned denoise.
  /// [srModel] — optional text super-resolution GGUF for low-DPI upscaling.
  /// [vlmModel] — optional VLM GGUF for escalation fallback.
  /// [vlmEngine] — VLM engine: 0=GOT, 1=GLM, 2=Qwen2-VL, 3=InternVL2.
  /// [punctModel] — optional punctuation restoration model GGUF.
  CrispOcrOrchestrator(String detModelPath, String recModelPath,
      {int nThreads = 4,
      bool router = true,
      bool cleanup = true,
      int minChars = 8,
      double minConfidence = 0.5,
      String? nafnetModel,
      String? srModel,
      String? vlmModel,
      int vlmEngine = 0,
      String? punctModel,
      String? libPath}) {
    _lib = _openNativeLib(libPath);

    final initFn = _lib.lookupFunction<CrispembedOcrPipelineInitNative,
        CrispembedOcrPipelineInitDart>('crispembed_ocr_pipeline_init');
    _freeFn = _lib.lookupFunction<CrispembedOcrPipelineFreeNative,
        CrispembedOcrPipelineFreeDart>('crispembed_ocr_pipeline_free');
    _runFn = _lib.lookupFunction<CrispembedOcrPipelineRunNative,
        CrispembedOcrPipelineRunDart>('crispembed_ocr_pipeline_run');

    // Build crispembed_ocr_pipeline_params struct.
    // Layout (64-bit): 3×int32 + 1×float + 6×pointer + 1×int32 + 1×pointer
    //   router(4) + cleanup_enabled(4) + min_chars(4) + min_confidence(4)
    //   + det_model(8) + rec_model(8) + nafnet_model(8) + sr_model(8)
    //   + vlm_model(8) + vlm_engine(4) + pad(4) + punct_model(8)
    // Total: 72 bytes on 64-bit (with alignment padding)
    final ptrSize = sizeOf<Pointer>();
    // Use a generous allocation and write at known offsets
    final paramsSize = 16 + 6 * ptrSize + 4 + (ptrSize - 4) + ptrSize;
    final params = calloc<Uint8>(paramsSize);

    // Ints at byte 0, 4, 8
    params.cast<Int32>()[0] = router ? 1 : 0;
    params.cast<Int32>()[1] = cleanup ? 1 : 0;
    params.cast<Int32>()[2] = minChars;
    // Float at byte 12
    params.elementAt(12).cast<Float>()[0] = minConfidence;
    // Pointers at byte 16 (aligned)
    final ptrBase = params.elementAt(16).cast<Pointer<Utf8>>();
    final detPtr = detModelPath.toNativeUtf8();
    final recPtr = recModelPath.toNativeUtf8();
    final nafPtr = nafnetModel != null ? nafnetModel.toNativeUtf8() : nullptr.cast<Utf8>();
    final srPtr = srModel != null ? srModel.toNativeUtf8() : nullptr.cast<Utf8>();
    final vlmPtr = vlmModel != null ? vlmModel.toNativeUtf8() : nullptr.cast<Utf8>();
    ptrBase[0] = detPtr;      // det_model
    ptrBase[1] = recPtr;      // rec_model
    ptrBase[2] = nafPtr;      // nafnet_model
    ptrBase[3] = srPtr;       // sr_model
    ptrBase[4] = vlmPtr;      // vlm_model
    // vlm_engine at byte 16 + 5*ptrSize
    final vlmEngOff = 16 + 5 * ptrSize;
    params.elementAt(vlmEngOff).cast<Int32>()[0] = vlmEngine;
    // punct_model pointer at next aligned offset
    final punctOff = vlmEngOff + ptrSize; // aligned to pointer size
    final punctPtr = punctModel != null ? punctModel.toNativeUtf8() : nullptr.cast<Utf8>();
    params.elementAt(punctOff).cast<Pointer<Utf8>>()[0] = punctPtr;

    _ctx = initFn(params.cast<Void>(), nThreads);

    calloc.free(detPtr);
    calloc.free(recPtr);
    if (nafnetModel != null) calloc.free(nafPtr);
    if (srModel != null) calloc.free(srPtr);
    if (vlmModel != null) calloc.free(vlmPtr);
    if (punctModel != null) calloc.free(punctPtr);
    calloc.free(params);

    if (_ctx == nullptr) {
      throw Exception(
          'Failed to init OCR orchestrator: $detModelPath + $recModelPath');
    }
  }

  /// Run the full pipeline on an image. Returns text + metadata.
  ({String text, int nRegions, double meanConfidence}) run(String imagePath) {
    if (_disposed) return (text: '', nRegions: 0, meanConfidence: 0.0);
    final pathPtr = imagePath.toNativeUtf8();
    final outN = calloc<Int32>();
    final outText = calloc<Pointer<Utf8>>();
    final outConf = calloc<Float>();

    _runFn(_ctx, pathPtr, outN, outText, outConf);

    final n = outN.value;
    final text =
        outText.value != nullptr ? outText.value.toDartString() : '';
    final conf = outConf.value;

    calloc.free(pathPtr);
    calloc.free(outN);
    calloc.free(outText);
    calloc.free(outConf);

    return (text: text, nRegions: n, meanConfidence: conf);
  }

  void dispose() {
    if (!_disposed) {
      _freeFn(_ctx);
      _disposed = true;
    }
  }
}

// ---------------------------------------------------------------------------
// Classical Preprocessing (model-free, CPU-only)
// ---------------------------------------------------------------------------

/// Classical document preprocessing — dewarp, deskew, binarize, despeckle,
/// PDF DPI profiling. No model downloads needed. All operations are CPU-only.
class CrispPreprocess {
  late final DynamicLibrary _lib;
  late final CrispembedPdfPageDpiDart _pdfPageDpiFn;
  late final CrispembedDewarpDart _dewarpFn;
  late final CrispembedTpsAutoDewarpDart _tpsAutoDewarpFn;
  late final CrispembedFindSkewDart _findSkewFn;
  late final CrispembedAdaptiveBinarizeDart _binarizeFn;
  late final CrispembedBackgroundNormDart _bgNormFn;
  late final CrispembedDespeckleDart _despeckleFn;
  late final CrispembedCcDetectDart _ccDetectFn;

  CrispPreprocess({String? libPath}) {
    _lib = _openNativeLib(libPath);
    _pdfPageDpiFn = _lib.lookupFunction<CrispembedPdfPageDpiNative,
        CrispembedPdfPageDpiDart>('crispembed_pdf_page_dpi');
    _dewarpFn = _lib.lookupFunction<CrispembedDewarpNative,
        CrispembedDewarpDart>('crispembed_dewarp');
    _tpsAutoDewarpFn = _lib.lookupFunction<CrispembedTpsAutoDewarpNative,
        CrispembedTpsAutoDewarpDart>('crispembed_tps_auto_dewarp');
    _findSkewFn = _lib.lookupFunction<CrispembedFindSkewNative,
        CrispembedFindSkewDart>('crispembed_find_skew');
    _binarizeFn = _lib.lookupFunction<CrispembedAdaptiveBinarizeNative,
        CrispembedAdaptiveBinarizeDart>('crispembed_adaptive_binarize');
    _bgNormFn = _lib.lookupFunction<CrispembedBackgroundNormNative,
        CrispembedBackgroundNormDart>('crispembed_background_norm');
    _despeckleFn = _lib.lookupFunction<CrispembedDespeckleNative,
        CrispembedDespeckleDart>('crispembed_despeckle');
    _ccDetectFn = _lib.lookupFunction<CrispembedCcDetectNative,
        CrispembedCcDetectDart>('crispembed_cc_detect');
  }

  /// PDF DPI profiling -- analyse embedded images on a PDF page.
  /// Returns (dpi, nImages), or null on failure.
  ({double dpi, int nImages})? pdfPageDpi(String path, {int page = 0}) {
    final cPath = path.toNativeUtf8();
    final outDpi = calloc<Float>();
    final outN = calloc<Int32>();
    final ret = _pdfPageDpiFn(cPath, page, outDpi, outN);
    final result = ret == 0
        ? (dpi: outDpi.value.toDouble(), nImages: outN.value)
        : null;
    calloc.free(cPath);
    calloc.free(outDpi);
    calloc.free(outN);
    return result;
  }

  /// Dewarp a grayscale page image (straighten curved text lines).
  /// Returns dewarped image bytes, or null on failure.
  Uint8List? dewarp(Uint8List gray, int w, int h) {
    final inp = calloc<Uint8>(w * h);
    inp.asTypedList(w * h).setAll(0, gray);
    final out = calloc<Uint8>(w * h);
    final ow = calloc<Int32>();
    final oh = calloc<Int32>();
    final ret = _dewarpFn(inp, w, h, out, ow, oh);
    Uint8List? result;
    if (ret == 0) {
      result = Uint8List.fromList(out.asTypedList(ow.value * oh.value));
    }
    calloc.free(inp);
    calloc.free(out);
    calloc.free(ow);
    calloc.free(oh);
    return result;
  }

  /// TPS auto-dewarp using a learned localizer model (GGUF).
  /// Returns dewarped image bytes, or null on failure.
  Uint8List? tpsDewarp(Uint8List gray, int w, int h, String modelPath) {
    final inp = calloc<Uint8>(w * h);
    inp.asTypedList(w * h).setAll(0, gray);
    final out = calloc<Uint8>(w * h);
    final cPath = modelPath.toNativeUtf8();
    final ret = _tpsAutoDewarpFn(inp, w, h, cPath.cast<Utf8>(), out);
    Uint8List? result;
    if (ret == 0) {
      result = Uint8List.fromList(out.asTypedList(w * h));
    }
    calloc.free(inp);
    calloc.free(out);
    calloc.free(cPath);
    return result;
  }

  /// Find skew angle in degrees. Returns (angle, confidence).
  ({double angle, double confidence}) findSkew(Uint8List gray, int w, int h) {
    final inp = calloc<Uint8>(w * h);
    inp.asTypedList(w * h).setAll(0, gray);
    final angle = calloc<Float>();
    final conf = calloc<Float>();
    _findSkewFn(inp, w, h, angle, conf);
    final result = (angle: angle.value.toDouble(), confidence: conf.value.toDouble());
    calloc.free(inp);
    calloc.free(angle);
    calloc.free(conf);
    return result;
  }

  /// Adaptive Otsu binarization (handles uneven lighting).
  Uint8List adaptiveBinarize(Uint8List gray, int w, int h) {
    final inp = calloc<Uint8>(w * h);
    inp.asTypedList(w * h).setAll(0, gray);
    final out = calloc<Uint8>(w * h);
    _binarizeFn(inp, w, h, out);
    final result = Uint8List.fromList(out.asTypedList(w * h));
    calloc.free(inp);
    calloc.free(out);
    return result;
  }

  /// Normalize background (handle gradients/shadows).
  Uint8List backgroundNorm(Uint8List gray, int w, int h) {
    final inp = calloc<Uint8>(w * h);
    inp.asTypedList(w * h).setAll(0, gray);
    final out = calloc<Uint8>(w * h);
    _bgNormFn(inp, w, h, out);
    final result = Uint8List.fromList(out.asTypedList(w * h));
    calloc.free(inp);
    calloc.free(out);
    return result;
  }

  /// Remove small noise components from a binary image.
  Uint8List despeckle(Uint8List gray, int w, int h,
      {int maxW = 5, int maxH = 5}) {
    final inp = calloc<Uint8>(w * h);
    inp.asTypedList(w * h).setAll(0, gray);
    final out = calloc<Uint8>(w * h);
    _despeckleFn(inp, w, h, maxW, maxH, out);
    final result = Uint8List.fromList(out.asTypedList(w * h));
    calloc.free(inp);
    calloc.free(out);
    return result;
  }

  /// Detect text line regions using connected components (model-free).
  /// Returns list of bounding boxes {x, y, w, h}.
  List<({int x, int y, int w, int h})> ccDetect(Uint8List gray, int width, int height) {
    final inp = calloc<Uint8>(width * height);
    inp.asTypedList(width * height).setAll(0, gray);
    final outN = calloc<Int32>();
    final ptr = _ccDetectFn(inp, width, height, outN);
    final n = outN.value;
    calloc.free(inp);
    calloc.free(outN);
    if (ptr == nullptr || n <= 0) return [];
    // Parse crispembed_ocr_result structs (same layout as OcrResult)
    final structSize = sizeOf<Float>() * 5 + sizeOf<Pointer>() + sizeOf<Int32>();
    final results = <({int x, int y, int w, int h})>[];
    for (var i = 0; i < n; i++) {
      final base = ptr.cast<Uint8>().elementAt(i * structSize);
      final floats = base.cast<Float>();
      results.add((
        x: floats[0].toInt(),
        y: floats[1].toInt(),
        w: floats[2].toInt(),
        h: floats[3].toInt(),
      ));
    }
    calloc.free(ptr);
    return results;
  }
}

// ---------------------------------------------------------------------------
// Punctuation Restoration
// ---------------------------------------------------------------------------

/// Post-process OCR text with punctuation/spacing restoration.
/// Loads a FireRedPunc or PCS model from a GGUF file.
class CrispPunct {
  late final DynamicLibrary _lib;
  late final Pointer<Void> _ctx;
  late final CrispembedPunctFreeDart _freeFn;
  late final CrispembedPunctProcessDart _processFn;
  bool _disposed = false;

  CrispPunct(String modelPath, {int nThreads = 4, String? libPath}) {
    _lib = _openNativeLib(libPath);
    final initFn = _lib.lookupFunction<CrispembedPunctInitNative,
        CrispembedPunctInitDart>('crispembed_punct_init');
    _freeFn = _lib.lookupFunction<CrispembedPunctFreeNative,
        CrispembedPunctFreeDart>('crispembed_punct_free');
    _processFn = _lib.lookupFunction<CrispembedPunctProcessNative,
        CrispembedPunctProcessDart>('crispembed_punct_process');

    final pathPtr = modelPath.toNativeUtf8();
    _ctx = initFn(pathPtr, nThreads);
    calloc.free(pathPtr);
    if (_ctx == nullptr) {
      throw Exception('Failed to load punct model: $modelPath');
    }
  }

  /// Add punctuation/spacing to text. Returns processed text.
  String process(String text) {
    if (_disposed) return text;
    final textPtr = text.toNativeUtf8();
    final result = _processFn(_ctx, textPtr);
    calloc.free(textPtr);
    return result != nullptr ? result.toDartString() : text;
  }

  void dispose() {
    if (!_disposed) {
      _freeFn(_ctx);
      _disposed = true;
    }
  }
}

// ---------------------------------------------------------------------------
// Restormer image restoration (CVPR 2022)
// ---------------------------------------------------------------------------

/// Result from Restormer image restoration.
class RestormerResult {
  final Uint8List pixels;
  final int width;
  final int height;

  const RestormerResult({
    required this.pixels,
    required this.width,
    required this.height,
  });
}

/// Restormer image restoration model — denoise, deblur, or derain images.
class CrispRestormer {
  late final DynamicLibrary _lib;
  late final Pointer<Void> _ctx;
  bool _disposed = false;

  late final CrispembedRestormerFreeDart _freeFn;
  late final CrispembedRestormerProcessDart _processFn;
  late final CrispembedRestormerFreeImageDart _freeImageFn;

  CrispRestormer(String modelPath, {int nThreads = 0, String? libPath}) {
    _lib = _openNativeLib(libPath);

    final pathPtr = modelPath.toNativeUtf8();
    _ctx = _lib
        .lookupFunction<CrispembedRestormerInitNative,
            CrispembedRestormerInitDart>('crispembed_restormer_init')
        .call(pathPtr, nThreads);
    calloc.free(pathPtr);

    if (_ctx == nullptr) {
      throw StateError('Failed to load Restormer model: $modelPath');
    }

    _freeFn = _lib.lookupFunction<CrispembedRestormerFreeNative,
        CrispembedRestormerFreeDart>('crispembed_restormer_free');
    _processFn = _lib.lookupFunction<CrispembedRestormerProcessNative,
        CrispembedRestormerProcessDart>('crispembed_restormer_process');
    _freeImageFn = _lib.lookupFunction<CrispembedRestormerFreeImageNative,
        CrispembedRestormerFreeImageDart>('crispembed_restormer_free_image');
  }

  /// Restore an RGB image (denoise / deblur / derain).
  ///
  /// [pixels] must be RGB uint8 bytes of length [width] * [height] * 3.
  /// Returns a [RestormerResult] with the same dimensions as the input.
  RestormerResult process(
    Uint8List pixels,
    int width,
    int height, {
    int tileSize = 0,
    int tileOverlap = 0,
  }) {
    _checkDisposed();
    final pxNative = calloc<Uint8>(pixels.length);
    final outPxPtr = calloc<Pointer<Uint8>>();

    try {
      for (var i = 0; i < pixels.length; i++) {
        pxNative[i] = pixels[i];
      }

      final rc = _processFn(
        _ctx,
        pxNative,
        width,
        height,
        tileSize,
        tileOverlap,
        outPxPtr,
      );

      if (rc != 0 || outPxPtr.value == nullptr) {
        throw StateError('Restormer processing failed (rc=$rc)');
      }

      final outPtr = outPxPtr.value;
      final len = width * height * 3;
      final resultPixels = Uint8List(len);
      for (var i = 0; i < len; i++) {
        resultPixels[i] = outPtr[i];
      }
      _freeImageFn(outPtr);

      return RestormerResult(pixels: resultPixels, width: width, height: height);
    } finally {
      calloc.free(pxNative);
      calloc.free(outPxPtr);
    }
  }

  void dispose() {
    if (!_disposed) {
      _freeFn(_ctx);
      _disposed = true;
    }
  }

  void _checkDisposed() {
    if (_disposed) throw StateError('CrispRestormer has been disposed');
  }
}

// ---------------------------------------------------------------------------
// OCR Rendering (text, hOCR, ALTO, PDF)
// ---------------------------------------------------------------------------

/// Render OCR results to structured format (hOCR, ALTO, or plain text).
/// This is a one-shot function — no persistent state needed.
String? ocrRender(DynamicLibrary lib, List<OcrResult> results,
    int pageWidth, int pageHeight, String format) {
  if (results.isEmpty) return null;
  final renderFn = lib.lookupFunction<CrispembedOcrRenderNative,
      CrispembedOcrRenderDart>('crispembed_ocr_render');

  // Build the C result array. crispembed_ocr_result layout:
  // 5 floats (x,y,w,h,conf) + pointer (text) + int32 (text_len)
  final structSize = sizeOf<Float>() * 5 + sizeOf<Pointer>() + sizeOf<Int32>();
  final arr = calloc<Uint8>(results.length * structSize);
  final textPtrs = <Pointer<Utf8>>[];

  for (var i = 0; i < results.length; i++) {
    final r = results[i];
    final base = arr.elementAt(i * structSize);
    final floats = base.cast<Float>();
    floats[0] = r.x.toDouble();
    floats[1] = r.y.toDouble();
    floats[2] = r.w.toDouble();
    floats[3] = r.h.toDouble();
    floats[4] = r.confidence.toDouble();
    final textPtr = r.text.toNativeUtf8();
    textPtrs.add(textPtr);
    base.elementAt(sizeOf<Float>() * 5).cast<Pointer<Utf8>>()[0] = textPtr;
    base.elementAt(sizeOf<Float>() * 5 + sizeOf<Pointer>()).cast<Int32>()[0] =
        r.text.length;
  }

  final fmtPtr = format.toNativeUtf8();
  final ptr = renderFn(
      arr.cast<Void>(), results.length, pageWidth, pageHeight, fmtPtr);
  calloc.free(fmtPtr);
  for (final tp in textPtrs) calloc.free(tp);
  calloc.free(arr);

  if (ptr == nullptr) return null;
  final result = ptr.toDartString();
  calloc.free(ptr);
  return result;
}
