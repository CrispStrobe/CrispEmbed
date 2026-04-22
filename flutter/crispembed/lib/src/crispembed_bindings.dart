/// Raw FFI bindings to libcrispembed.
///
/// These mirror the C API in src/crispembed.h exactly.
/// Prefer using the high-level [CrispEmbed] class instead.
import 'dart:ffi';

import 'package:ffi/ffi.dart';

// Opaque handle
typedef CrispembedContext = Void;

// --- Lifecycle ---
typedef CrispembedInitNative = Pointer<CrispembedContext> Function(
    Pointer<Utf8> modelPath, Int32 nThreads);
typedef CrispembedInit = Pointer<CrispembedContext> Function(
    Pointer<Utf8> modelPath, int nThreads);

typedef CrispembedFreeNative = Void Function(Pointer<CrispembedContext> ctx);
typedef CrispembedFree = void Function(Pointer<CrispembedContext> ctx);

// --- Configuration ---
typedef CrispembedSetDimNative = Void Function(
    Pointer<CrispembedContext> ctx, Int32 dim);
typedef CrispembedSetDim = void Function(
    Pointer<CrispembedContext> ctx, int dim);

typedef CrispembedSetPrefixNative = Void Function(
    Pointer<CrispembedContext> ctx, Pointer<Utf8> prefix);
typedef CrispembedSetPrefix = void Function(
    Pointer<CrispembedContext> ctx, Pointer<Utf8> prefix);

typedef CrispembedGetPrefixNative = Pointer<Utf8> Function(
    Pointer<CrispembedContext> ctx);
typedef CrispembedGetPrefix = Pointer<Utf8> Function(
    Pointer<CrispembedContext> ctx);

typedef CrispembedCacheDirNative = Pointer<Utf8> Function();
typedef CrispembedCacheDir = Pointer<Utf8> Function();

typedef CrispembedResolveModelNative = Pointer<Utf8> Function(
    Pointer<Utf8> arg, Int32 autoDownload);
typedef CrispembedResolveModel = Pointer<Utf8> Function(
    Pointer<Utf8> arg, int autoDownload);

typedef CrispembedNModelsNative = Int32 Function();
typedef CrispembedNModels = int Function();

typedef CrispembedModelStringNative = Pointer<Utf8> Function(Int32 index);
typedef CrispembedModelString = Pointer<Utf8> Function(int index);

// --- Dense encoding ---
typedef CrispembedEncodeNative = Pointer<Float> Function(
    Pointer<CrispembedContext> ctx,
    Pointer<Utf8> text,
    Pointer<Int32> outNDim);
typedef CrispembedEncode = Pointer<Float> Function(
    Pointer<CrispembedContext> ctx,
    Pointer<Utf8> text,
    Pointer<Int32> outNDim);

typedef CrispembedEncodeBatchNative = Pointer<Float> Function(
    Pointer<CrispembedContext> ctx,
    Pointer<Pointer<Utf8>> texts,
    Int32 nTexts,
    Pointer<Int32> outNDim);
typedef CrispembedEncodeBatch = Pointer<Float> Function(
    Pointer<CrispembedContext> ctx,
    Pointer<Pointer<Utf8>> texts,
    int nTexts,
    Pointer<Int32> outNDim);

// --- Capability queries ---
typedef CrispembedHasSparseNative = Int32 Function(
    Pointer<CrispembedContext> ctx);
typedef CrispembedHasSparse = int Function(Pointer<CrispembedContext> ctx);

typedef CrispembedHasColbertNative = Int32 Function(
    Pointer<CrispembedContext> ctx);
typedef CrispembedHasColbert = int Function(Pointer<CrispembedContext> ctx);

typedef CrispembedIsRerankerNative = Int32 Function(
    Pointer<CrispembedContext> ctx);
typedef CrispembedIsReranker = int Function(Pointer<CrispembedContext> ctx);

// --- Sparse encoding ---
typedef CrispembedEncodeSparseNative = Int32 Function(
    Pointer<CrispembedContext> ctx,
    Pointer<Utf8> text,
    Pointer<Pointer<Int32>> outIndices,
    Pointer<Pointer<Float>> outValues);
typedef CrispembedEncodeSparse = int Function(
    Pointer<CrispembedContext> ctx,
    Pointer<Utf8> text,
    Pointer<Pointer<Int32>> outIndices,
    Pointer<Pointer<Float>> outValues);

// --- ColBERT multi-vector encoding ---
typedef CrispembedEncodeMultivecNative = Pointer<Float> Function(
    Pointer<CrispembedContext> ctx,
    Pointer<Utf8> text,
    Pointer<Int32> outNTokens,
    Pointer<Int32> outDim);
typedef CrispembedEncodeMultivec = Pointer<Float> Function(
    Pointer<CrispembedContext> ctx,
    Pointer<Utf8> text,
    Pointer<Int32> outNTokens,
    Pointer<Int32> outDim);

// --- Reranker ---
typedef CrispembedRerankNative = Float Function(
    Pointer<CrispembedContext> ctx,
    Pointer<Utf8> query,
    Pointer<Utf8> document);
typedef CrispembedRerank = double Function(
    Pointer<CrispembedContext> ctx,
    Pointer<Utf8> query,
    Pointer<Utf8> document);
