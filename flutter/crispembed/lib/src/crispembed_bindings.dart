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

// --- Audio encoding (BidirLM-Omni and similar) ---
typedef CrispembedHasAudioNative = Int32 Function(Pointer<CrispembedContext> ctx);
typedef CrispembedHasAudio = int Function(Pointer<CrispembedContext> ctx);

typedef CrispembedEncodeAudioNative = Pointer<Float> Function(
    Pointer<CrispembedContext> ctx,
    Pointer<Float> pcmSamples,
    Int32 nSamples,
    Pointer<Int32> outDim);
typedef CrispembedEncodeAudio = Pointer<Float> Function(
    Pointer<CrispembedContext> ctx,
    Pointer<Float> pcmSamples,
    int nSamples,
    Pointer<Int32> outDim);

// --- Image encoding (BidirLM-Omni vision tower) ---
typedef CrispembedHasVisionNative = Int32 Function(Pointer<CrispembedContext> ctx);
typedef CrispembedHasVision = int Function(Pointer<CrispembedContext> ctx);

typedef CrispembedEncodeImageNative = Pointer<Float> Function(
    Pointer<CrispembedContext> ctx,
    Pointer<Float> pixelPatches,
    Int32 nPatches,
    Pointer<Int32> gridThw,
    Int32 nImages,
    Pointer<Int32> outDim);
typedef CrispembedEncodeImage = Pointer<Float> Function(
    Pointer<CrispembedContext> ctx,
    Pointer<Float> pixelPatches,
    int nPatches,
    Pointer<Int32> gridThw,
    int nImages,
    Pointer<Int32> outDim);

typedef CrispembedEncodeImageRawNative = Pointer<Float> Function(
    Pointer<CrispembedContext> ctx,
    Pointer<Float> pixelPatches,
    Int32 nPatches,
    Pointer<Int32> gridThw,
    Int32 nImages,
    Pointer<Int32> outNMerged,
    Pointer<Int32> outDim,
    Pointer<Int32> outNDeepstack);
typedef CrispembedEncodeImageRaw = Pointer<Float> Function(
    Pointer<CrispembedContext> ctx,
    Pointer<Float> pixelPatches,
    int nPatches,
    Pointer<Int32> gridThw,
    int nImages,
    Pointer<Int32> outNMerged,
    Pointer<Int32> outDim,
    Pointer<Int32> outNDeepstack);

// --- Ctx prefix (GGUF metadata) ---
typedef CrispembedCtxQueryPrefixNative = Pointer<Utf8> Function(Pointer<Void>);
typedef CrispembedCtxQueryPrefixDart = Pointer<Utf8> Function(Pointer<Void>);

typedef CrispembedCtxPassagePrefixNative = Pointer<Utf8> Function(Pointer<Void>);
typedef CrispembedCtxPassagePrefixDart = Pointer<Utf8> Function(Pointer<Void>);

// --- Face detection & recognition ---
typedef CrispembedFaceInitNative = Pointer<Void> Function(Pointer<Utf8>, Int32);
typedef CrispembedFaceInitDart = Pointer<Void> Function(Pointer<Utf8>, int);

typedef CrispembedFaceDimNative = Int32 Function(Pointer<Void>);
typedef CrispembedFaceDimDart = int Function(Pointer<Void>);

typedef CrispembedFaceTypeNative = Pointer<Utf8> Function(Pointer<Void>);
typedef CrispembedFaceTypeDart = Pointer<Utf8> Function(Pointer<Void>);

typedef CrispembedDetectFacesNative = Pointer<Void> Function(
    Pointer<Void>, Pointer<Utf8>, Float, Int32, Pointer<Int32>);
typedef CrispembedDetectFacesDart = Pointer<Void> Function(
    Pointer<Void>, Pointer<Utf8>, double, int, Pointer<Int32>);

typedef CrispembedEncodeFaceNative = Pointer<Float> Function(
    Pointer<Void>, Pointer<Utf8>, Pointer<Float>, Pointer<Int32>);
typedef CrispembedEncodeFaceDart = Pointer<Float> Function(
    Pointer<Void>, Pointer<Utf8>, Pointer<Float>, Pointer<Int32>);

typedef CrispembedFacePipelineNative = Pointer<Void> Function(
    Pointer<Void>, Pointer<Void>, Pointer<Utf8>, Float, Int32, Pointer<Int32>);
typedef CrispembedFacePipelineDart = Pointer<Void> Function(
    Pointer<Void>, Pointer<Void>, Pointer<Utf8>, double, int, Pointer<Int32>);

typedef CrispembedFaceFreeNative = Void Function(Pointer<Void>);
typedef CrispembedFaceFreeDart = void Function(Pointer<Void>);

// --- Vision file-based encoding ---
typedef CrispembedEncodeImageFileNative = Pointer<Float> Function(
    Pointer<CrispembedContext> ctx, Pointer<Utf8> imagePath, Pointer<Int32> outDim);
typedef CrispembedEncodeImageFileDart = Pointer<Float> Function(
    Pointer<CrispembedContext> ctx, Pointer<Utf8> imagePath, Pointer<Int32> outDim);

typedef CrispembedEncodeTextWithImageFileNative = Pointer<Float> Function(
    Pointer<CrispembedContext> ctx,
    Pointer<Utf8> text,
    Pointer<Utf8> imagePath,
    Pointer<Int32> outDim);
typedef CrispembedEncodeTextWithImageFileDart = Pointer<Float> Function(
    Pointer<CrispembedContext> ctx,
    Pointer<Utf8> text,
    Pointer<Utf8> imagePath,
    Pointer<Int32> outDim);

// --- Math OCR (unified — auto-detects architecture) ---
typedef CrispembedMathOcrInitNative = Pointer<Void> Function(
    Pointer<Utf8> modelPath, Int32 nThreads);
typedef CrispembedMathOcrInitDart = Pointer<Void> Function(
    Pointer<Utf8> modelPath, int nThreads);

typedef CrispembedMathOcrRecognizeNative = Pointer<Utf8> Function(
    Pointer<Void> ctx,
    Pointer<Uint8> pixelBytes,
    Int32 width,
    Int32 height,
    Int32 channels,
    Pointer<Int32> outLen);
typedef CrispembedMathOcrRecognizeDart = Pointer<Utf8> Function(
    Pointer<Void> ctx,
    Pointer<Uint8> pixelBytes,
    int width,
    int height,
    int channels,
    Pointer<Int32> outLen);

typedef CrispembedMathOcrFreeNative = Void Function(Pointer<Void> ctx);
typedef CrispembedMathOcrFreeDart = void Function(Pointer<Void> ctx);

// --- Standalone ViT image embedding (SigLIP, CLIP) ---
typedef CrispembedVitInitNative = Pointer<Void> Function(
    Pointer<Utf8> modelPath, Int32 nThreads);
typedef CrispembedVitInitDart = Pointer<Void> Function(
    Pointer<Utf8> modelPath, int nThreads);

typedef CrispembedVitDimNative = Int32 Function(Pointer<Void> ctx);
typedef CrispembedVitDimDart = int Function(Pointer<Void> ctx);

typedef CrispembedVitEncodeFileNative = Pointer<Float> Function(
    Pointer<Void> ctx, Pointer<Utf8> imagePath, Pointer<Int32> outDim);
typedef CrispembedVitEncodeFileDart = Pointer<Float> Function(
    Pointer<Void> ctx, Pointer<Utf8> imagePath, Pointer<Int32> outDim);

typedef CrispembedVitFreeNative = Void Function(Pointer<Void> ctx);
typedef CrispembedVitFreeDart = void Function(Pointer<Void> ctx);

// --- General OCR Pipeline (text detection + recognition) ---
typedef CrispembedOcrInitNative = Pointer<Void> Function(
    Pointer<Utf8> detModelPath, Pointer<Utf8> recModelPath, Int32 nThreads);
typedef CrispembedOcrInitDart = Pointer<Void> Function(
    Pointer<Utf8> detModelPath, Pointer<Utf8> recModelPath, int nThreads);

typedef CrispembedOcrFreeNative = Void Function(Pointer<Void> ctx);
typedef CrispembedOcrFreeDart = void Function(Pointer<Void> ctx);

typedef CrispembedOcrRunNative = Pointer<Void> Function(
    Pointer<Void> ctx, Pointer<Utf8> imagePath, Pointer<Int32> outN);
typedef CrispembedOcrRunDart = Pointer<Void> Function(
    Pointer<Void> ctx, Pointer<Utf8> imagePath, Pointer<Int32> outN);

typedef CrispembedOcrRecognizeNative = Pointer<Utf8> Function(
    Pointer<Void> ctx, Pointer<Utf8> imagePath, Pointer<Int32> outLen);
typedef CrispembedOcrRecognizeDart = Pointer<Utf8> Function(
    Pointer<Void> ctx, Pointer<Utf8> imagePath, Pointer<Int32> outLen);

// --- Named Entity Recognition (GLiNER) ---
typedef CrispembedNerInitNative = Pointer<Void> Function(
    Pointer<Utf8> modelPath, Int32 nThreads);
typedef CrispembedNerInitDart = Pointer<Void> Function(
    Pointer<Utf8> modelPath, int nThreads);

typedef CrispembedNerFreeNative = Void Function(Pointer<Void> ctx);
typedef CrispembedNerFreeDart = void Function(Pointer<Void> ctx);

typedef CrispembedNerExtractNative = Int32 Function(
    Pointer<Void> ctx,
    Pointer<Utf8> text,
    Pointer<Pointer<Utf8>> labels,
    Int32 nLabels,
    Float threshold,
    Pointer<Pointer<Void>> outEntities);
typedef CrispembedNerExtractDart = int Function(
    Pointer<Void> ctx,
    Pointer<Utf8> text,
    Pointer<Pointer<Utf8>> labels,
    int nLabels,
    double threshold,
    Pointer<Pointer<Void>> outEntities);

// --- LiLT — Language-independent Layout Transformer ---
typedef CrispembedLiltInitNative = Pointer<Void> Function(
    Pointer<Utf8> modelPath, Int32 nThreads);
typedef CrispembedLiltInitDart = Pointer<Void> Function(
    Pointer<Utf8> modelPath, int nThreads);

typedef CrispembedLiltFreeNative = Void Function(Pointer<Void> ctx);
typedef CrispembedLiltFreeDart = void Function(Pointer<Void> ctx);

typedef CrispembedLiltClassifyNative = Pointer<Void> Function(
    Pointer<Void> ctx, Pointer<Int32> inputIds, Pointer<Int32> bbox,
    Int32 nTokens, Pointer<Int32> outN);
typedef CrispembedLiltClassifyDart = Pointer<Void> Function(
    Pointer<Void> ctx, Pointer<Int32> inputIds, Pointer<Int32> bbox,
    int nTokens, Pointer<Int32> outN);

typedef CrispembedLiltNumLabelsNative = Int32 Function(Pointer<Void> ctx);
typedef CrispembedLiltNumLabelsDart = int Function(Pointer<Void> ctx);

// --- Key Information Extraction (KIE) — OCR + NER pipeline ---
typedef CrispembedKieInitNative = Pointer<Void> Function(
    Pointer<Utf8> ocrDetModel, Pointer<Utf8> ocrRecModel,
    Pointer<Utf8> nerModel, Int32 nThreads);
typedef CrispembedKieInitDart = Pointer<Void> Function(
    Pointer<Utf8> ocrDetModel, Pointer<Utf8> ocrRecModel,
    Pointer<Utf8> nerModel, int nThreads);

typedef CrispembedKieFreeNative = Void Function(Pointer<Void> ctx);
typedef CrispembedKieFreeDart = void Function(Pointer<Void> ctx);

// crispembed_kie_extract returns crispembed_kie_result by value.
// Layout: Pointer fields, int n_fields, char* ocr_text, float ocr_confidence, int n_ocr_regions
typedef CrispembedKieExtractNative = CrispembedKieResultFFI Function(
    Pointer<Void> ctx, Pointer<Utf8> imagePath,
    Pointer<Pointer<Utf8>> labels, Int32 nLabels,
    Float threshold);
typedef CrispembedKieExtractDart = CrispembedKieResultFFI Function(
    Pointer<Void> ctx, Pointer<Utf8> imagePath,
    Pointer<Pointer<Utf8>> labels, int nLabels,
    double threshold);

/// FFI struct matching crispembed_kie_result (returned by value).
final class CrispembedKieResultFFI extends Struct {
  external Pointer<Void> fields;

  @Int32()
  external int nFields;

  external Pointer<Utf8> ocrText;

  @Float()
  external double ocrConfidence;

  @Int32()
  external int nOcrRegions;
}

// --- OCR Orchestrator (source-type routing + cleanup + accept-gate) ---
// Uses the simple init path: crispembed_ocr_pipeline_init(params*, n_threads)
// The params struct is built in Dart and passed as a pointer.
typedef CrispembedOcrPipelineInitNative = Pointer<Void> Function(
    Pointer<Void> params, Int32 nThreads);
typedef CrispembedOcrPipelineInitDart = Pointer<Void> Function(
    Pointer<Void> params, int nThreads);

typedef CrispembedOcrPipelineRunNative = Pointer<Void> Function(
    Pointer<Void> ctx, Pointer<Utf8> imagePath,
    Pointer<Int32> outN, Pointer<Pointer<Utf8>> outText, Pointer<Float> outConf);
typedef CrispembedOcrPipelineRunDart = Pointer<Void> Function(
    Pointer<Void> ctx, Pointer<Utf8> imagePath,
    Pointer<Int32> outN, Pointer<Pointer<Utf8>> outText, Pointer<Float> outConf);

typedef CrispembedOcrPipelineFreeNative = Void Function(Pointer<Void> ctx);
typedef CrispembedOcrPipelineFreeDart = void Function(Pointer<Void> ctx);

typedef CrispembedOcrPipelineDefaultsNative = Void Function(Pointer<Void> outParams);
typedef CrispembedOcrPipelineDefaultsDart = void Function(Pointer<Void> outParams);

// --- PDF DPI Profiling ---
typedef CrispembedPdfPageDpiNative = Int32 Function(
    Pointer<Utf8> pdfPath, Int32 page,
    Pointer<Float> outDpi, Pointer<Int32> outNImages);
typedef CrispembedPdfPageDpiDart = int Function(
    Pointer<Utf8> pdfPath, int page,
    Pointer<Float> outDpi, Pointer<Int32> outNImages);

// --- Classical Preprocessing ---
typedef CrispembedDewarpNative = Int32 Function(
    Pointer<Uint8> gray, Int32 w, Int32 h,
    Pointer<Uint8> out, Pointer<Int32> outW, Pointer<Int32> outH);
typedef CrispembedDewarpDart = int Function(
    Pointer<Uint8> gray, int w, int h,
    Pointer<Uint8> out, Pointer<Int32> outW, Pointer<Int32> outH);

typedef CrispembedTpsAutoDewarpNative = Int32 Function(
    Pointer<Uint8> gray, Int32 w, Int32 h,
    Pointer<Utf8> modelPath, Pointer<Uint8> out);
typedef CrispembedTpsAutoDewarpDart = int Function(
    Pointer<Uint8> gray, int w, int h,
    Pointer<Utf8> modelPath, Pointer<Uint8> out);

typedef CrispembedFindSkewNative = Int32 Function(
    Pointer<Uint8> gray, Int32 w, Int32 h,
    Pointer<Float> angle, Pointer<Float> confidence);
typedef CrispembedFindSkewDart = int Function(
    Pointer<Uint8> gray, int w, int h,
    Pointer<Float> angle, Pointer<Float> confidence);

typedef CrispembedAdaptiveBinarizeNative = Void Function(
    Pointer<Uint8> gray, Int32 w, Int32 h, Pointer<Uint8> out);
typedef CrispembedAdaptiveBinarizeDart = void Function(
    Pointer<Uint8> gray, int w, int h, Pointer<Uint8> out);

typedef CrispembedBackgroundNormNative = Void Function(
    Pointer<Uint8> gray, Int32 w, Int32 h, Pointer<Uint8> out);
typedef CrispembedBackgroundNormDart = void Function(
    Pointer<Uint8> gray, int w, int h, Pointer<Uint8> out);

typedef CrispembedDespeckleNative = Void Function(
    Pointer<Uint8> gray, Int32 w, Int32 h,
    Int32 maxW, Int32 maxH, Pointer<Uint8> out);
typedef CrispembedDespeckleDart = void Function(
    Pointer<Uint8> gray, int w, int h,
    int maxW, int maxH, Pointer<Uint8> out);

typedef CrispembedCcDetectNative = Pointer<Void> Function(
    Pointer<Uint8> gray, Int32 w, Int32 h, Pointer<Int32> outN);
typedef CrispembedCcDetectDart = Pointer<Void> Function(
    Pointer<Uint8> gray, int w, int h, Pointer<Int32> outN);

// --- OCR Rendering ---
typedef CrispembedOcrRenderNative = Pointer<Utf8> Function(
    Pointer<Void> results, Int32 nResults,
    Int32 pageWidth, Int32 pageHeight, Pointer<Utf8> format);
typedef CrispembedOcrRenderDart = Pointer<Utf8> Function(
    Pointer<Void> results, int nResults,
    int pageWidth, int pageHeight, Pointer<Utf8> format);

// --- Punctuation Restoration ---
typedef CrispembedPunctInitNative = Pointer<Void> Function(
    Pointer<Utf8> modelPath, Int32 nThreads);
typedef CrispembedPunctInitDart = Pointer<Void> Function(
    Pointer<Utf8> modelPath, int nThreads);

typedef CrispembedPunctFreeNative = Void Function(Pointer<Void> ctx);
typedef CrispembedPunctFreeDart = void Function(Pointer<Void> ctx);

typedef CrispembedPunctProcessNative = Pointer<Utf8> Function(
    Pointer<Void> ctx, Pointer<Utf8> text);
typedef CrispembedPunctProcessDart = Pointer<Utf8> Function(
    Pointer<Void> ctx, Pointer<Utf8> text);

// --- Text Super-Resolution (text_sr) ---
typedef CrispembedTextSrInitNative = Pointer<Void> Function(
    Pointer<Utf8> modelPath, Int32 nThreads);
typedef CrispembedTextSrInitDart = Pointer<Void> Function(
    Pointer<Utf8> modelPath, int nThreads);

typedef CrispembedTextSrFreeNative = Void Function(Pointer<Void> ctx);
typedef CrispembedTextSrFreeDart = void Function(Pointer<Void> ctx);

typedef CrispembedTextSrUpscaleFactorNative = Int32 Function(Pointer<Void> ctx);
typedef CrispembedTextSrUpscaleFactorDart = int Function(Pointer<Void> ctx);

typedef CrispembedTextSrProcessNative = Int32 Function(
    Pointer<Void> ctx,
    Pointer<Uint8> pixels,
    Int32 width,
    Int32 height,
    Int32 tileSize,
    Int32 tileOverlap,
    Pointer<Pointer<Uint8>> outPixels,
    Pointer<Int32> outWidth,
    Pointer<Int32> outHeight);
typedef CrispembedTextSrProcessDart = int Function(
    Pointer<Void> ctx,
    Pointer<Uint8> pixels,
    int width,
    int height,
    int tileSize,
    int tileOverlap,
    Pointer<Pointer<Uint8>> outPixels,
    Pointer<Int32> outWidth,
    Pointer<Int32> outHeight);

typedef CrispembedTextSrFreeImageNative = Void Function(Pointer<Uint8> pixels);
typedef CrispembedTextSrFreeImageDart = void Function(Pointer<Uint8> pixels);

// --- TBSRN text-line Super-Resolution (tbsrn_sr) ---
typedef CrispembedTbsrnSrInitNative = Pointer<Void> Function(
    Pointer<Utf8> modelPath, Int32 nThreads);
typedef CrispembedTbsrnSrInitDart = Pointer<Void> Function(
    Pointer<Utf8> modelPath, int nThreads);

typedef CrispembedTbsrnSrFreeNative = Void Function(Pointer<Void> ctx);
typedef CrispembedTbsrnSrFreeDart = void Function(Pointer<Void> ctx);

typedef CrispembedTbsrnSrProcessNative = Int32 Function(
    Pointer<Void> ctx,
    Pointer<Uint8> pixels,
    Int32 width,
    Int32 height,
    Pointer<Pointer<Uint8>> outPixels,
    Pointer<Int32> outWidth,
    Pointer<Int32> outHeight);
typedef CrispembedTbsrnSrProcessDart = int Function(
    Pointer<Void> ctx,
    Pointer<Uint8> pixels,
    int width,
    int height,
    Pointer<Pointer<Uint8>> outPixels,
    Pointer<Int32> outWidth,
    Pointer<Int32> outHeight);

typedef CrispembedTbsrnSrFreeImageNative = Void Function(Pointer<Uint8> pixels);
typedef CrispembedTbsrnSrFreeImageDart = void Function(Pointer<Uint8> pixels);

// --- PAN Super-Resolution (pan_sr) ---
typedef CrispembedPanSrInitNative = Pointer<Void> Function(
    Pointer<Utf8> modelPath, Int32 nThreads);
typedef CrispembedPanSrInitDart = Pointer<Void> Function(
    Pointer<Utf8> modelPath, int nThreads);

typedef CrispembedPanSrFreeNative = Void Function(Pointer<Void> ctx);
typedef CrispembedPanSrFreeDart = void Function(Pointer<Void> ctx);

typedef CrispembedPanSrScaleNative = Int32 Function(Pointer<Void> ctx);
typedef CrispembedPanSrScaleDart = int Function(Pointer<Void> ctx);

typedef CrispembedPanSrProcessNative = Int32 Function(
    Pointer<Void> ctx,
    Pointer<Uint8> pixels,
    Int32 width,
    Int32 height,
    Int32 tileSize,
    Int32 tileOverlap,
    Pointer<Pointer<Uint8>> outPixels,
    Pointer<Int32> outWidth,
    Pointer<Int32> outHeight);
typedef CrispembedPanSrProcessDart = int Function(
    Pointer<Void> ctx,
    Pointer<Uint8> pixels,
    int width,
    int height,
    int tileSize,
    int tileOverlap,
    Pointer<Pointer<Uint8>> outPixels,
    Pointer<Int32> outWidth,
    Pointer<Int32> outHeight);

typedef CrispembedPanSrFreeImageNative = Void Function(Pointer<Uint8> pixels);
typedef CrispembedPanSrFreeImageDart = void Function(Pointer<Uint8> pixels);

// --- Restormer image restoration ---
typedef CrispembedRestormerInitNative = Pointer<Void> Function(
    Pointer<Utf8> modelPath, Int32 nThreads);
typedef CrispembedRestormerInitDart = Pointer<Void> Function(
    Pointer<Utf8> modelPath, int nThreads);

typedef CrispembedRestormerFreeNative = Void Function(Pointer<Void> ctx);
typedef CrispembedRestormerFreeDart = void Function(Pointer<Void> ctx);

typedef CrispembedRestormerProcessNative = Int32 Function(
    Pointer<Void> ctx,
    Pointer<Uint8> pixels,
    Int32 width,
    Int32 height,
    Int32 tileSize,
    Int32 tileOverlap,
    Pointer<Pointer<Uint8>> outPixels);
typedef CrispembedRestormerProcessDart = int Function(
    Pointer<Void> ctx,
    Pointer<Uint8> pixels,
    int width,
    int height,
    int tileSize,
    int tileOverlap,
    Pointer<Pointer<Uint8>> outPixels);

typedef CrispembedRestormerFreeImageNative = Void Function(Pointer<Uint8> pixels);
typedef CrispembedRestormerFreeImageDart = void Function(Pointer<Uint8> pixels);

// --- TBSRN Super-Resolution (tbsrn_sr, always 2×) ---
typedef CrispembedTbsrnSrInitNative = Pointer<Void> Function(
    Pointer<Utf8> modelPath, Int32 nThreads);
typedef CrispembedTbsrnSrInitDart = Pointer<Void> Function(
    Pointer<Utf8> modelPath, int nThreads);

typedef CrispembedTbsrnSrFreeNative = Void Function(Pointer<Void> ctx);
typedef CrispembedTbsrnSrFreeDart = void Function(Pointer<Void> ctx);

typedef CrispembedTbsrnSrProcessNative = Int32 Function(
    Pointer<Void> ctx,
    Pointer<Uint8> input,
    Int32 width,
    Int32 height,
    Pointer<Pointer<Uint8>> outPixels,
    Pointer<Int32> outWidth,
    Pointer<Int32> outHeight);
typedef CrispembedTbsrnSrProcessDart = int Function(
    Pointer<Void> ctx,
    Pointer<Uint8> input,
    int width,
    int height,
// --- SAFMN Super-Resolution (safmn_sr) ---
typedef CrispembedSafmnSrInitNative = Pointer<Void> Function(
    Pointer<Utf8> modelPath, Int32 nThreads);
typedef CrispembedSafmnSrInitDart = Pointer<Void> Function(
    Pointer<Utf8> modelPath, int nThreads);

typedef CrispembedSafmnSrFreeNative = Void Function(Pointer<Void> ctx);
typedef CrispembedSafmnSrFreeDart = void Function(Pointer<Void> ctx);

typedef CrispembedSafmnSrScaleNative = Int32 Function(Pointer<Void> ctx);
typedef CrispembedSafmnSrScaleDart = int Function(Pointer<Void> ctx);

typedef CrispembedSafmnSrProcessNative = Int32 Function(
    Pointer<Void> ctx,
    Pointer<Uint8> pixels,
    Int32 width,
    Int32 height,
    Int32 tileSize,
    Int32 tileOverlap,
    Pointer<Pointer<Uint8>> outPixels,
    Pointer<Int32> outWidth,
    Pointer<Int32> outHeight);
typedef CrispembedSafmnSrProcessDart = int Function(
    Pointer<Void> ctx,
    Pointer<Uint8> pixels,
    int width,
    int height,
    int tileSize,
    int tileOverlap,
    Pointer<Pointer<Uint8>> outPixels,
    Pointer<Int32> outWidth,
    Pointer<Int32> outHeight);

typedef CrispembedTbsrnSrFreeImageNative = Void Function(Pointer<Uint8> pixels);
typedef CrispembedTbsrnSrFreeImageDart = void Function(Pointer<Uint8> pixels);
typedef CrispembedSafmnSrFreeImageNative = Void Function(Pointer<Uint8> pixels);
typedef CrispembedSafmnSrFreeImageDart = void Function(Pointer<Uint8> pixels);

// --- Table Structure Recognition (table_parse) ---
typedef CrispembedTableParseInitNative = Pointer<Void> Function(
    Pointer<Utf8> ocrModelPath, Int32 nThreads);
typedef CrispembedTableParseInitDart = Pointer<Void> Function(
    Pointer<Utf8> ocrModelPath, int nThreads);

typedef CrispembedTableParseFreeNative = Void Function(Pointer<Void> ctx);
typedef CrispembedTableParseFreeDart = void Function(Pointer<Void> ctx);

typedef CrispembedTableParseToHtmlNative = Pointer<Utf8> Function(
    Pointer<Void> ctx,
    Pointer<Uint8> gray,
    Int32 width,
    Int32 height);
typedef CrispembedTableParseToHtmlDart = Pointer<Utf8> Function(
    Pointer<Void> ctx,
    Pointer<Uint8> gray,
    int width,
    int height);

typedef CrispembedTableParseFreeStringNative = Void Function(Pointer<Utf8> str);
typedef CrispembedTableParseFreeStringDart = void Function(Pointer<Utf8> str);
// --- Real-ESRGAN Super-Resolution (esrgan_sr) ---
typedef CrispembedEsrganSrInitNative = Pointer<Void> Function(
    Pointer<Utf8> modelPath, Int32 nThreads);
typedef CrispembedEsrganSrInitDart = Pointer<Void> Function(
    Pointer<Utf8> modelPath, int nThreads);

typedef CrispembedEsrganSrFreeNative = Void Function(Pointer<Void> ctx);
typedef CrispembedEsrganSrFreeDart = void Function(Pointer<Void> ctx);

typedef CrispembedEsrganSrScaleNative = Int32 Function(Pointer<Void> ctx);
typedef CrispembedEsrganSrScaleDart = int Function(Pointer<Void> ctx);

typedef CrispembedEsrganSrProcessNative = Int32 Function(
    Pointer<Void> ctx,
    Pointer<Uint8> pixels,
    Int32 width,
    Int32 height,
    Int32 tileSize,
    Int32 tileOverlap,
    Pointer<Pointer<Uint8>> outPixels,
    Pointer<Int32> outWidth,
    Pointer<Int32> outHeight);
typedef CrispembedEsrganSrProcessDart = int Function(
    Pointer<Void> ctx,
    Pointer<Uint8> pixels,
    int width,
    int height,
    int tileSize,
    int tileOverlap,
    Pointer<Pointer<Uint8>> outPixels,
    Pointer<Int32> outWidth,
    Pointer<Int32> outHeight);

typedef CrispembedEsrganSrFreeImageNative = Void Function(Pointer<Uint8> pixels);
typedef CrispembedEsrganSrFreeImageDart = void Function(Pointer<Uint8> pixels);

// --- SwinIR-light Super-Resolution (swinir_sr) ---
typedef CrispembedSwinirSrInitNative = Pointer<Void> Function(
    Pointer<Utf8> modelPath, Int32 nThreads);
typedef CrispembedSwinirSrInitDart = Pointer<Void> Function(
    Pointer<Utf8> modelPath, int nThreads);

typedef CrispembedSwinirSrFreeNative = Void Function(Pointer<Void> ctx);
typedef CrispembedSwinirSrFreeDart = void Function(Pointer<Void> ctx);

typedef CrispembedSwinirSrScaleNative = Int32 Function(Pointer<Void> ctx);
typedef CrispembedSwinirSrScaleDart = int Function(Pointer<Void> ctx);

typedef CrispembedSwinirSrProcessNative = Int32 Function(
    Pointer<Void> ctx,
    Pointer<Uint8> pixels,
    Int32 width,
    Int32 height,
    Int32 tileSize,
    Int32 tileOverlap,
    Pointer<Pointer<Uint8>> outPixels,
    Pointer<Int32> outWidth,
    Pointer<Int32> outHeight);
typedef CrispembedSwinirSrProcessDart = int Function(
    Pointer<Void> ctx,
    Pointer<Uint8> pixels,
    int width,
    int height,
    int tileSize,
    int tileOverlap,
    Pointer<Pointer<Uint8>> outPixels,
    Pointer<Int32> outWidth,
    Pointer<Int32> outHeight);

typedef CrispembedSwinirSrFreeImageNative = Void Function(Pointer<Uint8> pixels);
typedef CrispembedSwinirSrFreeImageDart = void Function(Pointer<Uint8> pixels);
