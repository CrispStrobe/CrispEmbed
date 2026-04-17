/// CrispEmbed — lightweight text embedding inference via ggml.
///
/// Supports dense, sparse (BGE-M3/SPLADE), ColBERT multi-vector,
/// and cross-encoder reranking — all on-device with GPU acceleration.
library crispembed;

export 'src/crispembed_bindings.dart';
export 'src/crispembed.dart';
