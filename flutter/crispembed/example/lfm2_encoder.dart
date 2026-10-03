import 'package:crispembed/crispembed.dart';

// dart run example/lfm2_encoder.dart MODEL.gguf /path/to/libcrispembed.so
void main(List<String> args) {
  if (args.length != 2) throw ArgumentError('Provide model and library paths');
  final model = CrispEmbed(args[0], nThreads: 2, libPath: args[1]);
  try {
    if (!model.hasMaskedLm) throw StateError('Masked encoder API unavailable');
    final features = model.encodeTokens(
      'The capital of France is <|mask|>.',
      normalize: false,
    );
    if (features.tokenIds.length != 9 ||
        features.tokenIds[7] != 16 ||
        features.features[0].length != 1024) {
      throw StateError('Unexpected token IDs or feature dimensions');
    }
    final normalized = model.encodeTokens('The capital of France is <|mask|>.');
    if (normalized.features.length != features.features.length)
      throw StateError('Wrong normalized shape');
    final logits = model.maskedLogits('The capital of France is <|mask|>.');
    if (logits.positions.single != 7 || logits.logits.single.length != 65536)
      throw StateError('Wrong logit shape');
    final result = model.fillMask('The capital of France is [MASK].');
    final predictions =
        result.single['predictions'] as List<Map<String, Object>>;
    if (predictions.first['token_id'] != 5242 ||
        predictions.first['token'] != ' Paris') {
      throw StateError('Unexpected masked prediction');
    }
    if (predictions.first['logit'] != logits.logits.single[5242])
      throw StateError('Changed logit');
    final padded = model.tokenBytes(65535);
    if (padded.isNotEmpty)
      throw StateError('Unused padded token must decode empty');
    print(predictions.first);
    print(
      'PASS: Dart raw/normalized features, token IDs, masked logits, decoded prediction and padding',
    );
  } finally {
    model.dispose();
  }
}
