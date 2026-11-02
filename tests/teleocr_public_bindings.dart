import 'dart:convert';
import 'dart:ffi';
import 'dart:io';

import 'package:teleocr_probe/crispembed.dart';
import 'package:teleocr_probe/crispembed_bindings.dart';

void check(bool value, String label) {
  if (!value) throw StateError(label);
}

void main(List<String> args) {
  check(
    args.length == 8,
    'model raw.rgb width height image library expected-utf8-file max-tokens',
  );
  final pixels = File(args[1]).readAsBytesSync();
  final width = int.parse(args[2]), height = int.parse(args[3]);
  final expected = File(args[6]).readAsStringSync();
  final cap = int.parse(args[7]);
  check(
    cap > 0 && expected.isNotEmpty,
    'nonempty expected output and positive cap',
  );
  check(pixels.length == width * height * 3, 'RGB input dimensions');
  check(crispOcrEngineTeleocr == 24, 'append-only TeleOCR stage ID');

  final model = CrispOcrModel(args[0], nThreads: 2, libPath: args[5]);
  try {
    check(model.maxTokens == 4096, 'model-card token cap');
    final defaults = model.generationParams;
    check(defaults != null, 'generation getter');
    check(
      (defaults!['repetition_penalty']!.toDouble() - 1.05).abs() < 1e-6 &&
          defaults['presence_penalty'] == 0 &&
          defaults['frequency_penalty'] == 0 &&
          defaults['no_repeat_ngram'] == 0,
      'model-card penalties',
    );
    check(
      model.setGenerationParams(
        repetitionPenalty: 1,
        presencePenalty: 1,
        frequencyPenalty: .05,
        noRepeatNgram: 100,
      ),
      'set pipeline penalties',
    );
    check(
      model.generationParams!['no_repeat_ngram'] == 100,
      'generation roundtrip',
    );
    check(
      !model.setGenerationParams(repetitionPenalty: double.nan),
      'reject NaN',
    );
    check(
      model.generationParams!['no_repeat_ngram'] == 100,
      'invalid setter leaves state unchanged',
    );
    check(model.setGenerationParams(), 'restore model-card penalties');
    model.setMaxTokens(cap);
    check(model.maxTokens == cap, 'generation cap');
    check(model.setPrompt('Read the image.'), 'custom prompt');
    check(model.setPrompt(null), 'default prompt restore');
    final text = model.recognizeRaw(pixels, width, height);
    check(text == expected, 'RGB decoded text');
    final confidence = model.confidences;
    check(
      confidence.isNotEmpty && confidence.length <= cap,
      'confidence count',
    );
    check(
      confidence.every((p) => p.isFinite && p >= 0 && p <= 1),
      'confidence range',
    );
    check(model.meanConfidence.isFinite, 'mean confidence');
    print(
      jsonEncode({
        'surface': 'dart-model',
        'passed': true,
        'text': text,
        'token_count': confidence.length,
        'confidence': model.meanConfidence,
        'max_tokens': cap,
        'stage_size': sizeOf<CrispembedOcrStageFFI>(),
        'cleanup_size': sizeOf<CrispembedScanCleanupParamsFFI>(),
      }),
    );
  } finally {
    model.dispose();
  }

  final pipeline = CrispOcrOrchestrator.fromStages(
    [
      CrispOcrStageSpec(
        engine: crispOcrEngineTeleocr,
        modelA: args[0],
        maxTokens: cap,
      ),
    ],
    nThreads: 2,
    libPath: args[5],
  );
  try {
    final result = pipeline.run(args[4]);
    check(result.text == expected, 'stage 24 decoded text');
    check(result.nRegions == 1, 'one native region');
    print(
      jsonEncode({
        'surface': 'dart-stage',
        'passed': true,
        'engine': 24,
        'text': result.text,
        'regions': result.nRegions,
        'confidence': result.meanConfidence,
      }),
    );
  } finally {
    pipeline.dispose();
  }
}
