import 'dart:convert';
import 'dart:ffi';
import 'dart:io';
import 'package:ovisocr3_probe/crispembed.dart';
import 'package:ovisocr3_probe/crispembed_bindings.dart';

void check(bool value, String label) { if (!value) throw StateError(label); }
void main(List<String> args) {
  check(args.length == 7, 'model raw.rgb width height image library expected-text');
  print('ABI stage_size=${sizeOf<CrispembedOcrStageFFI>()} cleanup_size=${sizeOf<CrispembedScanCleanupParamsFFI>()} params_size=${sizeOf<CrispembedOcrPipelineParamsFFI>()}');
  final pixels = File(args[1]).readAsBytesSync();
  final w = int.parse(args[2]), h = int.parse(args[3]);
  final model = CrispOcrModel(args[0], nThreads: 2, libPath: args[5]);
  try {
    model.setMaxTokens(12);
    check(model.maxTokens == 12, 'generation cap');
    check(!model.setPrompt('different instruction'), 'custom prompt rejection');
    check(model.setPrompt(null), 'default prompt restore');
    final text = model.recognizeRaw(pixels, w, h);
    check(text == args[6], 'RGB decoded text');
    final confidence = model.confidences;
    check(confidence.isNotEmpty && confidence.length <= 12, 'confidence count');
    check(confidence.every((p) => p.isFinite && p >= 0 && p <= 1), 'confidence range');
    check(model.meanConfidence.isFinite, 'mean confidence');
    print('PASS Dart unified RGB cap=12 prompt-rejection text_bytes=${utf8.encode(text).length} confidence_count=${confidence.length} mean=${model.meanConfidence}');
  } finally { model.dispose(); }
  final pipeline = CrispOcrOrchestrator.fromStages(
    [CrispOcrStageSpec(engine: 22, modelA: args[0], maxTokens: 12)], libPath: args[5]);
  try { check(pipeline.run(args[4]).text == args[6], 'stage 22 decoded text');
    print('PASS Dart orchestrator engine=22');
  } finally { pipeline.dispose(); }
}
