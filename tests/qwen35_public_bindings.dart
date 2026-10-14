import 'dart:ffi';
import 'dart:io';
import 'package:qwen35_probe/crispembed.dart';
import 'package:qwen35_probe/crispembed_bindings.dart';
void check(bool value, String label) { if (!value) throw StateError(label); }
void main(List<String> args) {
  print('ABI stage_size=${sizeOf<CrispembedOcrStageFFI>()} cleanup_size=${sizeOf<CrispembedScanCleanupParamsFFI>()} params_size=${sizeOf<CrispembedOcrPipelineParamsFFI>()}');
  final pixels=File(args[1]).readAsBytesSync();final w=int.parse(args[2]),h=int.parse(args[3]);
  final model=CrispOcrModel(args[0],nThreads:4,libPath:args[5]);
  try {
    model.setMaxTokens(12);check(model.maxTokens==12,'generation cap');
    check(model.recognizeRaw(pixels,w,h)=='Hello OCR  \n12345','RGB text');
    check(model.confidences.length==9 && (model.meanConfidence-0.995560692).abs()<1e-6,'confidence');
    check(model.setPrompt('x') && model.setPrompt(null),'prompt reset');
    model.setMaxTokens(2);check(model.recognizeRaw(pixels,w,h)=='Hello OCR','truncation');
    print('PASS Dart unified RGB, prompt/reset, cap and confidence');
  } finally {model.dispose();}
  final pipeline=CrispOcrOrchestrator.fromStages([CrispOcrStageSpec(engine:20,modelA:args[0],maxTokens:12)],libPath:args[5]);
  try {check(pipeline.run(args[4]).text=='Hello OCR  \n12345','stage text');print('PASS Dart orchestrator ID20');}
  finally {pipeline.dispose();}
}
