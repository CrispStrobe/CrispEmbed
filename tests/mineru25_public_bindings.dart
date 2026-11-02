// MinerU2.5-Pro Dart wrapper consumer (CrispMinerU) against a real GGUF.
// Interface consistency, not reference parity: the expected text and token IDs are
// the Python/C anchor from the same native engine. Built as a pure-Dart probe package
// (copied flutter/crispembed/lib/src) and run by tools/ci-heavy/ocr_wrappers.sh.
//
//   dart run mineru25_public_bindings.dart LIB model.gguf crop.rgb W H anchor.json max_tokens page.rgb PW PH
import 'dart:convert';
import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';

import 'package:mineru_probe/mineru.dart';

void main(List<String> args) {
  if (args.length != 10) {
    stderr.writeln('lib model crop.rgb width height anchor.json max-tokens page.rgb width height');
    exit(2);
  }
  final lib = DynamicLibrary.open(args[0]);
  final crop = File(args[2]).readAsBytesSync();
  final width = int.parse(args[3]), height = int.parse(args[4]);
  final anchor = jsonDecode(File(args[5]).readAsStringSync()) as Map<String, dynamic>;
  final maxTokens = int.parse(args[6]);
  final page = File(args[7]).readAsBytesSync();
  final pw = int.parse(args[8]), ph = int.parse(args[9]);

  final model = CrispMinerU(lib, args[1], nThreads: 2);
  var ok = true;
  try {
    final region = model.predictRgb(Uint8List.fromList(crop), width, height, '\nText Recognition:',
        maxTokens: maxTokens);
    final predictOk = region['text'] == anchor['text'] &&
        jsonEncode(region['token_ids']) == jsonEncode(anchor['token_ids']);
    print(jsonEncode({'surface': 'dart-predict', 'passed': predictOk, 'text': region['text']}));
    ok = ok && predictOk;

    var rejected = false;
    try {
      model.predictRgb(Uint8List.fromList(crop), width, height + 1, '\nText Recognition:', maxTokens: maxTokens);
    } on ArgumentError {
      rejected = true;
    }
    print(jsonEncode({'surface': 'dart-validation', 'passed': rejected}));
    ok = ok && rejected;

    final pageJson = model.extractNativeBasic(Uint8List.fromList(page), pw, ph);
    final blocks = (pageJson['blocks'] as List?)?.length ?? 0;
    final markdown = (pageJson['markdown'] as String?)?.length ?? 0;
    final pageOk = blocks > 0 && markdown > 0 && pageJson['capped'] == false;
    print(jsonEncode({'surface': 'dart-extract', 'passed': pageOk, 'blocks': blocks, 'markdown_bytes': markdown}));
    ok = ok && pageOk;
  } finally {
    model.dispose();
  }
  exit(ok ? 0 : 1);
}
