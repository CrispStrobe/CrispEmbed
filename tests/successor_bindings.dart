import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:successor_binding_probe/crispembed.dart';

void require(bool value, String message) {
  if (!value) throw StateError(message);
}
void valid(List<double> values) {
  require(values.length == 384 && values.every((v) => v.isFinite), 'Invalid384-vector');
  final norm = values.fold<double>(0, (sum, v) => sum + v * v);
  require((norm - 1).abs() < .002, 'Vector is not normalized');
}
bool equal(List<double> a, List<double> b) {
  final aa = Float32List.fromList(a).buffer.asUint32List();
  final bb = Float32List.fromList(b).buffer.asUint32List();
  return aa.length == bb.length && List.generate(aa.length, (i) => aa[i] == bb[i]).every((v) => v);
}
Future<void> main(List<String> args) async {
  require(args.length == 3, 'Expected cached model, requests and prebuilt library');
  require(Platform.environment['CRISPEMBED_OFFLINE'] == '1', 'Offline environment absent');
  final modelPath = File(args[0]).resolveSymbolicLinksSync();
  final library = File(args[2]).resolveSymbolicLinksSync();
  final texts = File(args[1]).readAsLinesSync();
  require(texts.length == 3, 'Wrong request coverage');
  require(CrispEmbed.resolveModel(modelPath, libPath: library, autoDownload: false) == modelPath,
      'Cached exact path was not reused');
  var rejected = false;
  try {
    CrispEmbed.resolveModel('${File(modelPath).parent.path}/successor-deliberately-absent.gguf',
        libPath: library, autoDownload: false);
  } catch (_) { rejected = true; }
  require(rejected, 'Missing offline path was accepted');
  final model = CrispEmbed(modelPath, nThreads: 2, libPath: library, autoDownload: false);
  try {
    require(model.prefix.isEmpty, 'Unexpected prefix');
    final scalar = texts.map((text) => model.encode(text)).toList();
    for (final vector in scalar) { valid(vector); }
    final repeat = model.encode(texts[0]);
    valid(repeat);
    require(equal(repeat, scalar[0]), 'Repeat changed output');
    final batch = model.encodeBatch([texts[2], texts[0], texts[1], texts[0]]);
    require(batch.length == 4, 'Batch count changed');
    for (final vector in batch) { valid(vector); }
    final maps = File('/proc/self/maps').readAsLinesSync();
    require(maps.any((line) => line.endsWith(library)), 'Requested prebuilt library is not mapped');
    stdout.writeln(jsonEncode({'scalar': scalar, 'repeat': [repeat], 'batch': batch,
      'contracts': {'dimensions': true, 'repeat_exact': true, 'offline_cached_path': true,
        'missing_offline_rejected': true, 'actual_prebuilt_mapped': true}}));
    await stdout.flush();
  } finally { model.dispose(); }
}
