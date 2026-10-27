import 'dart:io';
import 'package:systemone_audit/systemone.dart';

// The hosted worker copies the production wrapper byte-for-byte into this
// package; no Flutter engine or fake native implementation is involved.
void main(List<String> args) {
  if (Platform.environment['GITHUB_ACTIONS'] != 'true') {
    throw StateError('Hosted model audit only');
  }
  if (args.length != 3) throw ArgumentError('model.gguf library requests.tsv');
  final model = CrispSystemOne(args[0], libPath: args[1],
      autoDownload: false, offline: true, nThreads: 2);
  try {
    if (model.profile != 'official-pytorch-text-batch1-512-no-extra-temperature') {
      throw StateError('Profile differs');
    }
    for (final invalid in ['{', '{}', 'x\u0000y', '\ud800', '\udc00']) {
      var rejected = false;
      try { model.runJson(invalid); } catch (_) { rejected = true; }
      if (!rejected) throw StateError('Invalid request accepted');
    }
    final rows = File(args[2]).readAsLinesSync();
    if (rows.length != 4) throw StateError('Four frozen raw requests required');
    String? firstRequest, firstResult;
    for (final line in rows) {
      final index = line.indexOf('\t');
      if (index <= 0) throw StateError('Raw TSV record invalid');
      final request = line.substring(index + 1);
      final result = model.runJson(request);
      firstRequest ??= request;
      firstResult ??= result;
      stdout.writeln('${line.substring(0, index)}\t$result');
    }
    var rejected = false;
    try { model.runJson('{'); } catch (_) { rejected = true; }
    if (!rejected || model.runJson(firstRequest!) != firstResult) {
      throw StateError('Post-success rejection/recovery differs');
    }
  } finally { model.dispose(); }
}
