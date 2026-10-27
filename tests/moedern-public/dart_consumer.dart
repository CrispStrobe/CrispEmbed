import 'dart:convert';
import 'dart:io';
import 'package:moedern_audit/moedern_text.dart';

// Actual Dart consumer through the production wrapper; hex TSV protocol.
String text(String h) => h == '-' ? '' : utf8.decode([for (var i = 0; i < h.length; i += 2) int.parse(h.substring(i, i + 2), radix: 16)]);
void main(List<String> a) {
  final m = CrispMoEdern(a[0], libPath: a[1]);
  try {
    for (final line in File(a[2]).readAsLinesSync()) {
      final f = line.split('\t');
      final r = m.encode(text(f.length > 2 ? f[2] : '-'), f[1] == 'query' ? MoEdernRole.query : MoEdernRole.document);
      stdout.writeln(jsonEncode({'kind': 'case', 'id': f[0], ...r}));
    }
    for (final line in File(a[3]).readAsLinesSync()) {
      final f = line.split('\t');
      stdout.writeln(jsonEncode({'kind': 'score', 'id': f[0], 'score': m.score(text(f[1]), text(f[2]))}));
    }
    var rejected = false;
    try { m.encode('x\u0000y', MoEdernRole.query); } catch (_) { rejected = true; }
    if (!rejected) throw StateError('NUL accepted');
  } finally { m.dispose(); }
}
