import 'dart:convert';
import 'dart:io';
import 'package:ultrabert_probe/crispembed.dart';

void require(bool condition, String message) {
  if (!condition) throw StateError(message);
}
bool equal(List<num> a, List<num> b) =>
    a.length == b.length && List.generate(a.length, (i) => a[i] == b[i]).every((v) => v);
void rejected(void Function() call) {
  var failed = false;
  try { call(); } catch (_) { failed = true; }
  require(failed, 'Unsupported input accepted');
}
void main(List<String> args) {
  if (args[1] == '--reject') {
    rejected(() { final m = CrispEmbed(args[0], nThreads: 2, libPath: args[2], autoDownload: false); m.dispose(); });
    print(jsonEncode({'rejected': true}));
    return;
  }
  final request = jsonDecode(File(args[1]).readAsStringSync()) as List;
  final model = CrispEmbed(args[0], nThreads: 2, libPath: args[2], autoDownload: false);
  final other = CrispEmbed(args[0], nThreads: 2, libPath: args[2], autoDownload: false);
  try {
    final rows = <Map<String, Object>>[];
    for (final row in request) {
      final text = row['text'] as String, prefix = row['prefix'] as String;
      model.setPrefix(prefix);
      require(model.prefix == prefix, 'Prefix changed');
      final dense = model.encode(text);
      final raw = model.encodeTokens(text, normalize: false);
      final normalized = model.encodeTokens(text);
      require(raw.features.length == raw.tokenIds.length, 'Raw row count');
      require(equal(raw.tokenIds, normalized.tokenIds), 'Normalized IDs changed');
      require(equal(model.encode(text), dense), 'Repeat differs');
      final batch = model.encodeBatch([text, text]);
      require(batch.length == 2 && batch.every((v) => equal(v, dense)), 'Batch differs');
      other.setPrefix('');
      require(equal(other.encode(prefix + text), dense), 'Context/prefix differs');
      rejected(() { model.encode('<|mask|>'); });
      rejected(() { model.encodeTokens('<|mask|>', normalize: false); });
      rejected(() { model.encodeBatch([text, '<|mask|>']); });
      rejected(() { model.encode(List.filled(513, ' x').join()); });
      rejected(() { model.encode(List.filled(65537, 'x').join()); });
      require(equal(model.encode(text), dense), 'Failure recovery differs');
      rows.add({'id': row['id'] as String, 'ids': raw.tokenIds,
        'raw_cls': raw.features[0], 'dense': dense, 'token_normalized_cls': normalized.features[0]});
    }
    model.setPrefix('');
    final texts = request.where((r) => r['prefix'] == '').map((r) => r['text'] as String).toList();
    final batch = model.encodeBatch(texts);
    require(batch.length == texts.length, 'Mixed batch count');
    for (var i = 0; i < texts.length; i++) {
      require(equal(batch[i], model.encode(texts[i])), 'Mixed batch ordering');
    }
    print(jsonEncode({'rows': rows, 'contracts': {'repeat': true, 'batch': true,
      'context_prefix': true, 'rejection_recovery': true}}));
  } finally { model.dispose(); other.dispose(); }
}
