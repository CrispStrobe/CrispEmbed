import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:decision_probe/crispembed.dart';

void main(List<String> args) {
  final model = CrispEmbed(args[0], nThreads: 2, libPath: args[4]);
  try {
    if (!model.hasDecision) throw StateError('Missing decision capability');
    final bytes = File(args[2]).readAsBytesSync();
    final data = ByteData.sublistView(bytes);
    final vectors = List.generate(bytes.length ~/ (768 * 4), (r) => List.generate(768, (c) => data.getFloat32((r * 768 + c) * 4, Endian.little)));
    print(jsonEncode(model.decisionScores(vectors[0], vectors[1], vectors.sublist(2), typeId: int.parse(args[3]))));
    final request = jsonDecode(File(args[1]).readAsStringSync()) as Map<String, dynamic>;
    final answer = model.decide(request['state'], request['questions'], workflow: request['workflow']);
    try {
      model.decide('x', <String, dynamic>{});
      throw StateError('Accepted invalid schema');
    } on StateError catch (e) {
      if (e.message == 'Accepted invalid schema') rethrow;
    }
    if (jsonEncode(model.decide(request['state'], request['questions'], workflow: request['workflow'])) != jsonEncode(answer)) throw StateError('Repeat changed answer');
    print(jsonEncode(answer));
  } finally {
    model.dispose();
  }
}
