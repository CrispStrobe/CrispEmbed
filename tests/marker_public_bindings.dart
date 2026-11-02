import 'dart:convert';
import 'dart:typed_data';

import 'package:marker_binding_test/marker.dart';

void require(bool value) {
  if (!value) throw StateError('Marker Dart assertion failed');
}

void main(List<String> args) {
  if (args.length != 3)
    throw ArgumentError('native library, PDFium and owned PDF are required');
  final marker = CrispMarker(
    mode: MarkerMode.text,
    libPath: args[0],
    pdfiumPath: args[1],
  );
  var checks = 0;
  for (final format in ['markdown', 'html', 'json', 'chunks']) {
    require(
      marker
          .convert(args[2], format: format)
          .contains('Hello from CrispEmbed.'),
    );
    checks++;
  }
  require(jsonEncode(marker.metadata).contains('crispembed-marker-native'));
  checks++;
  require(marker.assets.isEmpty);
  checks++;
  try {
    marker.render('bad-format');
    throw Exception('missing failure');
  } on StateError {
    checks++;
  }
  require(marker.render().contains('Hello from CrispEmbed.'));
  checks++;
  try {
    marker.convertImage(Uint8List(2), 2, 2, 3);
    throw Exception('missing failure');
  } on ArgumentError {
    checks++;
  }
  try {
    marker.convert('bad\u0000path');
    throw Exception('missing failure');
  } on ArgumentError {
    checks++;
  }
  try {
    marker.convert('missing-marker-input.pdf');
    throw Exception('missing failure');
  } on StateError {
    checks++;
  }
  try {
    marker.render();
    throw Exception('stale output');
  } on StateError {
    checks++;
  }
  require(marker.convert(args[2]).contains('Hello from CrispEmbed.'));
  checks++;
  marker.dispose();
  marker.dispose();
  try {
    marker.render();
    throw Exception('use after dispose');
  } on StateError {
    checks++;
  }
  print(jsonEncode({'client': 'Dart', 'passed': checks, 'total': checks}));
}
