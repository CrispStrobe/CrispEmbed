// Actual shipped Dart wrapper against an explicit immutable older library.
import 'dart:convert';
import 'dart:ffi';
import 'dart:io';

import 'package:marker_legacy_test/marker.dart';

void main(List<String> args) {
  if (args.length != 3) {
    throw ArgumentError(
      'old native library, real PDFium and owned PDF required',
    );
  }
  final rows = <Map<String, dynamic>>[];
  final errors = <Map<String, dynamic>>[];
  void check(bool value, String name) {
    rows.add({'name': name, 'passed': value});
    if (!value) throw StateError(name);
  }

  final native = DynamicLibrary.open(args[0]);
  const legacy = [
    'default_config',
    'init',
    'free',
    'convert',
    'convert_raw',
    'render',
    'metadata',
    'error',
    'asset_count',
    'asset_name',
    'asset_data',
  ];
  const extensions = [
    'save_extraction',
    'open_extraction',
    'start',
    'step',
    'cancel',
    'progress',
  ];
  check(
    legacy.every((name) => native.providesSymbol('crispembed_marker_$name')),
    'old release exports every legacy Marker symbol',
  );
  final missing = <String, bool>{};
  for (final name in extensions) {
    missing[name] = !native.providesSymbol('crispembed_marker_$name');
    check(missing[name]!, 'old release lacks $name');
  }

  final folder = Directory.systemTemp.createTempSync('marker-dart-old-');
  final target = File('${folder.path}/unavailable.cemk');
  final marker = CrispMarker(
    mode: MarkerMode.text,
    libPath: args[0],
    pdfiumPath: args[1],
    extractImages: false,
  );
  try {
    final initial = marker.convert(args[2]);
    const formats = ['markdown', 'html', 'json', 'chunks'];
    const required = [
      'Native document test',
      'Hello from CrispEmbed.',
      'PDF text stays searchable.',
    ];
    check(
      required.every(initial.contains),
      'current Dart wrapper converts the real owned PDF with v0.17.15',
    );
    check(
      formats.every((format) => required.every(marker.render(format).contains)),
      'all four legacy renderers contain the owned text',
    );
    check(
      marker.metadata['page_count'] == 1 && marker.assets.isEmpty,
      'legacy metadata and asset access remain usable',
    );
    final operations = <String, void Function()>{
      'save_extraction': () => marker.saveExtraction(target.path),
      'open_extraction': () => marker.openExtraction(target.path),
      'start': () => marker.start(args[2], target.path),
      'step': () => marker.step(),
      'cancel': marker.cancel,
      'progress': () => marker.progress,
    };
    for (final entry in operations.entries) {
      final before = [for (final format in formats) marker.render(format)];
      var rejected = false;
      try {
        entry.value();
      } on ArgumentError catch (error) {
        final message = error.toString();
        check(
          message.contains('Failed to lookup symbol') &&
              message.contains('crispembed_marker_${entry.key}'),
          '${entry.key} reports the unavailable native symbol',
        );
        errors.add({
          'method': entry.key,
          'error_type': 'ArgumentError',
          'missing_symbol': 'crispembed_marker_${entry.key}',
          'message': message,
        });
        rejected = true;
      }
      if (!rejected)
        throw StateError(
          '${entry.key} unexpectedly succeeded on the older library',
        );
      check(
        !target.existsSync(),
        '${entry.key} creates no artifact or checkpoint',
      );
      check(
        Iterable<int>.generate(formats.length)
            .every((i) => marker.render(formats[i]) == before[i]),
        '${entry.key} failure preserves all four existing renders',
      );
      check(
        marker.convert(args[2]) == initial,
        '${entry.key} failure permits later exact legacy conversion',
      );
    }
    print(
      jsonEncode({
        'client': 'Dart',
        'suite': 'actual-v0.17.15-compatibility',
        'passed': true,
        'checks': rows,
        'total': rows.length,
        'missing_symbols': missing,
        'expected_unavailable_errors': errors,
        'model_weights_used': false,
        'real_pdfium_used': true,
      }),
    );
  } finally {
    marker.dispose();
    folder.deleteSync(recursive: true);
  }
}
