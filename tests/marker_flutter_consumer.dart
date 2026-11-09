// SPDX-License-Identifier: MIT
// Actual Flutter package consumer. The supervisor supplies an owned three-page
// PDF as assets/owned.pdf, a real PDFium library, and the native install bundle.
// This source is copied to lib/main.dart in a generated Linux Flutter app.
import 'dart:convert';
import 'dart:io';

import 'package:crispembed/crispembed.dart';
import 'package:crypto/crypto.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

const _texts = [
  'First owned real PDF page.',
  'Second owned real PDF page.',
  'Third owned real PDF page.',
];
const _labels = ['iv', '章-2', '📖-3'];
const _formats = ['markdown', 'html', 'json', 'chunks'];
const _noticeHashes = {
  'NOTICE.marker':
      'f4bb1277697be95bac2ded2126f1a89e13c9c1b12db71d9549f3f00fe9c05d50',
  'NOTICE.teleocr':
      '839a413b743d1aa53e415774982065c1c195db1c96d82265f76dd5395970af1b',
  'Apache-2.0.txt':
      'cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30',
};

String _required(String name) {
  final value = Platform.environment[name];
  if (value == null || value.isEmpty) throw StateError('$name is required');
  return value;
}

Map<String, String> _renders(CrispMarker marker) => {
  for (final format in _formats) format: marker.render(format),
};

bool _sameRenders(Map<String, String> a, Map<String, String> b) =>
    _formats.every((format) => a[format] == b[format]);

String _comparable(String value, String format) {
  if (format != 'json' && format != 'chunks') return value;
  final decoded = jsonDecode(value) as Map<String, dynamic>;
  final configuration =
      decoded['metadata']['configuration'] as Map<String, dynamic>;
  for (final key in [
    'elapsed_seconds',
    'completed_pages',
    'next_page',
    'extraction_phase',
    'extraction_state',
  ]) {
    configuration.remove(key);
  }
  return jsonEncode(decoded);
}

bool _exactSourceRanges(Map<String, dynamic> chunks) {
  final records = {
    for (final dynamic source in chunks['source_records'] as List<dynamic>)
      source['id'] as String: source as Map<String, dynamic>,
  };
  final assembled = chunks['assembled_chunks'] as List<dynamic>;
  if (records.length != 3 || assembled.length != 3) return false;
  for (var page = 0; page < 3; page++) {
    final dynamic chunk = assembled[page];
    if (chunk['text'] != _texts[page] ||
        !listEquals(chunk['pages'] as List<dynamic>, [page])) {
      return false;
    }
    final bytes = utf8.encode(chunk['text'] as String);
    var cursor = 0;
    for (final dynamic range in chunk['source_ranges'] as List<dynamic>) {
      final source = records[range['source_id']];
      if (source == null || source['page'] != page) return false;
      final original = utf8.encode(source['text'] as String);
      final start = range['text_start'] as int, end = range['text_end'] as int;
      final sourceStart = range['source_start'] as int,
          sourceEnd = range['source_end'] as int;
      if (range['transform'] != 'copy' ||
          start != cursor ||
          start > end ||
          end > bytes.length ||
          sourceStart < 0 ||
          sourceStart > sourceEnd ||
          sourceEnd > original.length ||
          !listEquals(
            bytes.sublist(start, end),
            original.sublist(sourceStart, sourceEnd),
          )) {
        return false;
      }
      cursor = end;
    }
    if (cursor != bytes.length) return false;
  }
  return true;
}

Future<Map<String, dynamic>> runMarkerFlutterProbe({
  void Function(String)? progress,
}) async {
  final rows = <Map<String, dynamic>>[];
  void check(bool value, String name) {
    rows.add({'name': name, 'passed': value});
    if (!value) throw StateError(name);
    progress?.call(name);
  }

  final licenseAssets = <String, String>{};
  for (final entry in _noticeHashes.entries) {
    final data = await rootBundle.load('packages/crispembed/${entry.key}');
    final bytes = data.buffer.asUint8List(
      data.offsetInBytes,
      data.lengthInBytes,
    );
    final digest = sha256.convert(bytes).toString();
    licenseAssets[entry.key] = digest;
    check(
      digest == entry.value,
      'installed package asset ${entry.key} has exact canonical bytes',
    );
  }

  for (final key in [
    'MARKER',
    'MARKER_ARTIFACTS',
    'MARKER_INCREMENTAL',
    'MARKER_SOURCE_LINKS',
    'MARKER_REFERENCES',
  ]) {
    check(
      Platform.environment['CRISPEMBED_$key'] == '1',
      '$key test gate is explicitly enabled',
    );
  }
  check(
    Platform.environment['CRISPEMBED_TELEOCR'] == '0',
    'model inference is disabled',
  );
  final directory = Directory(_required('MARKER_FLUTTER_WORK')).absolute;
  await directory.create(recursive: true);
  final nativePath = _required('MARKER_FLUTTER_NATIVE_LIBRARY');
  final pdfiumPath = _required('MARKER_FLUTTER_PDFIUM_LIBRARY');
  final pdf = File('${directory.path}/owned.pdf');
  final asset = await rootBundle.load('assets/owned.pdf');
  await pdf.writeAsBytes(
    asset.buffer.asUint8List(asset.offsetInBytes, asset.lengthInBytes),
    flush: true,
  );
  final raw = File('${directory.path}/resume.cemk');
  final view = File('${directory.path}/processed.cemk');

  CrispMarker marker({String? provider, int dpi = 144}) => CrispMarker(
    mode: MarkerMode.text,
    libPath: nativePath,
    pdfiumPath: provider ?? pdfiumPath,
    nThreads: 1,
    extractImages: false,
    dpi: dpi,
  );

  late Map<String, String> legacy;
  final first = marker();
  try {
    check(
      first.progress['state'] == 'idle',
      'public package creates a native idle context',
    );
    first.convert(pdf.path);
    legacy = _renders(first);
    for (final text in _texts) {
      check(
        legacy['markdown']!.contains(text),
        'legacy conversion retains $text',
      );
    }
    first.start(pdf.path, raw.path);
    check(
      raw.existsSync() && first.progress['completed_pages'] == 0,
      'start creates a durable empty checkpoint',
    );
    check(
      first.step(1) == MarkerJobStatus.running &&
          first.progress['completed_pages'] == 1,
      'one-page budget commits exactly one page',
    );
    check(
      first.render().contains(_texts.first) &&
          !first.render().contains(_texts[1]),
      'partial view contains only completed real PDF text',
    );
    final committed = await raw.readAsBytes();
    first.cancel();
    check(
      first.step(64) == MarkerJobStatus.cancelled &&
          first.progress['cancel_requested'] == true,
      'cancel reaches the native cancelled state',
    );
    check(
      listEquals(await raw.readAsBytes(), committed),
      'cancel preserves the durable checkpoint bytes',
    );
  } finally {
    first.dispose();
  }
  await Future<void>.delayed(Duration.zero);

  late Map<String, String> complete;
  final resumed = marker();
  try {
    resumed.start(pdf.path, raw.path, resume: true);
    check(
      resumed.progress['completed_pages'] == 1 &&
          resumed.progress['cancel_requested'] == false,
      'fresh native context resumes the committed page',
    );
    check(
      resumed.step(1) == MarkerJobStatus.running &&
          resumed.progress['completed_pages'] == 2,
      'resumed step respects the one-page budget',
    );
    await Future<void>.delayed(Duration.zero);
    check(
      resumed.step(64) == MarkerJobStatus.complete &&
          resumed.progress['completed_pages'] == 3,
      'resumed extraction completes three real PDF pages',
    );
    complete = _renders(resumed);
    for (final format in _formats) {
      check(
        _comparable(complete[format]!, format) ==
            _comparable(legacy[format]!, format),
        'incremental $format equals legacy except documented run metadata',
      );
    }
    final document = jsonDecode(complete['json']!) as Map<String, dynamic>;
    final pages = document['children'] as List<dynamic>;
    check(
      listEquals(
        pages.map((dynamic page) => page['physical_page_index']).toList(),
        [0, 1, 2],
      ),
      'physical page indices survive the Flutter FFI path',
    );
    check(
      listEquals(
        pages.map((dynamic page) => page['printed_page_label']).toList(),
        _labels,
      ),
      'BMP and astral printed labels survive the Flutter FFI path',
    );
    check(
      _exactSourceRanges(
        jsonDecode(complete['chunks']!) as Map<String, dynamic>,
      ),
      'retrieval chunks reconstruct exact UTF-8 source bytes',
    );
    final completedRaw = await raw.readAsBytes();
    check(
      resumed.step(1) == MarkerJobStatus.complete &&
          _sameRenders(_renders(resumed), complete) &&
          listEquals(await raw.readAsBytes(), completedRaw),
      'completed step preserves renders and checkpoint bytes',
    );
    resumed.saveExtraction(view.path);
    check(
      view.existsSync() && view.lengthSync() > 0,
      'processed extraction artifact is saved',
    );
  } finally {
    resumed.dispose();
  }

  final reopened = marker(
    provider: '${directory.path}/no-provider.so',
    dpi: 150,
  );
  try {
    reopened.openExtraction(view.path);
    check(
      _sameRenders(_renders(reopened), complete),
      'processed artifact reopens exactly without a usable PDFium path',
    );
    check(
      reopened.assets.isEmpty,
      'text-only consumer has no fabricated image assets',
    );
  } finally {
    reopened.dispose();
  }

  return {
    'schema': 'crispembed.marker-flutter-consumer/1',
    'client': 'Flutter',
    'passed': true,
    'passed_count': rows.length,
    'total_count': rows.length,
    'checks': rows,
    'package_license_assets': licenseAssets,
    'fixture_sha256': sha256.convert(await pdf.readAsBytes()).toString(),
    'render_sha256': {
      for (final entry in complete.entries)
        entry.key: sha256.convert(utf8.encode(entry.value)).toString(),
    },
    'model_weights_used': false,
    'general_document_quality_qualified': false,
  };
}

class MarkerProbeView extends StatelessWidget {
  const MarkerProbeView({super.key, required this.status});
  final ValueListenable<String> status;

  @override
  Widget build(BuildContext context) => MaterialApp(
    title: 'CrispEmbed Marker consumer',
    home: Scaffold(
      appBar: AppBar(title: const Text('CrispEmbed Marker consumer')),
      body: Center(
        child: ValueListenableBuilder<String>(
          valueListenable: status,
          builder: (context, value, child) =>
              Text(value, textAlign: TextAlign.center),
        ),
      ),
    ),
  );
}

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final status = ValueNotifier('Starting real PDF conversion');
  runApp(MarkerProbeView(status: status));
  await WidgetsBinding.instance.endOfFrame;
  Map<String, dynamic> result;
  try {
    result = await runMarkerFlutterProbe(
      progress: (value) => status.value = value,
    );
    status.value = 'PASS: ${result['passed_count']} checks';
    await WidgetsBinding.instance.endOfFrame;
    result['completion_frame_rendered'] = true;
  } catch (error, stack) {
    result = {
      'schema': 'crispembed.marker-flutter-consumer/1',
      'passed': false,
      'error': error.toString(),
      'stack': stack.toString(),
    };
  }
  await File(_required('MARKER_FLUTTER_REPORT'))
      .writeAsString('${jsonEncode(result)}\n', flush: true);
  print(jsonEncode(result));
  exit(result['passed'] == true ? 0 : 1);
}
