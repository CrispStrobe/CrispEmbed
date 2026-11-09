// Actual Dart/native calls. Lifecycle mode uses the owned PDFium ABI fixture,
// not a PDF parser or model. Its gates are set by the supervising process.
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:marker_binding_test/marker.dart';

void require(bool value) {
  if (!value) throw StateError('Marker Dart assertion failed');
}

void legacy(List<String> args) {
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

class Checks {
  final rows = <Map<String, Object>>[];

  void check(bool condition, String name) {
    if (!condition) throw StateError(name);
    rows.add({'name': name, 'passed': true});
  }

  void rejects(
    void Function() action,
    String fragment,
    String name, {
    bool argument = false,
  }) {
    try {
      action();
    } catch (error) {
      check(
        (argument ? error is ArgumentError : error is StateError) &&
            error.toString().contains(fragment),
        '$name: $error',
      );
      return;
    }
    throw StateError('$name: operation unexpectedly succeeded');
  }

  Map<String, dynamic> progress(
    CrispMarker marker,
    String state,
    int completed,
    int next,
    String name,
  ) {
    final value = marker.progress;
    check(
      value['schema'] == 'crispembed.marker-progress/1' &&
          value['state'] == state &&
          value['completed_pages'] == completed &&
          value['next_page'] == next &&
          value['total_pages'] == (state == 'idle' ? 0 : 3),
      name,
    );
    return value;
  }

  void report(String scenario) {
    print(
      jsonEncode({
        'client': 'Dart',
        'suite': 'native-lifecycle',
        'scenario': scenario,
        'model_weights_used': false,
        'pdf_rendering_qualified': false,
        'checks': rows,
        'passed': rows.length,
        'total': rows.length,
      }),
    );
  }
}

bool sameBytes(List<int> left, List<int> right) {
  if (left.length != right.length) return false;
  for (var i = 0; i < left.length; i++) {
    if (left[i] != right[i]) return false;
  }
  return true;
}

List<String> renderAll(CrispMarker marker) =>
    ['markdown', 'html', 'json', 'chunks'].map(marker.render).toList();

bool sameRenders(List<String> left, List<String> right) =>
    left.length == right.length &&
    Iterable<int>.generate(left.length).every((i) => left[i] == right[i]);

bool sourceMapsMatch(Map<String, dynamic> chunks, int pages) {
  final sources = chunks['source_records'] as List<dynamic>;
  final assembled = chunks['assembled_chunks'] as List<dynamic>;
  if (sources.length != pages || assembled.length != pages) return false;
  for (final dynamic chunk in assembled) {
    final text = utf8.encode(chunk['text'] as String);
    var covered = 0;
    for (final dynamic range in chunk['source_ranges'] as List<dynamic>) {
      final matches = sources.where(
        (dynamic source) => source['id'] == range['source_id'],
      );
      if (matches.length != 1) return false;
      final original = utf8.encode(matches.single['text'] as String);
      final start = range['text_start'] as int, end = range['text_end'] as int;
      final sourceStart = range['source_start'] as int,
          sourceEnd = range['source_end'] as int;
      if (range['transform'] != 'copy' ||
          start != covered ||
          start < 0 ||
          start > end ||
          end > text.length ||
          sourceStart < 0 ||
          sourceStart > sourceEnd ||
          sourceEnd > original.length ||
          !sameBytes(
            text.sublist(start, end),
            original.sublist(sourceStart, sourceEnd),
          )) {
        return false;
      }
      covered = end;
    }
    if (covered != text.length) return false;
  }
  return true;
}

void lifecycle(String library, String pdfium, String scenario) {
  if (!['artifacts-off', 'incremental-off', 'enabled'].contains(scenario)) {
    throw ArgumentError('Unknown lifecycle scenario');
  }
  final expected = {
    'CRISPEMBED_MARKER': '1',
    'CRISPEMBED_MARKER_ARTIFACTS': scenario == 'artifacts-off' ? '0' : '1',
    'CRISPEMBED_MARKER_INCREMENTAL': scenario == 'incremental-off' ? '0' : '1',
    'CRISPEMBED_MARKER_SOURCE_LINKS': '1',
    'CRISPEMBED_MARKER_REFERENCES': '1',
    'CRISPEMBED_MARKER_MULTILINGUAL_NORMALIZE': '0',
    'CRISPEMBED_MARKER_REPETITION_CHECK': '0',
    'CRISPEMBED_TELEOCR': '0',
    'CRISPEMBED_ACCEPT_QWEN_RESEARCH_LICENSE': '0',
  };
  for (final entry in expected.entries) {
    if (Platform.environment[entry.key] != entry.value) {
      throw StateError(
        'Explicit probe gate ${entry.key} must be ${entry.value}',
      );
    }
  }
  final folder = Directory.systemTemp.createTempSync('marker-dart-lifecycle-');
  final contexts = <CrispMarker>[];
  CrispMarker create({bool standalone = false}) {
    final marker = CrispMarker(
      mode: MarkerMode.text,
      libPath: library,
      pdfiumPath: standalone ? '' : pdfium,
      dpi: standalone ? 150 : 144,
      extractImages: false,
    );
    contexts.add(marker);
    return marker;
  }

  File file(String name) => File('${folder.path}/$name');
  try {
    final source = file('owned.pdf')
      ..writeAsStringSync(
        '%PDF-1.7\nMARKER_LIFECYCLE_FIXTURE\nowned Dart client input\n',
      );
    final raw = file('raw.cemk'),
        partialView = file('partial.cemk'),
        view = file('complete.cemk');
    final marker = create(), checks = Checks();
    checks.progress(marker, 'idle', 0, 0, 'new native context is idle');
    if (scenario != 'enabled') {
      final output = marker.convert(source.path);
      checks.check(
        output.contains('Third owned page is complete.'),
        'gate-off keeps synchronous conversion usable',
      );
      final String gate;
      if (scenario == 'artifacts-off') {
        checks.rejects(
          () => marker.saveExtraction(view.path),
          'CRISPEMBED_MARKER_ARTIFACTS',
          'native artifact save gate',
        );
        checks.rejects(
          () => marker.openExtraction(view.path),
          'CRISPEMBED_MARKER_ARTIFACTS',
          'native artifact open gate',
        );
        checks.check(!view.existsSync(), 'closed artifact gate writes no file');
        gate = 'CRISPEMBED_MARKER_ARTIFACTS';
      } else {
        marker.saveExtraction(view.path);
        marker.openExtraction(view.path);
        checks.check(
          marker.render() == output,
          'artifacts work with incremental gate off',
        );
        gate = 'CRISPEMBED_MARKER_INCREMENTAL';
      }
      checks.rejects(
        () => marker.start(source.path, raw.path),
        gate,
        'native start gate reaches Dart',
      );
      checks.rejects(
        () => marker.step(),
        gate,
        'native step gate reaches Dart',
      );
      checks.check(
        !raw.existsSync() && marker.render() == output,
        'rejected lifecycle calls preserve view and create no checkpoint',
      );
      checks.report(scenario);
      return;
    }

    marker.start(source.path, raw.path);
    checks.progress(
      marker,
      'running',
      0,
      0,
      'start creates an empty three-page job',
    );
    final initial = raw.readAsBytesSync();
    checks.check(
      initial.length >= 8 &&
          sameBytes(initial.sublist(0, 8), utf8.encode('CRISPMK\u0000')),
      'start writes a native extraction checkpoint',
    );
    for (final budget in [0, 65]) {
      checks.rejects(
        () => marker.step(budget),
        'Page budget',
        'Dart rejects invalid page budget',
        argument: true,
      );
    }
    checks.check(
      sameBytes(raw.readAsBytesSync(), initial),
      'wrapper validation leaves checkpoint bytes unchanged',
    );
    checks.check(
      marker.step() == MarkerJobStatus.running,
      'one-page native step remains running',
    );
    checks.progress(
      marker,
      'running',
      1,
      1,
      'one-page step advances only one cursor',
    );
    final partial = renderAll(marker), committed = raw.readAsBytesSync();
    checks.check(
      !sameBytes(committed, initial) &&
          partial[0].contains('First owned page') &&
          !partial[0].contains('Second owned page'),
      'partial Dart view contains only the committed page',
    );
    marker.saveExtraction(partialView.path);
    final partialReader = create(standalone: true);
    partialReader.openExtraction(partialView.path);
    checks.check(
      sameRenders(renderAll(partialReader), partial),
      'partial processed artifact preserves all four renders',
    );
    checks.check(
      partialReader.metadata['configuration']['extraction_state'] == 'partial',
      'partial reopen retains extraction state',
    );
    partialReader.dispose();
    checks.rejects(
      () => marker.saveExtraction(raw.path),
      'separately',
      'processed save cannot overwrite raw checkpoint',
    );
    marker.cancel();
    final requested = checks.progress(
      marker,
      'running',
      1,
      1,
      'cancellation request preserves current cursor',
    );
    checks.check(
      requested['cancel_requested'] == true,
      'Dart cancellation reaches the native flag',
    );
    checks.check(
      marker.step(64) == MarkerJobStatus.cancelled,
      'next boundary observes cancellation',
    );
    checks.progress(
      marker,
      'cancelled',
      1,
      1,
      'cancelled progress retains completed units',
    );
    checks.check(
      marker.step() == MarkerJobStatus.cancelled &&
          sameBytes(raw.readAsBytesSync(), committed),
      'repeated cancelled steps do not rewrite the checkpoint',
    );
    checks.check(
      sameRenders(renderAll(marker), partial),
      'cancelled view stays inspectable',
    );
    marker.dispose();
    checks.rejects(
      () => marker.step(),
      'disposed',
      'disposed wrapper cannot call the native lifecycle',
    );

    final resumed = create();
    resumed.start(source.path, raw.path, resume: true);
    final progress = checks.progress(
      resumed,
      'running',
      1,
      1,
      'fresh context resumes committed cursor',
    );
    checks.check(
      progress['cancel_requested'] == false,
      'resume clears cancellation',
    );
    checks.check(
      resumed.step() == MarkerJobStatus.running,
      'resumed page budget remains bounded',
    );
    checks.progress(
      resumed,
      'running',
      2,
      2,
      'resumed step commits only the next page',
    );
    checks.check(
      resumed.step(64) == MarkerJobStatus.complete,
      'resumed job completes remaining page',
    );
    checks.progress(resumed, 'complete', 3, 3, 'completed progress is exact');
    final complete = renderAll(resumed);
    checks.check(
      [
        'First owned page',
        'Second owned page',
        'Third owned page',
      ].every((text) => text.allMatches(complete[0]).length == 1),
      'resume neither loses nor duplicates pages',
    );
    final document = jsonDecode(complete[2]) as Map<String, dynamic>;
    final pages = document['children'] as List<dynamic>;
    final labels = ['iv', '章-2', '📖-3'];
    checks.check(
      pages.length == 3 &&
          Iterable<int>.generate(3).every(
            (i) =>
                pages[i]['printed_page_label'] == labels[i] &&
                pages[i]['physical_page_index'] == i,
          ),
      'Dart preserves distinct physical indices and Unicode printed labels',
    );
    checks.check(
      sourceMapsMatch(jsonDecode(complete[3]) as Map<String, dynamic>, 3),
      'Dart chunk source ranges match exact immutable UTF-8 bytes',
    );
    checks.check(
      resumed.assets.isEmpty,
      'text-only lifecycle has no image assets',
    );
    final completeRaw = raw.readAsBytesSync();
    checks.check(
      resumed.step() == MarkerJobStatus.complete &&
          sameBytes(raw.readAsBytesSync(), completeRaw),
      'completed step is idempotent',
    );
    resumed.saveExtraction(view.path);
    resumed.dispose();

    final reader = create(standalone: true);
    reader.openExtraction(view.path);
    checks.check(
      sameRenders(renderAll(reader), complete),
      'completed reopen preserves all four renders and source maps',
    );
    checks.rejects(
      () => reader.openExtraction(raw.path),
      'matching processing settings',
      'raw checkpoint rejects changed reader configuration',
    );
    checks.check(
      sameRenders(renderAll(reader), complete),
      'failed raw open preserves prior processed view',
    );
    final corrupt = view.readAsBytesSync();
    corrupt[24] = corrupt[24] == 48 ? 49 : 48;
    final broken = file('corrupted.cemk')..writeAsBytesSync(corrupt);
    checks.rejects(
      () => reader.openExtraction(broken.path),
      'integrity mismatch',
      'native corruption error reaches Dart',
    );
    checks.check(
      sameRenders(renderAll(reader), complete),
      'corrupt open preserves the complete Dart view',
    );
    final finished = create();
    finished.start(source.path, raw.path, resume: true);
    checks.check(
      finished.step() == MarkerJobStatus.complete &&
          sameBytes(raw.readAsBytesSync(), completeRaw),
      'fresh-context completed resume performs no extra work',
    );
    checks.check(
      finished.render() == complete[0],
      'completed raw resume retains exact text',
    );

    final failing = file('failing.pdf')
      ..writeAsStringSync(
        '%PDF-1.7\nMARKER_LIFECYCLE_FIXTURE\nFAIL_SECOND_PAGE\n',
      );
    final failedRaw = file('failed.cemk'), failed = create();
    failed.start(failing.path, failedRaw.path);
    checks.rejects(
      () => failed.step(64),
      'page 1',
      'later page failure reaches Dart',
    );
    checks.progress(
      failed,
      'error',
      1,
      1,
      'native failure retains committed progress',
    );
    checks.check(
      failed.render() == partial[0],
      'failed job retains its complete first page',
    );
    final retained = failedRaw.readAsBytesSync();
    failed.dispose();
    final retry = create();
    retry.start(failing.path, failedRaw.path, resume: true);
    checks.rejects(
      () => retry.step(),
      'page 1',
      'failed checkpoint resumes at the uncompleted page',
    );
    checks.check(
      sameBytes(failedRaw.readAsBytesSync(), retained),
      'failed retry preserves durable checkpoint bytes',
    );
    checks.report(scenario);
  } finally {
    for (final marker in contexts.reversed) {
      marker.dispose();
    }
    folder.deleteSync(recursive: true);
  }
}

void main(List<String> args) {
  if (args.isNotEmpty && args[0] == '--lifecycle') {
    if (args.length != 4) {
      throw ArgumentError(
        '--lifecycle <native-library> <owned-PDFium-library> <artifacts-off|incremental-off|enabled>',
      );
    }
    lifecycle(args[1], args[2], args[3]);
  } else {
    legacy(args);
  }
}
