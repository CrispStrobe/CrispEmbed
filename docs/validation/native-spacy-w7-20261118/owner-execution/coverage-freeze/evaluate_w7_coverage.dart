// Explicit local experiment. Missing assets fail; existing receipts are kept.
import 'dart:convert';
import 'dart:io';

import 'package:crypto/crypto.dart';
import 'package:quds_office_engine/quds_office_engine.dart';
import 'package:quds_proofing_spacy/quds_proofing_spacy_grammar.dart';
import 'package:quds_proofing_spacy/quds_proofing_spacy_native.dart';

const nativeSha =
    'ea89842ebf22c08508aff1cfbf18d78a107118af320523009616ff62b6e42977';
const protocolPath = 'docs/validation/native-spacy-w7-extension-20261118/protocol.json';

Future<void> main(List<String> arguments) async {
  final options = <String, String>{};
  for (var i = 0; i < arguments.length; i += 2) {
    if (i + 1 >= arguments.length ||
        !arguments[i].startsWith('--') ||
        options.containsKey(arguments[i])) {
      throw ArgumentError('Expected unique --name value pairs.');
    }
    options[arguments[i]] = arguments[i + 1];
  }
  const required = {
    '--root',
    '--phase',
    '--panel',
    '--library',
    '--bundle',
    '--report',
  };
  require(
    required.every(options.containsKey) &&
        options.keys.every((key) => required.contains(key) || key == '--seal'),
    'Required: ${required.join(', ')}; --seal mandatory for qualification.',
  );
  final phase = options['--phase']!;
  require(phase == 'development' || phase == 'qualification', 'Invalid phase.');
  final root = Directory(options['--root']!).absolute.path;
  final output = File(options['--report']!);
  require(!output.existsSync(), 'Receipt already exists; use a fresh path.');
  final panelBytes = File(options['--panel']!).readAsBytesSync();
  final panel = jsonDecode(utf8.decode(panelBytes)) as Map<String, dynamic>;
  require(
    panel['schema'] == 'quds.spacy.grammar.panel.v1' && panel['split'] == phase,
    'Wrong panel schema/split.',
  );
  final protocol =
      jsonDecode(File('$root/$protocolPath').readAsStringSync())
          as Map<String, dynamic>;
  final protocolPanel = protocol['qualification_panel'] as Map<String, dynamic>;
  final panelSha = digest(panelBytes);
  final before = sourceHashes(root);
  Map<String, dynamic>? seal;
  if (phase == 'qualification') {
    require(options.containsKey('--seal'), 'Qualification requires a seal.');
    seal =
        jsonDecode(File(options['--seal']!).readAsStringSync())
            as Map<String, dynamic>;
    require(
      seal['schema'] == 'quds.spacy.grammar.candidate.v1' &&
          seal['qualification_panel_sha256'] == panelSha &&
          protocolPanel['sha256'] == panelSha &&
          seal['native_library_sha256'] == nativeSha &&
          equalMap(before, seal['source_sha256'] as Map<String, dynamic>),
      'Candidate, panel or source seal changed before qualification.',
    );
  }
  final baselineBlobs = verifyBaseline(root, protocol);
  final cases = (panel['cases'] as List).cast<Map<String, dynamic>>();
  final ids = <String>{};
  final inputs = <String>{};
  final counts = <String, int>{
    for (final kind in ['error', 'valid', 'guard', 'unsupported']) kind: 0,
  };
  for (final item in cases) {
    require(
      item['id'] is String &&
          ids.add(item['id'] as String) &&
          counts.containsKey(item['kind']) &&
          item['text'] is String &&
          item['expected'] is String &&
          item['locale'] is String &&
          item['start_offset'] == 23 &&
          inputs.add(jsonEncode([item['locale'], item['text']])),
      'Duplicate, unclassified or invalid case.',
    );
    counts[item['kind'] as String] = counts[item['kind']]! + 1;
  }
  if (phase == 'qualification') {
    require(
      cases.length == protocolPanel['total'] &&
          equalMap(counts, protocolPanel['counts'] as Map<String, dynamic>),
      'Qualification denominator changed.',
    );
  }
  final report = <String, Object?>{
    'schema': 'quds.spacy.grammar.evaluation.v1',
    'phase': phase,
    'scope':
        'Authored finite comparison, not population-level grammar quality.',
    'started_at': DateTime.now().toUtc().toIso8601String(),
    'panel_sha256': panelSha,
    'protocol_sha256': digest(File('$root/$protocolPath').readAsBytesSync()),
    'candidate_seal_sha256': options.containsKey('--seal')
        ? digest(File(options['--seal']!).readAsBytesSync())
        : null,
    'candidate_commit': seal?['candidate_commit'],
    'source_sha256_before': before,
    'baseline_git_blobs_verified': baselineBlobs,
    'native_library_sha256': nativeSha,
    'native_heldout_acceptance_executions': 0,
    'ci_endpoints_queried': false,
    'dart_version': Platform.version.split(' ').first,
    'expected_counts': counts,
    'completed': false,
    'passed': false,
  };
  final rows = <Map<String, Object?>>[];
  report['cases'] = rows;
  // Reserve the destination atomically before native loading. A second process
  // cannot overwrite either a running or completed attempt.
  output.parent.createSync(recursive: true);
  output.createSync(exclusive: true);
  writeReport(output, report);
  if (phase == 'qualification') {
    final claim = File('${options['--seal']!}.consumed.json');
    claim.createSync(exclusive: true);
    claim.writeAsStringSync(
      '${jsonEncode({'schema': 'quds.spacy.grammar.qualification.claim.v1', 'candidate_seal_sha256': report['candidate_seal_sha256'], 'panel_sha256': panelSha, 'started_at': report['started_at'], 'policy': 'This sealed qualification has been started; never overwrite or rerun it.'})}\n',
      flush: true,
    );
    report['qualification_claim_sha256'] = digest(claim.readAsBytesSync());
    writeReport(output, report);
  }
  SpacyTransport? transport;
  final ordinary = OfficeProofingService(
    providers: [GermanWritingRulesProvider()],
  );
  final planner = SpacyGermanGrammarPlanner();
  try {
    final load = Stopwatch()..start();
    transport = await startSpacyWorker(
      bundle: const SpacyBundle.germanF32(),
      backendFactory: SpacyNativeBackendFactory(
        bundleDirectory: options['--bundle']!,
        libraryPath: options['--library']!,
        librarySha256: nativeSha,
      ),
    );
    report['load_microseconds'] = load.elapsedMicroseconds;
    report['identity'] = transport.identity.toJson();
    for (final item in cases) {
      final row = <String, Object?>{
        'id': item['id'],
        'kind': item['kind'],
        'family': item['family'],
        'locale': item['locale'],
        'text': item['text'],
        'expected': item['expected'],
        'origin_utf16': item['start_offset'],
      };
      rows.add(row); // Retain completed and failing rows if execution stops.
      final request = requestFor(item);
      final watch = Stopwatch()..start();
      final baseline = await ordinary.check(request);
      final analysis = await transport.analyze(analysisFor(request));
      final plan = await planner.plan(
        request,
        analysis,
        personalWords: (item['personal_words'] as List? ?? const [])
            .cast<String>()
            .toSet(),
      );
      // Exercise the actual service validator before merging planner findings.
      final secondary = await OfficeProofingService(
        providers: [_Completed(plan.result)],
      ).check(request);
      final combined = mergeOfficeProofingReports(
        primary: baseline,
        secondary: secondary,
      );
      final baseArm = arm(baseline, item['expected'] as String);
      final modelArm = arm(secondary, item['expected'] as String);
      final combinedArm = arm(combined, item['expected'] as String);
      row.addAll({
        'analysis': analysis.toJson(),
        'decisions': plan.decisions.map((d) => d.toJson()).toList(),
        'baseline': baseArm,
        'model': modelArm,
        'combined': combinedArm,
        'model_proposals_suppressed_by_merge':
            secondary.findings.length -
            combined.findings
                .where((f) => f.providerId == plan.result.providerId)
                .length,
        'elapsed_microseconds': watch.elapsedMicroseconds,
      });
      final expectedUnsupported = item['kind'] == 'unsupported';
      row['operational_ok'] =
          !baseline.results.any(
            (r) =>
                r.status == OfficeProofingStatus.failed ||
                r.status == OfficeProofingStatus.unavailable,
          ) &&
          (expectedUnsupported
              ? analysis.status == LinguisticAnalysisStatus.unsupported &&
                    secondary.results.single.status ==
                        OfficeProofingStatus.unsupported
              : analysis.isSuccess && secondary.complete);
      if (item['kind'] == 'error' && combinedArm['exact'] == true) {
        final correctedRequest = requestFor({
          ...item,
          'text': combinedArm['output'],
        });
        final reanalysis = await transport.analyze(
          analysisFor(correctedRequest),
        );
        final replan = await planner.plan(correctedRequest, reanalysis);
        row['corrected_recheck'] = {
          'analysis_status': reanalysis.status.name,
          'status': replan.result.status.name,
          'findings': replan.result.findings.map(finding).toList(),
          'decisions': replan.decisions.map((d) => d.toJson()).toList(),
          'annotations': reanalysis.toJson(),
        };
        if (!reanalysis.isSuccess ||
            replan.result.status != OfficeProofingStatus.success) {
          row['operational_ok'] = false;
        }
      }
      // Save progress with the original source map. The path was admitted once
      // above and is never reused by a subsequent invocation.
      writeReport(output, report);
    }
    report['completed'] = rows.length == cases.length;
    final summary = summarize(rows);
    report['summary'] = summary;
    final limits = protocol['predeclared_gate'] as Map<String, dynamic>;
    final gates = <String, bool>{
      'minimum_incremental_exact':
          summary['additional_exact_errors']! >=
          (limits['minimum_additional_exact_error_cases_over_baseline'] as int),
      'no_baseline_exact_loss': summary['baseline_exact_lost'] == 0,
      'no_native_valid_findings': summary['model_valid_flagged'] == 0,
      'no_native_guard_findings': summary['model_guard_flagged'] == 0,
      'no_added_changed_valid': summary['added_changed_valid'] == 0,
      'no_operational_failures': summary['operational_failures'] == 0,
      'clean_corrected_recheck': summary['recheck_model_findings'] == 0,
    };
    report['linguistic_gates'] = gates;
    report['linguistic_gate_passed'] = gates.values.every((v) => v);
    // Editor qualification is recorded separately; this result cannot alone
    // enable a provider or imply that the full W6 exit has passed.
    report['editor_gate_included'] = false;
    report['execution_passed'] =
        report['completed'] == true && summary['operational_failures'] == 0;
    report['passed'] =
        report['execution_passed'] == true &&
        report['linguistic_gate_passed'] == true;
  } on Object catch (error) {
    report['failure'] = error
        .toString()
        .replaceAll(root, '<QUDS_ROOT>')
        .replaceAll(Directory(options['--bundle']!).absolute.path, '<BUNDLE>')
        .replaceAll(
          File(options['--library']!).absolute.path,
          '<NATIVE_LIBRARY>',
        );
    report['summary'] = summarize(rows);
    rethrow;
  } finally {
    try {
      await transport?.dispose();
    } on Object catch (error) {
      report['cleanup_failure'] = error.runtimeType.toString();
      report['passed'] = false;
    }
    await ordinary.dispose();
    final after = sourceHashes(root);
    report['source_sha256_after'] = after;
    report['source_unchanged'] = equalMap(before, after);
    if (report['source_unchanged'] != true) report['passed'] = false;
    report['decision_action_counts'] = decisionCounts(rows, 'action');
    report['decision_reason_counts'] = decisionCounts(rows, 'reason');
    report['finished_at'] = DateTime.now().toUtc().toIso8601String();
    writeReport(output, report);
    stdout.writeln(
      jsonEncode({
        'phase': phase,
        'completed': report['completed'],
        'passed': report['passed'],
        'linguistic_gate_passed': report['linguistic_gate_passed'],
        'source_unchanged': report['source_unchanged'],
        'summary': report['summary'],
      }),
    );
    if (report['passed'] != true) exitCode = 1;
  }
}

void writeReport(File output, Map<String, Object?> report) {
  output.parent.createSync(recursive: true);
  final pending = File('${output.path}.next');
  pending.createSync(exclusive: true);
  pending.writeAsStringSync(
    '${const JsonEncoder.withIndent('  ').convert(report)}\n',
    flush: true,
  );
  pending.renameSync(output.path);
}

OfficeProofingRequest requestFor(Map<String, dynamic> item) =>
    OfficeProofingRequest(
      documentId: 'w7-expanded-grammar',
      blockId: item['id'] as String,
      revision: 1,
      text: item['text'] as String,
      language: item['locale'] as String,
      startOffset: item['start_offset'] as int,
      ignoredRuleIds: (item['ignored_rule_ids'] as List? ?? const [])
          .cast<String>()
          .toSet(),
    );

LinguisticAnalysisRequest analysisFor(OfficeProofingRequest request) =>
    LinguisticAnalysisRequest(
      text: request.text,
      documentId: request.documentId,
      blockId: request.blockId,
      revision: request.revision,
      locale: request.language,
      startOffset: request.startOffset,
      configurationId: 'w7-frozen-grammar-evaluation',
    );

final class _Completed extends OfficeProofingProvider {
  _Completed(this.result);
  final OfficeProofingResult result;
  @override
  String get id => result.providerId;
  @override
  bool supportsLanguage(String language) => true;
  @override
  Future<OfficeProofingResult> check(OfficeProofingRequest request) async =>
      result;
}

Map<String, Object?> finding(OfficeProofingFinding f) => {
  'provider_id': f.providerId,
  'rule_id': f.ruleId,
  'start': f.start,
  'end': f.end,
  'original': f.originalText,
  'replacements': f.replacements,
  'category': f.category.name,
};

Map<String, Object?> arm(OfficeProofingReport report, String expected) {
  final request = report.request;
  final edits =
      report.findings.where((f) => f.replacements.length == 1).toList()
        ..sort((a, b) => a.start.compareTo(b.start));
  var previousEnd = -1;
  var applicable = true;
  for (final f in edits) {
    final start = f.start - request.startOffset;
    final end = f.end - request.startOffset;
    require(
      start >= 0 &&
          end >= start &&
          end <= request.text.length &&
          request.text.substring(start, end) == f.originalText &&
          GermanAgreementFacts.hasSourceBoundaries(request.text, start, end),
      'Invalid original-text or grapheme mapping.',
    );
    if (start < previousEnd) applicable = false;
    previousEnd = end;
  }
  var text = request.text;
  if (applicable) {
    for (final f in edits.reversed) {
      text = text.replaceRange(
        f.start - request.startOffset,
        f.end - request.startOffset,
        f.replacements.single,
      );
    }
  }
  final choicesResolved = report.findings.every(
    (f) => f.replacements.length == 1,
  );
  return {
    'complete': report.complete,
    'statuses': [
      for (final r in report.results)
        {
          'provider': r.providerId,
          'status': r.status.name,
          'message': r.message,
        },
    ],
    'findings': report.findings.map(finding).toList(),
    'output': text,
    'changed': text != request.text,
    'applicable_nonoverlapping_edits': applicable,
    'all_choices_resolved': choicesResolved,
    'exact':
        report.complete && applicable && choicesResolved && text == expected,
  };
}

Map<String, int> summarize(List<Map<String, Object?>> rows) {
  int count(bool Function(Map<String, Object?>) predicate) =>
      rows.where(predicate).length;
  Map<String, Object?> armOf(Map<String, Object?> r, String name) =>
      r[name] is Map<String, Object?>
      ? r[name]! as Map<String, Object?>
      : const {};
  bool exact(Map<String, Object?> r, String name) =>
      armOf(r, name)['exact'] == true;
  bool flagged(Map<String, Object?> r, String name) =>
      (armOf(r, name)['findings'] as List<Object?>? ?? []).isNotEmpty;
  return {
    'completed_cases': count((r) => r.containsKey('combined')),
    'error_cases': count((r) => r['kind'] == 'error'),
    'valid_cases': count((r) => r['kind'] == 'valid'),
    'guard_cases': count((r) => r['kind'] == 'guard'),
    'unsupported_cases': count((r) => r['kind'] == 'unsupported'),
    'baseline_exact_errors': count(
      (r) => r['kind'] == 'error' && exact(r, 'baseline'),
    ),
    'model_exact_errors': count(
      (r) => r['kind'] == 'error' && exact(r, 'model'),
    ),
    'combined_exact_errors': count(
      (r) => r['kind'] == 'error' && exact(r, 'combined'),
    ),
    'combined_missed_errors': count(
      (r) => r['kind'] == 'error' && !exact(r, 'combined'),
    ),
    'additional_exact_errors': count(
      (r) =>
          r['kind'] == 'error' && exact(r, 'combined') && !exact(r, 'baseline'),
    ),
    'baseline_exact_lost': count(
      (r) =>
          r['kind'] == 'error' && exact(r, 'baseline') && !exact(r, 'combined'),
    ),
    'baseline_valid_flagged': count(
      (r) => r['kind'] == 'valid' && flagged(r, 'baseline'),
    ),
    'model_valid_flagged': count(
      (r) => r['kind'] == 'valid' && flagged(r, 'model'),
    ),
    'combined_valid_flagged': count(
      (r) => r['kind'] == 'valid' && flagged(r, 'combined'),
    ),
    'model_guard_flagged': count(
      (r) => r['kind'] == 'guard' && flagged(r, 'model'),
    ),
    'baseline_changed_valid': count(
      (r) => r['kind'] == 'valid' && armOf(r, 'baseline')['changed'] == true,
    ),
    'model_changed_valid': count(
      (r) => r['kind'] == 'valid' && armOf(r, 'model')['changed'] == true,
    ),
    'combined_changed_valid': count(
      (r) => r['kind'] == 'valid' && armOf(r, 'combined')['changed'] == true,
    ),
    'added_changed_valid': count(
      (r) =>
          r['kind'] == 'valid' &&
          armOf(r, 'combined')['changed'] == true &&
          armOf(r, 'baseline')['changed'] != true,
    ),
    'operational_failures': count((r) => r['operational_ok'] != true),
    'model_proposals': rows.fold(
      0,
      (n, r) => n + ((armOf(r, 'model')['findings'] as List?)?.length ?? 0),
    ),
    'merge_suppressed_proposals': rows.fold(
      0,
      (n, r) => n + ((r['model_proposals_suppressed_by_merge'] as int?) ?? 0),
    ),
    'model_abstentions_or_compatible_cases': count(
      (r) =>
          r['kind'] != 'unsupported' &&
          r['operational_ok'] == true &&
          !flagged(r, 'model'),
    ),
    'recheck_executions': count((r) => r.containsKey('corrected_recheck')),
    'recheck_model_findings': rows.fold(
      0,
      (n, r) =>
          n +
          (((r['corrected_recheck'] as Map<String, Object?>?)?['findings']
                      as List<Object?>?)
                  ?.length ??
              0),
    ),
  };
}

Map<String, int> decisionCounts(List<Map<String, Object?>> rows, String field) {
  final result = <String, int>{};
  for (final row in rows) {
    for (final raw in row['decisions'] as List<Object?>? ?? const []) {
      final decision = raw! as Map<String, Object?>;
      final key = decision[field]! as String;
      result[key] = (result[key] ?? 0) + 1;
    }
  }
  return result;
}

Map<String, String> verifyBaseline(String root, Map<String, dynamic> protocol) {
  final baseline = protocol['baseline'] as Map<String, dynamic>;
  final recorded =
      baseline['ordinary_provider_sources_sha256'] as Map<String, dynamic>;
  for (final entry in recorded.entries) {
    final file = File('$root/${entry.key}');
    var bytes = file.readAsBytesSync();
    if (entry.key.toString().endsWith('/german_writing_rules.dart')) {
      bytes = utf8.encode(
        utf8
            .decode(bytes)
            .replaceAll("part 'german_agreement_facts.dart';\n", ''),
      );
    }
    require(
      digest(bytes) == entry.value,
      'Current deterministic source changed: ${entry.key}',
    );
  }
  // The frozen protocol pins main. Verify its complete proofing data and
  // decompression closure, not merely the top-level provider source files.
  final listing = Process.runSync('git', [
    'ls-tree',
    '-r',
    baseline['main_commit']! as String,
    '--',
    'packages/quds_office_engine/lib/src/proofing',
    'packages/quds_office_engine/lib/src/opc/zip',
  ], workingDirectory: root);
  require(listing.exitCode == 0, 'Pinned baseline Git tree is unavailable.');
  final verified = <String, String>{};
  for (final line in (listing.stdout as String).trim().split('\n')) {
    final fields = line.split('\t');
    require(fields.length == 2, 'Malformed baseline tree record.');
    final metadata = fields.first.split(' ');
    require(
      metadata.length == 3 && metadata[1] == 'blob',
      'Unexpected baseline entry.',
    );
    final path = fields.last;
    var bytes = File('$root/$path').readAsBytesSync();
    if (path.endsWith('/german_writing_rules.dart')) {
      bytes = utf8.encode(
        utf8
            .decode(bytes)
            .replaceAll("part 'german_agreement_facts.dart';\n", ''),
      );
    }
    final blob = sha1.convert([
      ...utf8.encode('blob ${bytes.length}\u0000'),
      ...bytes,
    ]).toString();
    require(
      blob == metadata[2],
      'Deterministic baseline closure changed: $path',
    );
    verified[path] = blob;
  }
  require(verified.isNotEmpty, 'Empty deterministic baseline closure.');
  return verified;
}

Map<String, String> sourceHashes(String root) {
  final paths = <String>{
    'pubspec.yaml',
    'pubspec.lock',
    'packages/quds_office_engine/pubspec.yaml',
    'packages/quds_proofing_spacy/pubspec.yaml',
    'packages/quds_proofing_spacy/tool/evaluate_grammar.dart',
    'packages/quds_proofing_spacy/tool/evaluate_w7_coverage.dart',
    protocolPath,
  };
  for (final prefix in [
    'packages/quds_office_engine/lib',
    'packages/quds_proofing_spacy/lib',
  ]) {
    for (final entity in Directory(
      '$root/$prefix',
    ).listSync(recursive: true, followLinks: false)) {
      if (entity is File) paths.add(entity.path.substring(root.length + 1));
    }
  }
  return {
    for (final path in paths.toList()..sort())
      path: digest(File('$root/$path').readAsBytesSync()),
  };
}

String digest(List<int> bytes) => sha256.convert(bytes).toString();
bool equalMap(Map<Object?, Object?> a, Map<Object?, Object?> b) =>
    a.length == b.length && a.entries.every((e) => b[e.key] == e.value);
void require(bool value, String message) {
  if (!value) throw StateError(message);
}
