import 'dart:io';
import 'dart:math';

import '../lib/crispembed.dart';

void assertTrue(bool condition, String message) {
  if (!condition) {
    stderr.writeln(message);
    exitCode = 1;
    throw StateError(message);
  }
}

double l2Norm(List<double> values) {
  var sum = 0.0;
  for (final value in values) {
    sum += value * value;
  }
  return sqrt(sum);
}

bool vectorsDiffer(List<double> a, List<double> b, {double atol = 1e-6}) {
  for (var i = 0; i < a.length; i++) {
    if ((a[i] - b[i]).abs() > atol) {
      return true;
    }
  }
  return false;
}

void main(List<String> args) {
  if (args.isEmpty) {
    stderr.writeln(
      'usage: dart run example/feature_parity.dart <dense.gguf> [retrieval.gguf] [reranker.gguf] [lib-path]',
    );
    exit(64);
  }

  final denseModel = args[0];
  final retrievalModel = args.length > 1 ? args[1] : null;
  final rerankerModel = args.length > 2 ? args[2] : null;
  final libPath = args.length > 3 ? args[3] : Platform.environment['CRISPEMBED_LIB'];

  stdout.writeln('[dart] dense model: $denseModel');
  final dense = CrispEmbed(denseModel, nThreads: 4, libPath: libPath);
  try {
    final vec = dense.encode('Hello world');
    assertTrue(vec.isNotEmpty, 'single encode returned no values');
    assertTrue((l2Norm(vec) - 1.0).abs() < 1e-3, 'single encode is not normalized');

    final texts = [
      'query: crisp embeddings are fast',
      'dense retrieval with ggml',
      'batch inference should preserve order',
    ];
    final batch = dense.encodeBatch(texts);
    assertTrue(batch.length == texts.length, 'batch size mismatch');
    assertTrue(batch.first.length == vec.length, 'batch dim mismatch');
    final singleAgain = dense.encode(texts.first);
    assertTrue(
      batch.first.length == singleAgain.length &&
          !vectorsDiffer(batch.first, singleAgain, atol: 1e-5),
      'batch encode disagrees with single encode',
    );

    final truncDim = min(128, vec.length);
    dense.setDim(truncDim);
    final vecTrunc = dense.encode('Hello world');
    assertTrue(vecTrunc.length == truncDim, 'setDim did not truncate output');
    assertTrue((l2Norm(vecTrunc) - 1.0).abs() < 1e-3, 'truncated encode is not normalized');
    dense.setDim(0);
    assertTrue(dense.encode('Hello world').length == vec.length, 'setDim(0) did not restore native dim');

    dense.setPrefix('query: ');
    assertTrue(dense.prefix == 'query: ', 'prefix getter mismatch');
    final prefixed = dense.encode('hello');
    dense.setPrefix('');
    final cleared = dense.encode('hello');
    assertTrue(dense.prefix.isEmpty, 'prefix did not clear');
    assertTrue(prefixed.length == cleared.length, 'prefix changed output dim');
    assertTrue(vectorsDiffer(prefixed, cleared), 'prefix had no effect on embeddings');

    final docs = [
      'Paris is the capital city of France.',
      'A bicycle uses two wheels and a chain.',
      'Berlin is the capital of Germany.',
    ];
    final ranked = dense.rerankBiencoder('capital of france', docs, topN: 2);
    assertTrue(ranked.length == 2, 'rerankBiencoder topN mismatch');
    assertTrue(ranked.first.index == 0, 'rerankBiencoder did not rank the relevant document first');
    assertTrue(ranked.first.score >= ranked[1].score, 'rerankBiencoder results are not sorted');

    stdout.writeln('[dart] dense, batch, matryoshka, prefix, and bi-encoder rerank: PASS');
  } finally {
    dense.dispose();
  }

  if (retrievalModel != null) {
    stdout.writeln('[dart] retrieval model: $retrievalModel');
    final retrieval = CrispEmbed(retrievalModel, nThreads: 4, libPath: libPath);
    try {
      assertTrue(retrieval.hasSparse, 'retrieval model does not report sparse support');
      final sparse = retrieval.encodeSparse('Paris is the capital of France.');
      assertTrue(sparse.isNotEmpty, 'encodeSparse returned no entries');
      assertTrue(sparse.values.every((value) => value > 0.0), 'encodeSparse returned non-positive weights');

      assertTrue(retrieval.hasColbert, 'retrieval model does not report colbert support');
      final multi = retrieval.encodeMultivec('Paris is the capital of France.');
      assertTrue(multi.isNotEmpty, 'encodeMultivec returned no token vectors');
      assertTrue(multi.first.isNotEmpty, 'encodeMultivec returned zero-width token vectors');
      for (final token in multi) {
        assertTrue((l2Norm(token) - 1.0).abs() < 5e-3, 'token vector is not normalized');
      }

      stdout.writeln('[dart] sparse and colbert retrieval: PASS');
    } finally {
      retrieval.dispose();
    }
  } else {
    stdout.writeln('[dart] sparse and colbert retrieval: SKIP (no retrieval model)');
  }

  if (rerankerModel != null) {
    stdout.writeln('[dart] reranker model: $rerankerModel');
    final reranker = CrispEmbed(rerankerModel, nThreads: 4, libPath: libPath);
    try {
      assertTrue(reranker.isReranker, 'reranker model does not report reranker support');
      final positive = reranker.rerank('capital of france', 'Paris is the capital of France.');
      final negative = reranker.rerank('capital of france', 'Bicycles have handlebars and pedals.');
      assertTrue(positive.isFinite && negative.isFinite, 'rerank returned non-finite score');
      assertTrue(positive > negative, 'reranker failed to score the relevant document higher');

      stdout.writeln('[dart] cross-encoder rerank: PASS');
    } finally {
      reranker.dispose();
    }
  } else {
    stdout.writeln('[dart] cross-encoder rerank: SKIP (no reranker model)');
  }

  stdout.writeln('[dart] feature parity script completed');
}
