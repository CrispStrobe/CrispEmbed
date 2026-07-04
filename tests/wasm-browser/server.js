#!/usr/bin/env node
/**
 * tests/wasm-browser/server.js — static server for the WASM OCR browser e2e.
 *
 * Serves a composed document root (no build artifacts copied around):
 *   /                 -> examples/wasm-ocr/            (index.html)
 *   /crispembed_ocr.* -> build-wasm/                   (Emscripten output)
 *   /crispembed-ocr.js-> wasm/                         (high-level wrapper)
 *   /models/*         -> $CRISPEMBED_MODELS_DIR        (GGUF files, local)
 *   /images/*         -> tests/regression/images/      (fixtures)
 *
 * Env:
 *   PORT                   (default 8093)
 *   CRISPEMBED_MODELS_DIR  (default /mnt/storage/gguf-models)
 *   WASM_COI=1             also send COOP/COEP headers (threaded builds)
 *
 * Usage:  node tests/wasm-browser/server.js
 */

'use strict';

const http = require('http');
const fs = require('fs');
const path = require('path');

const repoRoot = path.join(__dirname, '..', '..');
const MODELS_DIR = process.env.CRISPEMBED_MODELS_DIR || '/mnt/storage/gguf-models';
const PORT = parseInt(process.env.PORT || '8093', 10);

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'application/javascript',
  '.wasm': 'application/wasm',
  '.gguf': 'application/octet-stream',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
};

function resolveUrl(urlPath) {
  // Normalize + strip query, forbid traversal.
  const clean = decodeURIComponent(urlPath.split('?')[0]).replace(/\.\./g, '');
  if (clean === '/' || clean === '/index.html') {
    return path.join(repoRoot, 'examples', 'wasm-ocr', 'index.html');
  }
  if (clean === '/serve.py') return path.join(repoRoot, 'examples', 'wasm-ocr', 'serve.py');
  if (clean === '/ocr-worker.js' || clean === '/coi-sw.js') {
    return path.join(repoRoot, 'examples', 'wasm-ocr', path.basename(clean));
  }
  if (clean === '/crispembed_ocr.js' || clean === '/crispembed_ocr.wasm') {
    return path.join(repoRoot, 'build-wasm', path.basename(clean));
  }
  if (clean.startsWith('/threaded/')) {
    // Deterministic tests: threaded build only exposed when requested.
    if (process.env.WASM_E2E_THREADS !== '1') return null;
    return path.join(repoRoot, 'build-wasm-threads', path.basename(clean));
  }
  if (clean === '/crispembed-ocr.js') {
    return path.join(repoRoot, 'wasm', 'crispembed-ocr.js');
  }
  if (clean.startsWith('/models/')) {
    return path.join(MODELS_DIR, path.basename(clean));
  }
  if (clean.startsWith('/images/')) {
    return path.join(repoRoot, 'tests', 'regression', 'images', path.basename(clean));
  }
  return null;
}

const server = http.createServer((req, res) => {
  const file = resolveUrl(req.url);
  if (!file || !fs.existsSync(file) || !fs.statSync(file).isFile()) {
    res.writeHead(404, { 'Content-Type': 'text/plain' });
    res.end('not found: ' + req.url);
    return;
  }
  const headers = {
    'Content-Type': MIME[path.extname(file)] || 'application/octet-stream',
    'Content-Length': fs.statSync(file).size,
    'Cache-Control': 'no-store',
  };
  if (process.env.WASM_COI === '1') {
    headers['Cross-Origin-Opener-Policy'] = 'same-origin';
    headers['Cross-Origin-Embedder-Policy'] = 'require-corp';
  }
  res.writeHead(200, headers);
  fs.createReadStream(file).pipe(res);
});

server.listen(PORT, '127.0.0.1', () => {
  console.log(`wasm-browser test server on http://127.0.0.1:${PORT}  (models: ${MODELS_DIR})`);
});

module.exports = server;
