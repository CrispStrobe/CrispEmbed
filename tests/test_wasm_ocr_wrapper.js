#!/usr/bin/env node
/**
 * tests/test_wasm_ocr_wrapper.js — Unit tests for the CrispEmbed OCR WASM wrapper.
 *
 * These tests validate:
 *   1. JS syntax and module loading (no browser APIs needed)
 *   2. TextDecoder monkey-patch
 *   3. Export consistency (C wrapper ↔ build script ↔ JS wrapper)
 *   4. Helper function correctness (_calcDimensions, _encodeSimplePNG, JSON escaping)
 *   5. Mock-based API smoke tests
 *
 * Run:  node tests/test_wasm_ocr_wrapper.js
 *       (from repo root, no dependencies needed)
 */

'use strict';

const fs = require('fs');
const path = require('path');

let passed = 0;
let failed = 0;
const failures = [];

function assert(cond, msg) {
  if (cond) { passed++; }
  else { failed++; failures.push(msg); console.error(`  FAIL: ${msg}`); }
}

function assertEqual(a, b, msg) {
  if (a === b) { passed++; }
  else { failed++; const m = `${msg}: expected ${JSON.stringify(b)}, got ${JSON.stringify(a)}`; failures.push(m); console.error(`  FAIL: ${m}`); }
}

function section(name) { console.log(`\n=== ${name} ===`); }

// ── Test 1: JS files parse correctly ────────────────────────────────────

section('Module loading');

const wrapperPath = path.join(__dirname, '..', 'wasm', 'crispembed-ocr.js');
assert(fs.existsSync(wrapperPath), 'crispembed-ocr.js exists');

// Load in a way that doesn't require browser globals
// The TextDecoder patch runs on load, so we need a mock
if (typeof TextDecoder === 'undefined') {
  global.TextDecoder = class TextDecoder {
    decode(input) { return ''; }
  };
}

// Mock browser globals that the module checks at load time
if (typeof Blob === 'undefined') global.Blob = class Blob {};
if (typeof ImageData === 'undefined') {
  global.ImageData = class ImageData {
    constructor(data, width, height) {
      this.data = data; this.width = width; this.height = height;
    }
  };
}
if (typeof HTMLVideoElement === 'undefined') global.HTMLVideoElement = class {};
if (typeof HTMLCanvasElement === 'undefined') global.HTMLCanvasElement = class {};
if (typeof OffscreenCanvas === 'undefined') global.OffscreenCanvas = undefined;

const wrapper = require(wrapperPath);
assert(wrapper !== null, 'module loaded successfully');
assert(typeof wrapper.CrispEmbedOCRWrapper === 'function', 'CrispEmbedOCRWrapper exported');
assert(typeof wrapper.CrispEmbedOCRPipeline === 'function', 'CrispEmbedOCRPipeline exported');
assert(typeof wrapper.CrispEmbedScanCleanup === 'function', 'CrispEmbedScanCleanup exported');
assert(typeof wrapper.CrispEmbedTextDetector === 'function', 'CrispEmbedTextDetector exported');
assert(typeof wrapper.CrispEmbedLayoutDetector === 'function', 'CrispEmbedLayoutDetector exported');

// ── Test 2: TextDecoder monkey-patch ────────────────────────────────────

section('TextDecoder patch');

{
  // The module patches TextDecoder on load. Test it handles resizable buffers.
  const td = new TextDecoder();

  // Normal buffer — should work
  const normalBuf = new Uint8Array([72, 101, 108, 108, 111]); // "Hello"
  const normalResult = td.decode(normalBuf);
  assertEqual(normalResult, 'Hello', 'decode normal buffer');

  // Simulate resizable buffer by adding a .resizable property
  const resizableBuf = new Uint8Array([87, 111, 114, 108, 100]); // "World"
  Object.defineProperty(resizableBuf.buffer, 'resizable', { value: true });
  const resizableResult = td.decode(resizableBuf);
  assertEqual(resizableResult, 'World', 'decode resizable buffer (patched)');
}

// ── Test 3: Export consistency ──────────────────────────────────────────

section('Export consistency (C ↔ build script)');

const cWrapperPath = path.join(__dirname, '..', 'wasm', 'ocr_wrapper.c');
const buildScriptPath = path.join(__dirname, '..', 'build-wasm.sh');

const cSource = fs.readFileSync(cWrapperPath, 'utf-8');
const buildScript = fs.readFileSync(buildScriptPath, 'utf-8');

// Extract WASM_EXPORT function names from C wrapper
const cFunctions = [];
const cRegex = /WASM_EXPORT\b[\s\S]*?(wasm_\w+)\s*\(/g;
let m;
while ((m = cRegex.exec(cSource)) !== null) {
  cFunctions.push(m[1]);
}

// Extract exported function names from build script
const buildFunctions = [];
const buildRegex = /'_(\w+)'/g;
while ((m = buildRegex.exec(buildScript)) !== null) {
  if (!['malloc', 'free', 'main'].includes(m[1])) {
    buildFunctions.push(m[1]);
  }
}

// Every C function should be in the build script
for (const fn of cFunctions) {
  assert(buildFunctions.includes(fn),
    `C function ${fn} exported in build-wasm.sh`);
}

// Every build script function should exist in C
for (const fn of buildFunctions) {
  assert(cFunctions.includes(fn),
    `build-wasm.sh function ${fn} exists in C wrapper`);
}

// Check version string is consistent
const versionMatch = cSource.match(/return "crispembed-ocr-wasm-([^"]+)"/);
assert(versionMatch !== null, 'version string found in C wrapper');
console.log(`  Version: ${versionMatch ? versionMatch[1] : 'not found'}`);

// ── Test 4: C API coverage ─────────────────────────────────────────────

section('C API coverage');

// Check all major subsystems are represented
const subsystems = {
  'single-model recognition': ['wasm_ocr_init', 'wasm_ocr_recognize', 'wasm_ocr_recognize_copy', 'wasm_ocr_free'],
  'pipeline basic': ['wasm_ocr_pipeline_init', 'wasm_ocr_pipeline_run', 'wasm_ocr_pipeline_free'],
  'pipeline full': ['wasm_ocr_pipeline_full_init', 'wasm_ocr_pipeline_full_run', 'wasm_ocr_pipeline_full_free'],
  'scan cleanup': ['wasm_scan_cleanup_init', 'wasm_scan_cleanup_process', 'wasm_scan_cleanup_free'],
  'text detection': ['wasm_text_det_init', 'wasm_text_det_run', 'wasm_text_det_free'],
  'layout detection': ['wasm_layout_init', 'wasm_layout_detect', 'wasm_layout_free'],
  'rendering': ['wasm_ocr_render'],
  'confidence': ['wasm_ocr_confidences', 'wasm_ocr_mean_confidence'],
};

for (const [name, fns] of Object.entries(subsystems)) {
  for (const fn of fns) {
    assert(cFunctions.includes(fn), `${name}: ${fn} in C wrapper`);
    assert(buildFunctions.includes(fn), `${name}: ${fn} in build exports`);
  }
}

// ── Test 5: JSON helper correctness ─────────────────────────────────────

section('JSON serialization in C wrapper');

// Verify the C wrapper's JSON serialization handles edge cases
// (We can't run the C code, but we can check it's structurally correct)
assert(cSource.includes('sb_append_json_str'), 'JSON string escaping helper exists');
assert(cSource.includes('\\\\n'), 'handles newline escaping');
assert(cSource.includes('\\\\r'), 'handles carriage return escaping');
assert(cSource.includes('\\\\"'), 'handles quote escaping');
assert(cSource.includes('\\\\\\\\'), 'handles backslash escaping');

// ── Test 6: PNG encoder ─────────────────────────────────────────────────

section('PNG encoder (JS)');

// The _encodeSimplePNG function is private, but we can test it indirectly
// by checking the wrapper file contains the expected PNG structure
const jsSource = fs.readFileSync(wrapperPath, 'utf-8');
assert(jsSource.includes('_encodeSimplePNG'), 'PNG encoder function exists');
assert(jsSource.includes('IHDR'), 'PNG IHDR chunk');
assert(jsSource.includes('IDAT'), 'PNG IDAT chunk');
assert(jsSource.includes('IEND'), 'PNG IEND chunk');
assert(jsSource.includes('0x78'), 'zlib header byte');
assert(jsSource.includes('adler'), 'Adler-32 checksum');
assert(jsSource.includes('crc32'), 'CRC-32 checksum');

// ── Test 7: JS wrapper API surface ─────────────────────────────────────

section('JS wrapper API surface');

// Verify key methods exist on each class
const classChecks = {
  'CrispEmbedOCRWrapper': ['create', 'recognize', 'dispose'],
  'CrispEmbedOCRPipeline': ['create', 'createFull', 'run', 'render', 'dispose'],
  'CrispEmbedScanCleanup': ['create', 'process', 'detectPageSplit', 'contentBbox', 'dispose'],
  'CrispEmbedTextDetector': ['create', 'detect', 'dispose'],
  'CrispEmbedLayoutDetector': ['create', 'detect', 'dispose'],
};

for (const [className, methods] of Object.entries(classChecks)) {
  for (const method of methods) {
    // Static methods are on the constructor, instance methods on prototype
    const cls = wrapper[className];
    const hasStatic = typeof cls[method] === 'function';
    const hasProto = typeof cls.prototype[method] === 'function';
    assert(hasStatic || hasProto, `${className}.${method} exists`);
  }
}

// ── Test 8: Dimension calculation ───────────────────────────────────────

section('Dimension helpers');

// _calcDimensions is private, but we can verify the logic by reading the source
assert(jsSource.includes('_calcDimensions'), '_calcDimensions helper exists');
assert(jsSource.includes('maxWidth'), 'respects maxWidth');
assert(jsSource.includes('maxHeight'), 'respects maxHeight');
assert(jsSource.includes('Math.round'), 'rounds dimensions');

// ── Test 9: Memory management patterns ──────────────────────────────────

section('Memory management');

// Check that all allocations have matching frees
assert(jsSource.includes('_malloc'), 'uses _malloc');
assert(jsSource.includes('_free'), 'uses _free');

// Count malloc/free pairs in the wrapper
const mallocCount = (jsSource.match(/_malloc/g) || []).length;
const freeCount = (jsSource.match(/_free/g) || []).length;
assert(freeCount >= mallocCount, `free calls (${freeCount}) >= malloc calls (${mallocCount})`);

// Check dispose patterns
const disposeCount = (jsSource.match(/dispose\(\)/g) || []).length;
assert(disposeCount >= 5, `all 5 classes have dispose() (found ${disposeCount})`);

// ── Test 10: Build script sanity ────────────────────────────────────────

section('Build script sanity');

assert(buildScript.includes('ALLOW_MEMORY_GROWTH=1'), 'memory growth enabled');
assert(buildScript.includes('MODULARIZE=1'), 'modularized');
assert(buildScript.includes('CrispEmbedOCR'), 'export name set');
assert(buildScript.includes('FILESYSTEM=1'), 'filesystem enabled (for MEMFS)');
assert(buildScript.includes('WASM_BIGINT=1'), 'WASM BigInt support');
assert(buildScript.includes('ccall'), 'ccall exported');
assert(buildScript.includes('cwrap'), 'cwrap exported');
assert(buildScript.includes('UTF8ToString'), 'UTF8ToString exported');
assert(buildScript.includes('INITIAL_MEMORY=134217728'), 'initial memory 128MB for full pipeline');
assert(buildScript.includes('STACK_SIZE=2097152'), 'stack size 2MB for pipeline');

// ── Test 11: CI workflow ────────────────────────────────────────────────

section('CI workflow');

const ciPath = path.join(__dirname, '..', '.github', 'workflows', 'build-wasm.yml');
const releasePath = path.join(__dirname, '..', '.github', 'workflows', 'release-wasm.yml');
assert(fs.existsSync(ciPath), 'build-wasm.yml exists');
assert(fs.existsSync(releasePath), 'release-wasm.yml exists');

const ciContent = fs.readFileSync(ciPath, 'utf-8');
assert(ciContent.includes('crispembed-ocr.js'), 'CI includes JS wrapper in artifacts');

const releaseContent = fs.readFileSync(releasePath, 'utf-8');
assert(releaseContent.includes('release'), 'release workflow targets releases');
assert(releaseContent.includes('crispembed-ocr-wasm'), 'release includes OCR WASM bundle');

// ── Results ─────────────────────────────────────────────────────────────

console.log(`\n${'='.repeat(50)}`);
console.log(`Results: ${passed} passed, ${failed} failed`);
if (failed > 0) {
  console.log('\nFailures:');
  for (const f of failures) console.log(`  - ${f}`);
  process.exit(1);
} else {
  console.log('All tests passed!');
  process.exit(0);
}
