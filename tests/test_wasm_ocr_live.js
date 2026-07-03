#!/usr/bin/env node
/**
 * tests/test_wasm_ocr_live.js — Live WASM module smoke tests.
 *
 * Loads the actual compiled WASM module in Node.js and validates:
 *   1. Module initializes without error
 *   2. All exported C functions are accessible
 *   3. Version string is correct
 *   4. Functions return expected types for edge cases (NULL ctx, etc.)
 *
 * Run:  node tests/test_wasm_ocr_live.js
 *       (requires build-wasm/ to exist — run build-wasm.sh first)
 */

'use strict';

const path = require('path');
const fs = require('fs');

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

const wasmDir = path.join(__dirname, '..', 'build-wasm');
const wasmJs = path.join(wasmDir, 'crispembed_ocr.js');

if (!fs.existsSync(wasmJs)) {
  console.error('build-wasm/ not found. Run build-wasm.sh first.');
  process.exit(2);
}

(async () => {
  try {
    // ── Test 1: Module loads ──────────────────────────────────────────
    section('Module initialization');

    const CrispEmbedOCR = require(wasmJs);
    assert(typeof CrispEmbedOCR === 'function', 'factory function exists');

    // In Node.js, we must provide locateFile so the module can find the .wasm
    // (the default expects fetch() which is only available in browsers).
    const wasmBinary = fs.readFileSync(path.join(wasmDir, 'crispembed_ocr.wasm'));
    const module = await CrispEmbedOCR({ wasmBinary });
    assert(module !== null, 'module initialized');
    assert(typeof module.ccall === 'function', 'ccall available');
    assert(typeof module.cwrap === 'function', 'cwrap available');
    assert(typeof module.FS === 'object', 'FS (MEMFS) available');
    assert(typeof module._malloc === 'function', '_malloc available');
    assert(typeof module._free === 'function', '_free available');
    assert(typeof module.UTF8ToString === 'function', 'UTF8ToString available');
    assert(typeof module.getValue === 'function', 'getValue available');
    assert(typeof module.setValue === 'function', 'setValue available');
    assert(module.HEAPU8 !== undefined || typeof module.getValue === 'function', 'HEAP or getValue available');

    // ── Test 2: Version ──────────────────────────────────────────────
    section('Version string');

    const version = module.ccall('wasm_ocr_version', 'string', [], []);
    assert(version !== null, 'version returns non-null');
    assert(version.startsWith('crispembed-ocr-wasm-'), `version prefix: ${version}`);
    assertEqual(version, 'crispembed-ocr-wasm-0.3.0', 'version matches');

    // ── Test 3: All exported functions accessible ────────────────────
    section('Exported functions');

    const expectedFunctions = [
      // Single-model recognition
      'wasm_ocr_version',
      'wasm_ocr_init',
      'wasm_ocr_recognize_gray',
      'wasm_ocr_recognize',
      'wasm_ocr_recognize_copy',
      'wasm_ocr_confidences',
      'wasm_ocr_mean_confidence',
      'wasm_ocr_set_max_tokens',
      'wasm_ocr_free',
      // Pipeline
      'wasm_ocr_pipeline_init',
      'wasm_ocr_pipeline_run',
      'wasm_ocr_pipeline_free',
      'wasm_ocr_pipeline_full_init',
      'wasm_ocr_pipeline_full_run',
      'wasm_ocr_pipeline_full_free',
      // Scan cleanup
      'wasm_scan_cleanup_init',
      'wasm_scan_cleanup_process',
      'wasm_scan_cleanup_free_image',
      'wasm_scan_cleanup_free',
      'wasm_scan_cleanup_detect_page_split',
      'wasm_scan_cleanup_content_bbox',
      // Render
      'wasm_ocr_render',
      // Text detection
      'wasm_text_det_init',
      'wasm_text_det_run',
      'wasm_text_det_free',
      // Layout detection
      'wasm_layout_init',
      'wasm_layout_detect',
      'wasm_layout_free',
    ];

    for (const fn of expectedFunctions) {
      const exists = typeof module[`_${fn}`] === 'function';
      assert(exists, `_${fn} exported in WASM`);
    }

    // ── Test 4: Null-safety of key functions ─────────────────────────
    section('Null-safety');

    // wasm_ocr_init with non-existent model should return 0 (NULL)
    const badCtx = module.ccall('wasm_ocr_init', 'number',
      ['string', 'number'], ['/nonexistent.gguf', 1]);
    assertEqual(badCtx, 0, 'wasm_ocr_init with bad path returns NULL');

    // wasm_ocr_free with NULL should not crash
    module.ccall('wasm_ocr_free', null, ['number'], [0]);
    passed++;
    console.log('  wasm_ocr_free(NULL) did not crash');

    // wasm_ocr_pipeline_init with bad paths should return NULL
    const badPipeline = module.ccall('wasm_ocr_pipeline_init', 'number',
      ['string', 'string', 'number'], ['/bad_det.gguf', '/bad_rec.gguf', 1]);
    assertEqual(badPipeline, 0, 'wasm_ocr_pipeline_init with bad paths returns NULL');

    // wasm_scan_cleanup_init with empty string (classical-only) should succeed
    const cleanupCtx = module.ccall('wasm_scan_cleanup_init', 'number',
      ['string', 'number'], ['', 1]);
    assert(cleanupCtx !== 0, 'wasm_scan_cleanup_init with empty model (classical) succeeds');

    if (cleanupCtx) {
      // Test scan cleanup on a small synthetic image (4x4 white)
      const w = 4, h = 4, ch = 3;
      const imgSize = w * h * ch;
      const pixelPtr = module._malloc(imgSize);
      // Fill with white (255)
      for (let i = 0; i < imgSize; i++) {
        module.setValue(pixelPtr + i, 255, 'i8');
      }

      const owPtr = module._malloc(4);
      const ohPtr = module._malloc(4);

      const outPtr = module.ccall('wasm_scan_cleanup_process', 'number',
        ['number', 'number', 'number', 'number', 'number',
         'number', 'number', 'number', 'number', 'number', 'number'],
        [cleanupCtx, pixelPtr, w, h, ch, 0, 0, 0, 0, owPtr, ohPtr]);

      if (outPtr) {
        const ow = module.getValue(owPtr, 'i32');
        const oh = module.getValue(ohPtr, 'i32');
        assert(ow > 0 && oh > 0, `cleanup output has valid dimensions: ${ow}x${oh}`);
        module.ccall('wasm_scan_cleanup_free_image', null, ['number'], [outPtr]);
      } else {
        // A 4x4 white image might legitimately fail cleanup — that's OK
        passed++;
        console.log('  cleanup returned NULL for tiny white image (expected)');
      }

      module._free(pixelPtr);
      module._free(owPtr);
      module._free(ohPtr);

      // Content bbox on white image should return -1 (blank page)
      const bboxPixelPtr = module._malloc(imgSize);
      for (let i = 0; i < imgSize; i++) module.setValue(bboxPixelPtr + i, 255, 'i8');
      const x0Ptr = module._malloc(4);
      const y0Ptr = module._malloc(4);
      const x1Ptr = module._malloc(4);
      const y1Ptr = module._malloc(4);
      const bboxRc = module.ccall('wasm_scan_cleanup_content_bbox', 'number',
        ['number', 'number', 'number', 'number', 'number', 'number', 'number', 'number'],
        [bboxPixelPtr, w, h, ch, x0Ptr, y0Ptr, x1Ptr, y1Ptr]);
      assertEqual(bboxRc, -1, 'content_bbox returns -1 for blank (white) image');

      module._free(bboxPixelPtr);
      module._free(x0Ptr); module._free(y0Ptr);
      module._free(x1Ptr); module._free(y1Ptr);

      // Page split on 4x4 image should return -1
      const splitPixelPtr = module._malloc(imgSize);
      for (let i = 0; i < imgSize; i++) module.setValue(splitPixelPtr + i, 255, 'i8');
      const splitCol = module.ccall('wasm_scan_cleanup_detect_page_split', 'number',
        ['number', 'number', 'number', 'number'], [splitPixelPtr, w, h, ch]);
      assertEqual(splitCol, -1, 'detect_page_split returns -1 for tiny image');
      module._free(splitPixelPtr);

      module.ccall('wasm_scan_cleanup_free', null, ['number'], [cleanupCtx]);
      passed++;
      console.log('  wasm_scan_cleanup_free did not crash');
    }

    // ── Test 5: MEMFS operations ────────────────────────────────────
    section('MEMFS');

    // Create directory and write a file
    try { module.FS.mkdir('/test'); } catch (_) {}
    module.FS.writeFile('/test/hello.txt', 'Hello WASM');
    const contents = module.FS.readFile('/test/hello.txt', { encoding: 'utf8' });
    assertEqual(contents, 'Hello WASM', 'MEMFS write/read roundtrip');

    // Clean up
    module.FS.unlink('/test/hello.txt');
    module.FS.rmdir('/test');
    passed++;
    console.log('  MEMFS cleanup succeeded');

    // ── Test 6: Memory allocation ───────────────────────────────────
    section('Memory allocation');

    // Allocate 1MB — should work with ALLOW_MEMORY_GROWTH
    const bigPtr = module._malloc(1024 * 1024);
    assert(bigPtr > 0, 'allocate 1MB succeeded');
    module._free(bigPtr);

    // Allocate 64MB — tests memory growth
    const hugePtr = module._malloc(64 * 1024 * 1024);
    assert(hugePtr > 0, 'allocate 64MB succeeded (memory growth)');
    module._free(hugePtr);

    // ── Results ─────────────────────────────────────────────────────
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

  } catch (e) {
    console.error(`\nFATAL: ${e.message}`);
    console.error(e.stack);
    process.exit(1);
  }
})();
