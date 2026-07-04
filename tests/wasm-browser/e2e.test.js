#!/usr/bin/env node
/**
 * tests/wasm-browser/e2e.test.js — REAL browser end-to-end test of the WASM
 * OCR demo (examples/wasm-ocr/index.html) in headless Chromium.
 *
 * This exercises the exact flow a user of the demo page hits (GitHub #31):
 *   page load -> Emscripten module init -> model fetch() -> MEMFS write ->
 *   wasm_ocr_init -> Canvas RGBA extraction -> wasm_ocr_recognize_copy ->
 *   result rendering. Any pageerror (e.g. the TextDecoder resizable
 *   ArrayBuffer crash) fails the test.
 *
 * Ground truth: native CLI output for the same model + image
 *   build/crispembed -m pix2tex-mfr-q4_k.gguf --ocr formula_quadratic.png
 *   => x = \frac { - b \pm \sqrt { b ^ { 2 } - 4 a c } } { 2 a }
 *
 * Requirements:
 *   - build-wasm/ artifacts (run ./build-wasm.sh)
 *   - npm install && npx playwright install chromium   (in this directory)
 *   - pix2tex-mfr-q4_k.gguf in $CRISPEMBED_MODELS_DIR
 *
 * Run:  CRISPEMBED_MODELS_DIR=/path/to/models node tests/wasm-browser/e2e.test.js
 */

'use strict';

const path = require('path');
const fs = require('fs');

const PORT = parseInt(process.env.PORT || '8093', 10);
const BASE = `http://127.0.0.1:${PORT}`;
const MODELS_DIR = process.env.CRISPEMBED_MODELS_DIR || '/mnt/storage/gguf-models';
const repoRoot = path.join(__dirname, '..', '..');

// Native ground truth (see header). Whitespace-normalized comparison.
const PIX2TEX_GT = 'x = \\frac { - b \\pm \\sqrt { b ^ { 2 } - 4 a c } } { 2 a }';

let passed = 0, failed = 0;
const failures = [];
function assert(cond, msg) {
  if (cond) { passed++; console.log(`  ok: ${msg}`); }
  else { failed++; failures.push(msg); console.error(`  FAIL: ${msg}`); }
}
function norm(s) { return (s || '').replace(/\s+/g, ' ').trim(); }

(async () => {
  // Preflight
  const wasmJs = path.join(repoRoot, 'build-wasm', 'crispembed_ocr.js');
  if (!fs.existsSync(wasmJs)) {
    console.error('build-wasm/ missing — run ./build-wasm.sh first');
    process.exit(2);
  }
  const pix2tex = path.join(MODELS_DIR, 'pix2tex-mfr-q4_k.gguf');
  if (!fs.existsSync(pix2tex)) {
    console.error(`pix2tex-mfr-q4_k.gguf not found in ${MODELS_DIR}`);
    process.exit(2);
  }

  let chromium;
  try { ({ chromium } = require('playwright')); }
  catch (_) {
    console.error('playwright not installed — cd tests/wasm-browser && npm install && npx playwright install chromium');
    process.exit(2);
  }

  const server = require('./server.js');

  const WEBGPU = process.env.WASM_E2E_WEBGPU === '1';
  // Headless WebGPU needs explicit enabling; ANGLE picks Metal on macOS,
  // Vulkan/SwiftShader elsewhere.
  const browser = await chromium.launch(WEBGPU ? {
    args: ['--enable-unsafe-webgpu', '--enable-gpu',
           '--enable-features=Vulkan', '--ignore-gpu-blocklist'],
  } : {});
  const page = await browser.newPage();

  const pageErrors = [];
  const consoleErrors = [];
  page.on('pageerror', (e) => pageErrors.push(String(e)));
  page.on('console', (m) => { if (m.type() === 'error') consoleErrors.push(m.text()); });

  try {
    console.log('\n=== Page load + module availability ===');
    await page.goto(BASE + '/', { waitUntil: 'load' });
    // The COI service worker reloads the page once after first install to
    // apply COOP/COEP — wait out that navigation before asserting.
    await page.waitForTimeout(1200);
    await page.waitForLoadState('load');
    assert(await page.title() !== '', 'page has a title');
    console.log('  crossOriginIsolated:', await page.evaluate(() => crossOriginIsolated));
    // Inference runs in a Web Worker now — the page only needs the worker
    // script and its imports to be served.
    for (const f of ['ocr-worker.js', 'crispembed_ocr.js', 'crispembed-ocr.js', 'coi-sw.js']) {
      const st = await page.evaluate(async (u) => (await fetch(u)).status, f);
      assert(st === 200, `${f} is served (HTTP ${st})`);
    }

    // Deliberately select the IMAGE FIRST (the ordering from the #31
    // follow-up): nothing may auto-run, and Process stays disabled until a
    // model is loaded too.
    console.log('\n=== Select image BEFORE model — no auto-run ===');
    await page.setInputFiles('#file-input',
      path.join(repoRoot, 'tests', 'regression', 'images', 'formula_quadratic.png'));
    assert(await page.isDisabled('#btn-process'),
      'Process disabled with image but no model');

    console.log('\n=== Single-model: load pix2tex via demo UI ===');
    if (WEBGPU) {
      assert(await page.evaluate(() => !!navigator.gpu),
        'browser exposes navigator.gpu');
      await page.waitForSelector('#webgpu-opt', { state: 'visible', timeout: 10000 });
      await page.check('#opt-webgpu');
      console.log('  WebGPU toggle enabled');
    }
    await page.fill('#model-url', BASE + '/models/pix2tex-mfr-q4_k.gguf');
    await page.click('#btn-init');
    await page.waitForFunction(
      () => document.getElementById('status').textContent.includes('Model loaded')
         || document.getElementById('status').textContent.startsWith('Error'),
      null, { timeout: 120000 });
    const loadStatus = await page.textContent('#status');
    assert(loadStatus.includes('Model loaded'), `model loaded via UI (status: "${loadStatus}")`);
    assert(!(await page.isDisabled('#btn-process')),
      'Process enabled once model + image are both present');
    const loaderUsed = await page.evaluate(() => window.__loaderUsed);
    console.log(`  loader used: ${loaderUsed}`);
    if (WEBGPU) {
      assert(loaderUsed === 'webgpu/crispembed_ocr.js',
        `webgpu loader selected (got ${loaderUsed})`);
    } else if (process.env.WASM_E2E_THREADS === '1') {
      assert(loaderUsed === 'threaded/crispembed_ocr.js',
        `threaded loader selected (got ${loaderUsed})`);
    }

    console.log('\n=== Single-model: OCR formula_quadratic.png via Process ===');
    await page.click('#btn-process');
    // The whole point of the worker refactor: the page must stay responsive
    // while WASM computes. A main-thread round-trip must return promptly.
    await page.waitForTimeout(700);
    const t0 = Date.now();
    await page.evaluate(() => 1 + 1);
    const rt = Date.now() - t0;
    assert(rt < 1500, `page responsive during processing (round-trip ${rt} ms)`);
    await page.waitForFunction(
      () => {
        const s = document.getElementById('status').textContent;
        return s === 'Done.' || s === 'Processing failed.';
      }, null, { timeout: 300000 });
    const status = await page.textContent('#status');
    assert(status === 'Done.', `recognition completed (status: "${status}")`);
    const text = norm(await page.textContent('#result-text'));
    console.log(`  result: "${text}"`);
    assert(text === norm(PIX2TEX_GT),
      `WASM output matches native ground truth (got "${text}")`);

    console.log('\n=== Browser environment health ===');
    assert(pageErrors.length === 0,
      `no uncaught page errors (TextDecoder etc.): ${JSON.stringify(pageErrors)}`);
    // Emscripten routes C stderr (engine debug logs like "math_ocr: ...")
    // to console.error — only genuine JS/wasm errors count.
    // 404s are expected for optional resources (threaded/ probe, favicon).
    const realErrors = consoleErrors.filter(e =>
      !e.includes('favicon') && !/^[a-z0-9_]+:\s/.test(e)
      && !/status of 404/.test(e));
    assert(realErrors.length === 0, `no console errors: ${JSON.stringify(realErrors)}`);

    // Optional: full det+rec pipeline through the UI (slow in single-threaded
    // WASM). Enabled when both models are present and WASM_E2E_PIPELINE=1.
    const det = path.join(MODELS_DIR, 'dbnet-ic15-q4_k.gguf');
    const rec = path.join(MODELS_DIR, 'trocr-small-printed-q4_k.gguf');
    if (process.env.WASM_E2E_PIPELINE === '1' && fs.existsSync(det) && fs.existsSync(rec)) {
      console.log('\n=== Pipeline (DBNet + TrOCR) through the UI ===');
      await page.click('#tab-pipeline');
      await page.fill('#det-url', BASE + '/models/dbnet-ic15-q4_k.gguf');
      await page.fill('#rec-url', BASE + '/models/trocr-small-printed-q4_k.gguf');
      await page.click('#btn-init');
      await page.waitForFunction(
        () => document.getElementById('status').textContent.includes('Pipeline loaded')
           || document.getElementById('status').textContent.startsWith('Error'),
        null, { timeout: 180000 });
      const pStatus = await page.textContent('#status');
      assert(pStatus.includes('Pipeline loaded'), `pipeline loaded (status: "${pStatus}")`);

      await page.setInputFiles('#file-input',
        path.join(repoRoot, 'tests', 'regression', 'images', 'scan_strip.png'));
      await page.click('#btn-process');
      await page.waitForFunction(
        () => {
          const s = document.getElementById('status').textContent;
          return s === 'Done.' || s === 'Processing failed.';
        }, null, { timeout: 900000 });
      const pDone = await page.textContent('#status');
      assert(pDone === 'Done.', `pipeline run completed (status: "${pDone}")`);
      const pText = norm(await page.textContent('#result-text'));
      console.log(`  pipeline result: "${pText}"`);
      assert(pText.length > 0 && pText !== '(no text detected)',
        `pipeline detected + recognized text (got "${pText}")`);
      // Native GT (build-native/crispembed, same models): 5 regions —
      // MAMMAA / LIKE / TOOO / SUMMER / HEAVEN. Require detection to have
      // found regions and recognition to have produced real words.
      const pMeta = await page.textContent('#result-meta');
      console.log(`  pipeline meta: "${pMeta}"`);
      assert(/[1-9]\d* region/.test(pMeta), `pipeline found regions (${pMeta})`);
      assert(/MAMMA|SUMMER|HEAVEN|LIKE/i.test(pText),
        `pipeline text overlaps native ground truth (got "${pText}")`);
    }
  } finally {
    await browser.close();
    server.close();
  }

  console.log('\n' + '='.repeat(50));
  console.log(`Results: ${passed} passed, ${failed} failed`);
  if (failed > 0) { failures.forEach(f => console.error('  FAIL: ' + f)); process.exit(1); }
  console.log('All browser e2e tests passed!');
})().catch((e) => { console.error(e); process.exit(1); });
