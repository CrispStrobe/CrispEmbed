#!/usr/bin/env node
/**
 * tests/wasm-browser/e2e-live.test.js — end-to-end test against the DEPLOYED
 * GitHub Pages demo, using the page's default Hugging Face model URL.
 *
 * This is the exact end-user path from issue #31: public URL, cross-origin
 * model download (HF CORS), WASM init, canvas pixels, recognition.
 *
 * Run:  node tests/wasm-browser/e2e-live.test.js
 *       LIVE_URL=https://... node tests/wasm-browser/e2e-live.test.js
 */

'use strict';

const path = require('path');

const LIVE_URL = process.env.LIVE_URL || 'https://crispstrobe.github.io/CrispEmbed/';
const repoRoot = path.join(__dirname, '..', '..');
const PIX2TEX_GT = 'x = \\frac { - b \\pm \\sqrt { b ^ { 2 } - 4 a c } } { 2 a }';

let passed = 0, failed = 0;
const failures = [];
function assert(cond, msg) {
  if (cond) { passed++; console.log(`  ok: ${msg}`); }
  else { failed++; failures.push(msg); console.error(`  FAIL: ${msg}`); }
}
function norm(s) { return (s || '').replace(/\s+/g, ' ').trim(); }

(async () => {
  const { chromium } = require('playwright');
  const browser = await chromium.launch();
  const page = await browser.newPage();
  const pageErrors = [];
  page.on('pageerror', (e) => pageErrors.push(String(e)));

  try {
    console.log(`\n=== Deployed demo: ${LIVE_URL} ===`);
    const resp = await page.goto(LIVE_URL, { waitUntil: 'load', timeout: 60000 });
    assert(resp.ok(), `page loads (HTTP ${resp.status()})`);
    assert(await page.evaluate(() => typeof CrispEmbedOCRWrapper === 'function'),
      'wrapper JS is served');

    console.log('\n=== Load default model from Hugging Face (CORS) ===');
    const defaultUrl = await page.inputValue('#model-url');
    console.log(`  default model URL: ${defaultUrl}`);
    assert(/huggingface\.co\/cstr\/pix2tex-mfr-gguf/.test(defaultUrl),
      'default model URL points at the real HF repo');
    await page.click('#btn-init');
    await page.waitForFunction(
      () => document.getElementById('status').textContent.includes('Model loaded')
         || document.getElementById('status').textContent.startsWith('Error'),
      null, { timeout: 300000 });
    const loadStatus = await page.textContent('#status');
    assert(loadStatus.includes('Model loaded'), `HF model loaded (status: "${loadStatus}")`);

    console.log('\n=== Recognize formula through the deployed UI ===');
    await page.setInputFiles('#file-input',
      path.join(repoRoot, 'tests', 'regression', 'images', 'formula_quadratic.png'));
    await page.waitForFunction(
      () => {
        const s = document.getElementById('status').textContent;
        return s === 'Done.' || s === 'Processing failed.';
      }, null, { timeout: 300000 });
    assert(await page.textContent('#status') === 'Done.', 'recognition completed');
    const text = norm(await page.textContent('#result-text'));
    console.log(`  result: "${text}"`);
    assert(text === norm(PIX2TEX_GT), `deployed demo output equals native ground truth`);
    assert(pageErrors.length === 0, `no uncaught page errors: ${JSON.stringify(pageErrors)}`);
  } finally {
    await browser.close();
  }

  console.log('\n' + '='.repeat(50));
  console.log(`Results: ${passed} passed, ${failed} failed`);
  if (failed > 0) { failures.forEach(f => console.error('  FAIL: ' + f)); process.exit(1); }
  console.log('Deployed demo verified end-to-end!');
})().catch((e) => { console.error(e); process.exit(1); });
