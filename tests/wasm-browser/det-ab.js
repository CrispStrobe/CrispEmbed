#!/usr/bin/env node
/**
 * tests/wasm-browser/det-ab.js — A/B the det+rec pipeline: CPU (plain build)
 * vs WebGPU (webgpu build + OCR_DETECT_USE_GPU=1 so DBNet runs on GPU too).
 * Asserts region parity against the CPU result and reports timing.
 *
 * Run: CRISPEMBED_MODELS_DIR=... node tests/wasm-browser/det-ab.js
 */
'use strict';
const path = require('path');
const fs = require('fs');

(async () => {
  process.env.WASM_E2E_WEBGPU = '1';
  const repoRoot = path.join(__dirname, '..', '..');
  const server = require('./server.js');
  const { chromium } = require('playwright');
  const browser = await chromium.launch({ args: ['--enable-unsafe-webgpu','--enable-gpu','--ignore-gpu-blocklist'] });
  const page = await browser.newPage();
  page.on('pageerror', e => console.log('[pageerror]', String(e).slice(0,150)));
  await page.goto('http://127.0.0.1:8093/', { waitUntil: 'load' });
  await page.waitForTimeout(1200);

  const imgB64 = fs.readFileSync(path.join(repoRoot, 'tests/regression/images/scan_strip.png')).toString('base64');

  const runOne = async (useGpu) => {
    await page.evaluate((v) => { window.__DEC_CPU__ = v; }, process.env.AB_DEC_CPU === '1');
    return page.evaluate(async ({ imgB64, useGpu }) => {
    // TextDecoder patch (normally applied by crispembed-ocr.js): growable
    // wasm heaps back UTF8ToString views with a resizable ArrayBuffer,
    // which Chrome's TextDecoder rejects.
    if (!window.__tdPatched) {
      window.__tdPatched = true;
      const orig = TextDecoder.prototype.decode;
      TextDecoder.prototype.decode = function (input, options) {
        if (input && input.buffer && input.buffer.resizable) input = new Uint8Array(input);
        return orig.call(this, input, options);
      };
    }
    const dir = useGpu ? 'webgpu/' : '';
    delete window.CrispEmbedOCR;
    const s = document.createElement('script');
    s.src = dir + 'crispembed_ocr.js?v=' + Math.random();
    document.head.appendChild(s);
    await new Promise(r => { s.onload = r; });
    const logs = [];
    const errLogs = [];
    try {
    const m = await CrispEmbedOCR({
      locateFile: (f) => dir + f,
      printErr: (l) => {
        const t = String(l);
        logs.push(t);
        if (errLogs.length < 20 && /error|Error|invalid|Invalid|SKIPPING/.test(t)) errLogs.push(t);
      },
      preRun: [(mod) => {
        if (useGpu) mod.ENV.OCR_DETECT_USE_GPU = '1';
        if (useGpu && window.__DEC_CPU__) mod.ENV.MATH_OCR_DEC_CPU = '1';
      }],
    });
    const copts = useGpu ? { async: true } : {};
    try { m.FS.mkdir('/models'); } catch (_) {}
    for (const f of ['dbnet-ic15-q4_k.gguf', 'trocr-small-printed-q4_k.gguf']) {
      const buf = await (await fetch('/models/' + f)).arrayBuffer();
      m.FS.writeFile('/models/' + (f.startsWith('dbnet') ? 'det.gguf' : 'rec.gguf'), new Uint8Array(buf));
    }
    const t0 = performance.now();
    const ctx = await m.ccall('wasm_ocr_pipeline_init', 'number', ['string','string','number'],
      ['/models/det.gguf', '/models/rec.gguf', 1], copts);
    if (!ctx) return { error: 'init failed', logs: logs.slice(-6) };
    const initMs = Math.round(performance.now() - t0);

    // write the PNG to MEMFS and run the pipeline on it
    const png = Uint8Array.from(atob(imgB64), c => c.charCodeAt(0));
    m.FS.writeFile('/img.png', png);
    const t1 = performance.now();
    const jp = await m.ccall('wasm_ocr_pipeline_run', 'number', ['number','string'], [ctx, '/img.png'], copts);
    const runMs = Math.round(performance.now() - t1);
    const json = jp ? m.UTF8ToString(jp) : '[]';
    return { regions: JSON.parse(json), initMs, runMs, logs: errLogs.concat(logs.slice(-2)) };
    } catch (e) {
      return { error: String(e && e.stack || e).slice(0, 300), logs: errLogs.concat(logs.slice(-2)) };
    }
  }, { imgB64, useGpu });
  };

  const cpu = process.env.AB_GPU_ONLY === '1'
    ? { regions: [], initMs: 0, runMs: 0, logs: [] }
    : await runOne(false);
  console.log('CPU :', cpu.error || `init ${cpu.initMs} ms, run ${cpu.runMs} ms, ${cpu.regions.length} regions`);
  const gpu = await runOne(true);
  console.log('GPU :', gpu.error || `init ${gpu.initMs} ms, run ${gpu.runMs} ms, ${gpu.regions.length} regions`);
  if (cpu.logs && cpu.error) console.log('cpu logs:', cpu.logs.join(' | ').slice(0,600));
  if (gpu.logs) console.log('gpu logs:', gpu.logs.join(' | ').slice(0,900));

  let parity = !cpu.error && !gpu.error && cpu.regions.length === gpu.regions.length;
  if (parity) {
    for (let i = 0; i < cpu.regions.length; i++) {
      const a = cpu.regions[i], b = gpu.regions[i];
      if (a.text !== b.text) { parity = false; console.log(`region ${i}: "${a.text}" vs "${b.text}"`); }
      for (const k of ['x','y','w','h']) {
        if (Math.abs(a[k] - b[k]) > 2) { parity = false; console.log(`region ${i} ${k}: ${a[k]} vs ${b[k]}`); }
      }
    }
  }
  console.log(parity ? 'PARITY OK' : 'PARITY DIVERGED (see above)');
  if (cpu.regions) console.log('CPU text:', cpu.regions.map(r => r.text).join(' '));
  if (gpu.regions) console.log('GPU text:', gpu.regions.map(r => r.text).join(' '));
  await browser.close(); server.close();
  process.exit(0);
})().catch(e => { console.error(e); process.exit(1); });
