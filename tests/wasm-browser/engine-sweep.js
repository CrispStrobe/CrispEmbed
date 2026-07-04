#!/usr/bin/env node
/**
 * tests/wasm-browser/engine-sweep.js — per-engine CPU-vs-WebGPU sweep using
 * the single-model wasm API (wasm_ocr_init auto-detects the architecture).
 * Reports recognized text, timing, and any ops the WebGPU backend skipped.
 *
 * Run: CRISPEMBED_MODELS_DIR=... node tests/wasm-browser/engine-sweep.js [engine...]
 * Needs build-wasm/ + build-wasm-webgpu/ and the models in MODELS_DIR.
 */
'use strict';
const path = require('path');
const fs = require('fs');

const ENGINES = [
  { name: 'pix2tex',    model: 'pix2tex-mfr-q4_k.gguf',              image: 'formula' },
  { name: 'trocr',      model: 'trocr-small-printed-q4_k.gguf',      image: 'word' },
  { name: 'trocr-hw',   model: 'trocr-small-handwritten-q8_0.gguf',  image: 'word' },
  { name: 'parseq',     model: 'parseq-tiny-q8_0.gguf',              image: 'word' },
  { name: 'hmer',       model: 'hmer-hw-q4_k.gguf',                  image: 'formula' },
  { name: 'bttr',       model: 'bttr-hw-q4_k.gguf',                  image: 'formula' },
  { name: 'posformer',  model: 'posformer-hw-q4_k.gguf',             image: 'formula' },
  { name: 'texo',       model: 'texo-distill-q4_k.gguf',             image: 'formula' },
  { name: 'mixtex',     model: 'mixtex-zhen-q4_k.gguf',              image: 'formula' },
  { name: 'ppformulanet', model: 'ppformulanet-l-q4_k.gguf',         image: 'formula' },
  { name: 'texteller',  model: 'texteller-3-q4_k.gguf',              image: 'formula' },
  { name: 'tesseract',  model: 'tesseract-eng-f16.gguf',             image: 'word' },
];

(async () => {
  process.env.WASM_E2E_WEBGPU = '1';
  const MODELS_DIR = process.env.CRISPEMBED_MODELS_DIR || '/mnt/storage/gguf-models';
  const repoRoot = path.join(__dirname, '..', '..');
  const only = process.argv.slice(2);
  const engines = only.length ? ENGINES.filter(e => only.includes(e.name)) : ENGINES;

  const images = {
    formula: fs.readFileSync(path.join(repoRoot, 'tests/regression/images/formula_quadratic.png')).toString('base64'),
    word: fs.readFileSync(path.join(MODELS_DIR, 'word.png')).toString('base64'),
  };

  const server = require('./server.js');
  const { chromium } = require('playwright');
  const browser = await chromium.launch({ args: ['--enable-unsafe-webgpu','--enable-gpu','--ignore-gpu-blocklist'] });

  const runOne = async (eng, useGpu) => {
    const page = await browser.newPage();  // fresh page per run: clean module state
    try {
      await page.goto('http://127.0.0.1:8093/', { waitUntil: 'load' });
      await page.waitForTimeout(800);
      return await page.evaluate(async ({ model, imgB64, useGpu }) => {
        if (!window.__tdPatched) {
          window.__tdPatched = true;
          const orig = TextDecoder.prototype.decode;
          TextDecoder.prototype.decode = function (input, options) {
            if (input && input.buffer && input.buffer.resizable) input = new Uint8Array(input);
            return orig.call(this, input, options);
          };
        }
        const dir = useGpu ? 'webgpu/' : '';
        const s = document.createElement('script');
        s.src = dir + 'crispembed_ocr.js?v=' + Math.random();
        document.head.appendChild(s);
        await new Promise(r => { s.onload = r; });
        const skipped = [], logs = [];
        try {
          const m = await CrispEmbedOCR({
            locateFile: (f) => dir + f,
            printErr: (l) => {
              const t = String(l); logs.push(t);
              const mm = t.match(/SKIPPING unsupported op (\w+)/);
              if (mm && !skipped.includes(mm[1])) skipped.push(mm[1]);
            },
          });
          const copts = useGpu ? { async: true } : {};
          const buf = await (await fetch('/models/' + model)).arrayBuffer();
          m.FS.writeFile('/model.gguf', new Uint8Array(buf));
          const ctx = await m.ccall('wasm_ocr_init', 'number', ['string','number'], ['/model.gguf', 1], copts);
          if (!ctx) return { error: 'init failed', skipped, logs: logs.slice(-4) };

          const blob = await (await fetch('data:image/png;base64,' + imgB64)).blob();
          const bmp = await createImageBitmap(blob);
          const cv = document.createElement('canvas');
          cv.width = bmp.width; cv.height = bmp.height;
          const c2 = cv.getContext('2d');
          c2.drawImage(bmp, 0, 0);
          const id = c2.getImageData(0, 0, bmp.width, bmp.height);
          const ptr = m._malloc(id.data.length);
          m.HEAPU8.set(id.data, ptr);
          const lenPtr = m._malloc(4);
          const t0 = performance.now();
          const sp = await m.ccall('wasm_ocr_recognize_copy', 'number',
            ['number','number','number','number','number','number'],
            [ctx, ptr, bmp.width, bmp.height, 4, lenPtr], copts);
          const ms = Math.round(performance.now() - t0);
          const text = sp ? m.UTF8ToString(sp) : '';
          return { text, ms, skipped };
        } catch (e) {
          return { error: String(e).slice(0, 200), skipped, logs: logs.slice(-4) };
        }
      }, { model: eng.model, imgB64: images[eng.image], useGpu });
    } finally {
      await page.close();
    }
  };

  for (const eng of engines) {
    if (!fs.existsSync(path.join(MODELS_DIR, eng.model))) {
      console.log(`${eng.name}: SKIP (model missing)`);
      continue;
    }
    const cpu = await runOne(eng, false);
    const gpu = await runOne(eng, true);
    const fmt = (r) => r.error ? `ERROR ${r.error}` : `${r.ms} ms "${(r.text||'').slice(0,60)}"`;
    console.log(`\n=== ${eng.name} ===`);
    console.log(`  CPU: ${fmt(cpu)}`);
    console.log(`  GPU: ${fmt(gpu)}${gpu.skipped && gpu.skipped.length ? '  [CPU-skipped ops: ' + gpu.skipped.join(',') + ']' : ''}`);
    if (gpu.error && gpu.logs) console.log('  gpu logs:', gpu.logs.join(' | ').slice(0, 300));
    const match = !cpu.error && !gpu.error &&
      (cpu.text || '').replace(/\s+/g,' ').trim() === (gpu.text || '').replace(/\s+/g,' ').trim();
    console.log(`  verdict: ${gpu.error ? 'GPU-BROKEN' : gpu.skipped.length ? 'PARTIAL-GPU' : 'FULL-GPU'}${!gpu.error ? (match ? ', text MATCHES CPU' : ', text DIFFERS') : ''}`);
  }
  await browser.close(); server.close(); process.exit(0);
})().catch(e => { console.error(e); process.exit(1); });
