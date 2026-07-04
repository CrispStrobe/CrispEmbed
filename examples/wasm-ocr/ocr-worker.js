/**
 * examples/wasm-ocr/ocr-worker.js — runs CrispEmbed OCR inference in a Web
 * Worker so the page never freezes during multi-second WASM compute.
 *
 * Protocol (postMessage):
 *   worker URL:  ocr-worker.js?loader=<emscripten js>   (page picks the build)
 *   -> { cmd:'init', mode:'single'|'pipeline'|'cleanup',
 *        modelUrl?, detUrl?, recUrl?, nThreads }
 *   <- { cmd:'progress', p }            model download / init progress 0..1
 *   <- { cmd:'log', line }              engine stderr (real-time progress)
 *   <- { cmd:'ready' } | { cmd:'error', message }
 *   -> { cmd:'run', width, height, data:ArrayBuffer, options }
 *   <- { cmd:'result', kind, ... }      kind: 'single'|'pipeline'|'cleanup'
 *   -> { cmd:'dispose' }
 */

'use strict';

const LOADER = new URLSearchParams(self.location.search).get('loader') || 'crispembed_ocr.js';

if (self.name === 'em-pthread') {
  // Pthread shim. Emscripten (mainScriptUrlOrBlob was removed) spawns
  // pthread workers from self.location.href — which is THIS script, because
  // the module was importScripts'ed into this worker. Hand the worker
  // straight to the Emscripten loader: evaluated with name "em-pthread" it
  // runs its own pthread bootstrap and takes over message handling.
  importScripts(LOADER);
} else {
  main();
}

function main() {

  let engine = null;
  let mode = null;

  const post = (msg, transfer) => self.postMessage(msg, transfer || []);

  // Forward engine stderr lines to the page — this is the live progress
  // ("ocr_pipeline: recognizing region 3/8", decoder steps, ...).
  self.CRISPEMBED_MODULE_OPTS = {
    print: (line) => post({ cmd: 'log', line: String(line) }),
    printErr: (line) => post({ cmd: 'log', line: String(line) }),
    // Threaded builds live in a subdirectory; resolve absolutely.
    locateFile: (f) => new URL(
      (LOADER.includes('/') ? LOADER.slice(0, LOADER.lastIndexOf('/') + 1) : '') + f,
      self.location.href).href,
  };

  // Import AND instantiate at top level (not inside onmessage): the
  // Emscripten pthread bootstrap deadlocks when the factory is first called
  // from an active message event.
  importScripts(LOADER, 'crispembed-ocr.js');
  self.CRISPEMBED_MODULE_PROMISE = CrispEmbedOCR(Object.assign({}, self.CRISPEMBED_MODULE_OPTS));
  self.CRISPEMBED_MODULE_PROMISE.then(
    () => post({ cmd: 'log', line: 'worker: module instantiated (' + LOADER + ')' }),
    (e) => post({ cmd: 'error', message: 'module init failed: ' + e }));

  self.onmessage = async (ev) => {
    const msg = ev.data;
    try {
      if (msg.cmd === 'init') {
        if (engine) { try { engine.dispose(); } catch (_) {} engine = null; }
        mode = msg.mode;
        const onProgress = (p) => post({ cmd: 'progress', p });
        if (mode === 'single') {
          engine = await CrispEmbedOCRWrapper.create({
            modelUrl: msg.modelUrl, nThreads: msg.nThreads, onProgress });
        } else if (mode === 'pipeline') {
          engine = await CrispEmbedOCRPipeline.create({
            detModelUrl: msg.detUrl, recModelUrl: msg.recUrl,
            nThreads: msg.nThreads, onProgress });
        } else if (mode === 'cleanup') {
          engine = await CrispEmbedScanCleanup.create({
            nThreads: msg.nThreads, onProgress });
        } else {
          throw new Error('unknown mode: ' + mode);
        }
        post({ cmd: 'ready' });

      } else if (msg.cmd === 'run') {
        if (!engine) throw new Error('engine not initialized');
        const imageData = new ImageData(
          new Uint8ClampedArray(msg.data), msg.width, msg.height);
        if (mode === 'single') {
          const { text, confidence } = await engine.recognize(imageData, msg.options || {});
          post({ cmd: 'result', kind: 'single', text, confidence });
        } else if (mode === 'pipeline') {
          const result = await engine.run(imageData, msg.options || {});
          post({ cmd: 'result', kind: 'pipeline', result });
        } else if (mode === 'cleanup') {
          const cleaned = await engine.process(imageData, msg.options || {});
          const buf = cleaned.data.buffer;
          post({ cmd: 'result', kind: 'cleanup',
                 width: cleaned.width, height: cleaned.height, data: buf }, [buf]);
        }

      } else if (msg.cmd === 'dispose') {
        if (engine) { try { engine.dispose(); } catch (_) {} engine = null; }
        mode = null;
      }
    } catch (e) {
      post({ cmd: 'error', message: String(e && e.message || e) });
    }
  };
}
