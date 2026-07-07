// Driver worker for the PROXY_TO_PTHREAD OCR no-deadlock test. Matches the real
// demo architecture: the Emscripten module runs HERE (this worker is the
// servicer), the page's main thread stays free. Two rules that avoid the
// Emscripten-in-a-worker bootstrap deadlock:
//   1. em-pthread workers do ONLY importScripts(loader).
//   2. instantiate the module at TOP LEVEL (not inside onmessage).
'use strict';
const params = new URLSearchParams(self.location.search);
const LOADER = params.get('loader') || '/proxy/crispembed_ocr.js';

if (self.name === 'em-pthread') {
  importScripts(LOADER); // pthread bootstrap only
} else {
  importScripts(LOADER, '/crispembed-ocr.js');
  // Forward the module's C-side logs so we can see det/rec progress (slow vs hung).
  self.CRISPEMBED_MODULE_OPTS = {
    print:    (m) => self.postMessage({ log: m }),
    printErr: (m) => self.postMessage({ log: m }),
    // Resolve the .wasm (and pthread worker script) relative to the LOADER's
    // dir (/proxy/) — else the proxy JS glue loads the root single-threaded
    // .wasm and instantiation fails with a LinkError (glue↔wasm mismatch).
    locateFile: (f) => new URL(
      (LOADER.includes('/') ? LOADER.slice(0, LOADER.lastIndexOf('/') + 1) : '') + f,
      self.location.href).href,
  };
  self.CRISPEMBED_MODULE_PROMISE = CrispEmbedOCR(Object.assign({}, self.CRISPEMBED_MODULE_OPTS));

  self.onmessage = async (e) => {
    const { detUrl, recUrl, imageUrl, nThreads } = e.data;
    try {
      const M = await self.CRISPEMBED_MODULE_PROMISE;
      const hasAsync = typeof M._wasm_ocr_pipeline_run_async === 'function';
      const t0 = (self.performance || Date).now();
      const pipe = await CrispEmbedOCRPipeline.create({ detModelUrl: detUrl, recModelUrl: recUrl, nThreads });
      self.postMessage({ log: 'pipeline ready — running OCR…' });
      const bmp = await createImageBitmap(await (await fetch(imageUrl)).blob());
      const result = await pipe.run(bmp);
      self.postMessage({
        ok: true,
        ms: Math.round((self.performance || Date).now() - t0),
        nThreads, usesAsyncEntry: hasAsync,
        isolated: self.crossOriginIsolated,
        nRegions: (result.regions || []).length,
        text: (result.text || (result.regions || []).map(r => r.text).join(' ') || '').slice(0, 200),
        sample: JSON.stringify(result).slice(0, 400)
      });
    } catch (err) {
      self.postMessage({ ok: false, error: (err && err.message) || String(err), stack: (err && err.stack || '').slice(0, 400) });
    }
  };
  self.postMessage({ ready: true });
}
