/**
 * crispembed-ocr.js — High-level JavaScript wrapper for CrispEmbed OCR WASM.
 *
 * Provides three levels of OCR:
 *   1. Single-model recognition (CrispEmbedOCRWrapper) — formula/text recognition
 *   2. Full pipeline (CrispEmbedOCRPipeline) — detection + recognition + reading order
 *   3. Standalone components — scan cleanup, text detection, layout detection
 *
 * Addresses integration blockers from GitHub issue #31:
 *   - TextDecoder crash with resizable ArrayBuffer (ALLOW_MEMORY_GROWTH=1)
 *   - Raw pixel bytes via Canvas API (abstracts _malloc / RGBA extraction)
 *   - One-shot API with automatic memory management
 *
 * @license MIT
 */

// ---------------------------------------------------------------------------
// TextDecoder fix for resizable ArrayBuffer (Chrome/V8 bug with WASM growth)
// ---------------------------------------------------------------------------
(function patchTextDecoder() {
  if (typeof TextDecoder === 'undefined') return;
  const origDecode = TextDecoder.prototype.decode;
  TextDecoder.prototype.decode = function (input, options) {
    if (input && input.buffer && input.buffer.resizable) {
      input = new Uint8Array(input);
    }
    return origDecode.call(this, input, options);
  };
})();

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// OPFS model cache (origin-private file system). Downloaded GGUFs are kept
// in opfs://crispembed-models/<encoded-url> so revisits skip the network
// entirely. Degrades to plain fetch when OPFS is unavailable (older Safari,
// private browsing). Clear with CrispEmbedModelCache.clear().
// ---------------------------------------------------------------------------

const _opfs = {
  DIR: 'crispembed-models',
  _name(url) { return encodeURIComponent(url); },
  async _dir(create) {
    if (typeof navigator === 'undefined' || !navigator.storage || !navigator.storage.getDirectory) return null;
    const root = await navigator.storage.getDirectory();
    return await root.getDirectoryHandle(this.DIR, { create: !!create });
  },
  async read(url) {
    try {
      const dir = await this._dir(false);
      if (!dir) return null;
      const fh = await dir.getFileHandle(this._name(url));
      const file = await fh.getFile();
      if (file.size === 0) return null;
      return new Uint8Array(await file.arrayBuffer());
    } catch (_) { return null; }
  },
  async write(url, bytes) {
    try {
      const dir = await this._dir(true);
      if (!dir) return false;
      // Ask for persistent storage once — without it Safari evicts
      // script-writable storage after 7 days without interaction.
      try { navigator.storage.persist && navigator.storage.persist(); } catch (_) {}
      const fh = await dir.getFileHandle(this._name(url), { create: true });
      const w = await fh.createWritable();
      await w.write(bytes);
      await w.close();
      return true;
    } catch (_) { return false; }
  },
  async clear() {
    try {
      const root = await navigator.storage.getDirectory();
      await root.removeEntry(this.DIR, { recursive: true });
      return true;
    } catch (_) { return false; }
  },
  async list() {
    const out = [];
    try {
      const dir = await this._dir(false);
      if (!dir) return out;
      for await (const [name, handle] of dir.entries()) {
        const f = await handle.getFile();
        out.push({ url: decodeURIComponent(name), size: f.size });
      }
    } catch (_) {}
    return out;
  },
};

/** Public cache management (page or worker context). */
const CrispEmbedModelCache = {
  clear: () => _opfs.clear(),
  list: () => _opfs.list(),
};

/** @private Fetch a GGUF model with streaming progress + OPFS caching. */
async function _fetchModel(url, onProgress, progressStart, progressEnd) {
  const cached = await _opfs.read(url);
  if (cached) {
    console.log(`[CrispEmbedOCR] model cache hit (OPFS): ${url} (${cached.length} bytes)`);
    if (onProgress) onProgress(progressEnd);
    return cached;
  }
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch model: ${response.status} ${response.statusText}`);
  }
  const contentLength = parseInt(response.headers.get('content-length') || '0', 10);
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    if (contentLength > 0 && onProgress) {
      onProgress(progressStart + (progressEnd - progressStart) * (received / contentLength));
    }
  }
  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.length;
  }
  // Persist before returning (awaited: a fire-and-forget write is killed
  // if the page navigates right after load). Failures (quota, private
  // mode) never block loading.
  if (await _opfs.write(url, bytes)) {
    console.log(`[CrispEmbedOCR] model cached (OPFS): ${url}`);
  }
  return bytes;
}

/** @private Initialize the Emscripten module (shared across all wrappers).
 *  Set globalThis.CRISPEMBED_MODULE_OPTS to pass Emscripten Module options
 *  (e.g. { printErr: line => ... } to capture engine progress logs). */
async function _initModule() {
  // Hosts may pre-instantiate the module (globalThis.CRISPEMBED_MODULE_PROMISE).
  // Required for pthread builds inside Web Workers: the Emscripten pthread
  // bootstrap deadlocks if the factory is first called from inside an active
  // onmessage handler, so workers instantiate at top level instead.
  if (typeof globalThis !== 'undefined' && globalThis.CRISPEMBED_MODULE_PROMISE) {
    return await globalThis.CRISPEMBED_MODULE_PROMISE;
  }
  if (typeof CrispEmbedOCR === 'undefined') {
    throw new Error('CrispEmbedOCR not found. Load crispembed_ocr.js first.');
  }
  const opts = (typeof globalThis !== 'undefined' && globalThis.CRISPEMBED_MODULE_OPTS) || {};
  return await CrispEmbedOCR(Object.assign({}, opts));
}

/** @private Write bytes to MEMFS, creating directories as needed. */
function _writeToMemfs(module, path, bytes) {
  // Create parent directories
  const parts = path.split('/').filter(Boolean);
  let dir = '';
  for (let i = 0; i < parts.length - 1; i++) {
    dir += '/' + parts[i];
    try { module.FS.mkdir(dir); } catch (_) { /* exists */ }
  }
  module.FS.writeFile(path, bytes);
}

/** @private Convert image source to RGBA ImageData. */
async function _toImageData(source, options = {}) {
  if (typeof source === 'string') {
    const resp = await fetch(source);
    source = await resp.blob();
  }
  if (source instanceof Blob) {
    source = await createImageBitmap(source);
  }
  if (source instanceof ImageData) {
    return _maybeResize(source, options);
  }

  let w, h;
  if (typeof HTMLVideoElement !== 'undefined' && source instanceof HTMLVideoElement) {
    w = source.videoWidth; h = source.videoHeight;
  } else if (typeof HTMLCanvasElement !== 'undefined' && source instanceof HTMLCanvasElement) {
    w = source.width; h = source.height;
  } else {
    w = source.naturalWidth || source.width;
    h = source.naturalHeight || source.height;
  }
  if (!w || !h) throw new Error('Image source has zero dimensions');

  const { drawW, drawH } = _calcDimensions(w, h, options);
  const canvas = typeof OffscreenCanvas !== 'undefined'
    ? new OffscreenCanvas(drawW, drawH)
    : document.createElement('canvas');
  canvas.width = drawW; canvas.height = drawH;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(source, 0, 0, drawW, drawH);
  return ctx.getImageData(0, 0, drawW, drawH);
}

function _maybeResize(imageData, options) {
  const { maxWidth, maxHeight } = options;
  if (!maxWidth && !maxHeight) return imageData;
  const { width: w, height: h } = imageData;
  const { drawW, drawH } = _calcDimensions(w, h, options);
  if (drawW === w && drawH === h) return imageData;
  const tmp = typeof OffscreenCanvas !== 'undefined'
    ? new OffscreenCanvas(w, h) : document.createElement('canvas');
  tmp.width = w; tmp.height = h;
  tmp.getContext('2d').putImageData(imageData, 0, 0);
  const canvas = typeof OffscreenCanvas !== 'undefined'
    ? new OffscreenCanvas(drawW, drawH) : document.createElement('canvas');
  canvas.width = drawW; canvas.height = drawH;
  canvas.getContext('2d').drawImage(tmp, 0, 0, drawW, drawH);
  return canvas.getContext('2d').getImageData(0, 0, drawW, drawH);
}

function _calcDimensions(w, h, { maxWidth, maxHeight } = {}) {
  let drawW = w, drawH = h;
  if (maxWidth && drawW > maxWidth) {
    drawH = Math.round(drawH * (maxWidth / drawW)); drawW = maxWidth;
  }
  if (maxHeight && drawH > maxHeight) {
    drawW = Math.round(drawW * (maxHeight / drawH)); drawH = maxHeight;
  }
  return { drawW, drawH };
}

/** @private Get a fresh Uint8Array view of the WASM heap.
 *  module.HEAPU8 requires 'HEAPU8' in EXPORTED_RUNTIME_METHODS (build-wasm.sh
 *  does this); recent Emscripten versions no longer attach heap views to
 *  modularized instances implicitly. */
function _heapU8(module) {
  if (module.HEAPU8) return module.HEAPU8;
  if (module.wasmMemory) return new Uint8Array(module.wasmMemory.buffer);
  throw new Error(
    'CrispEmbedOCR: module has no HEAPU8 view — rebuild the WASM with ' +
    "HEAPU8 in EXPORTED_RUNTIME_METHODS (current build-wasm.sh does this)");
}

/** @private ccall that tolerates suspension (WebGPU builds use JSPI /
 *  Asyncify: any call that reaches GPU work may suspend, which is illegal
 *  through a plain synchronous ccall). With {async:true} Emscripten returns
 *  a Promise on suspending builds and the raw value on plain builds —
 *  `await` normalizes both. Non-suspending getters keep using sync ccall. */
function _acall(module, name, ret, argTypes, args) {
  return Promise.resolve(module.ccall(name, ret, argTypes, args, { async: true }));
}

/** @private Multithreaded, deadlock-free pipeline recognize for the
 *  PROXY_TO_PTHREAD build (build-wasm.sh --proxy-to-pthread). The blocking OCR
 *  runs on the runtime pthread (not this servicer worker), and the JSON result
 *  comes back via Module.__ocrDeliver(reqId, ptr). Resolves to the JSON string.
 *  Only used when _wasm_ocr_pipeline_run_async is exported (the proxy build). */
function _ocrRecognizeProxied(module, ctxPtr, imagePath, full) {
  if (!module.__ocrPending) {
    module.__ocrPending = new Map();
    module.__ocrReqSeq = 0;
    module.__ocrDeliver = (reqId, ptr) => {
      const resolve = module.__ocrPending.get(reqId);
      if (!resolve) { if (ptr) module._free(ptr); return; }
      module.__ocrPending.delete(reqId);
      const json = ptr ? module.UTF8ToString(ptr) : '';
      if (ptr) module._free(ptr);
      resolve(json);
    };
  }
  return new Promise((resolve) => {
    const reqId = ++module.__ocrReqSeq;
    module.__ocrPending.set(reqId, resolve);
    // ccall allocs+frees the path string; the C side strdup's it for the job.
    module.ccall('wasm_ocr_pipeline_run_async', null,
      ['number', 'string', 'number', 'number'],
      [ctxPtr, imagePath, reqId, full ? 1 : 0]);
  });
}

/** @private Copy pixel data into WASM heap, run callback, clean up. */
async function _withPixels(module, imageData, fn) {
  const { width, height, data } = imageData;
  const nBytes = data.length;
  const pixelPtr = module._malloc(nBytes);
  try {
    _heapU8(module).set(data, pixelPtr);
    return await fn(pixelPtr, width, height, 4);
  } finally {
    module._free(pixelPtr);
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// CrispEmbedOCRWrapper — Single-model recognition
// ═══════════════════════════════════════════════════════════════════════════

class CrispEmbedOCRWrapper {
  #module = null;
  #ctxPtr = 0;
  #disposed = false;

  constructor(module, ctxPtr) {
    this.#module = module;
    this.#ctxPtr = ctxPtr;
  }

  static async create({ modelUrl, modelPath = '/model.gguf', nThreads = 1,
                         maxTokens, onProgress } = {}) {
    if (!modelUrl) throw new Error('modelUrl is required');
    onProgress?.(0);

    const module = await _initModule();
    onProgress?.(0.05);

    const modelBytes = await _fetchModel(modelUrl, onProgress, 0.05, 0.80);
    onProgress?.(0.82);

    _writeToMemfs(module, modelPath, modelBytes);
    onProgress?.(0.85);

    const ctxPtr = await _acall(module, 'wasm_ocr_init', 'number',
      ['string', 'number'], [modelPath, nThreads]);
    if (!ctxPtr) throw new Error('wasm_ocr_init failed');

    if (maxTokens != null) {
      module.ccall('wasm_ocr_set_max_tokens', null,
        ['number', 'number'], [ctxPtr, maxTokens]);
    }

    try { module.FS.unlink(modelPath); } catch (_) {}
    console.log(`[CrispEmbedOCR] initialized: ${module.ccall('wasm_ocr_version', 'string', [], [])}`);
    onProgress?.(1.0);

    return new CrispEmbedOCRWrapper(module, ctxPtr);
  }

  async recognize(source, options = {}) {
    if (this.#disposed) throw new Error('disposed');
    const imageData = await _toImageData(source, options);
    const lenPtr = this.#module._malloc(4);
    try {
      return await _withPixels(this.#module, imageData, async (pixelPtr, w, h, ch) => {
        const strPtr = await _acall(this.#module, 'wasm_ocr_recognize_copy', 'number',
          ['number', 'number', 'number', 'number', 'number', 'number'],
          [this.#ctxPtr, pixelPtr, w, h, ch, lenPtr]);
        if (!strPtr) return { text: '', confidence: 0 };
        const text = this.#module.UTF8ToString(strPtr);
        const confidence = this.#module.ccall('wasm_ocr_mean_confidence', 'number',
          ['number'], [this.#ctxPtr]);
        this.#module._free(strPtr);
        return { text, confidence };
      });
    } finally {
      this.#module._free(lenPtr);
    }
  }

  get version() {
    return this.#module.ccall('wasm_ocr_version', 'string', [], []);
  }

  dispose() {
    if (this.#disposed) return;
    this.#disposed = true;
    _acall(this.#module, 'wasm_ocr_free', null, ['number'], [this.#ctxPtr])
      .catch((e) => console.warn('[CrispEmbedOCR] dispose error:', e));
    this.#ctxPtr = 0;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// CrispEmbedOCRPipeline — Full detection + recognition pipeline
// ═══════════════════════════════════════════════════════════════════════════

class CrispEmbedOCRPipeline {
  #module = null;
  #ctxPtr = 0;
  #mode = 'basic'; // 'basic' or 'full'
  #disposed = false;

  constructor(module, ctxPtr, mode) {
    this.#module = module;
    this.#ctxPtr = ctxPtr;
    this.#mode = mode;
  }

  /**
   * Create a basic pipeline (detection + recognition).
   * @param {Object} options
   * @param {string} options.detModelUrl  - URL to DBNet detection GGUF
   * @param {string} options.recModelUrl  - URL to TrOCR recognition GGUF
   * @param {number} [options.nThreads=1]
   * @param {function} [options.onProgress]
   */
  static async create({ detModelUrl, recModelUrl, nThreads = 1, onProgress } = {}) {
    if (!detModelUrl || !recModelUrl) {
      throw new Error('Both detModelUrl and recModelUrl are required');
    }
    onProgress?.(0);
    const module = await _initModule();
    onProgress?.(0.05);

    const detBytes = await _fetchModel(detModelUrl, onProgress, 0.05, 0.40);
    const recBytes = await _fetchModel(recModelUrl, onProgress, 0.40, 0.75);

    const detPath = '/models/det.gguf';
    const recPath = '/models/rec.gguf';
    _writeToMemfs(module, detPath, detBytes);
    _writeToMemfs(module, recPath, recBytes);
    onProgress?.(0.80);

    const ctxPtr = await _acall(module, 'wasm_ocr_pipeline_init', 'number',
      ['string', 'string', 'number'], [detPath, recPath, nThreads]);
    if (!ctxPtr) throw new Error('wasm_ocr_pipeline_init failed');

    try { module.FS.unlink(detPath); } catch (_) {}
    try { module.FS.unlink(recPath); } catch (_) {}
    onProgress?.(1.0);
    console.log('[CrispEmbedOCR] pipeline initialized');

    return new CrispEmbedOCRPipeline(module, ctxPtr, 'basic');
  }

  /**
   * Create a full pipeline with cleanup, routing, and accept-gates.
   * @param {Object} options
   * @param {string} options.detModelUrl
   * @param {string} options.recModelUrl
   * @param {string} [options.nafnetModelUrl] - NAFNet denoising model (optional)
   * @param {string} [options.srModelUrl] - text super-resolution model (optional)
   * @param {boolean} [options.cleanupEnabled=true]
   * @param {boolean} [options.routerEnabled=true]
   * @param {number} [options.nThreads=1]
   * @param {function} [options.onProgress]
   */
  static async createFull({ detModelUrl, recModelUrl, nafnetModelUrl, srModelUrl,
                             cleanupEnabled = true, routerEnabled = true,
                             nThreads = 1, onProgress } = {}) {
    if (!detModelUrl || !recModelUrl) {
      throw new Error('Both detModelUrl and recModelUrl are required');
    }
    onProgress?.(0);
    const module = await _initModule();
    onProgress?.(0.05);

    // Download all models
    const detBytes = await _fetchModel(detModelUrl, onProgress, 0.05, 0.30);
    const recBytes = await _fetchModel(recModelUrl, onProgress, 0.30, 0.55);

    const detPath = '/models/det.gguf';
    const recPath = '/models/rec.gguf';
    let nafnetPath = '';
    let srPath = '';

    _writeToMemfs(module, detPath, detBytes);
    _writeToMemfs(module, recPath, recBytes);

    if (nafnetModelUrl) {
      const nafnetBytes = await _fetchModel(nafnetModelUrl, onProgress, 0.55, 0.65);
      nafnetPath = '/models/nafnet.gguf';
      _writeToMemfs(module, nafnetPath, nafnetBytes);
    }
    if (srModelUrl) {
      const srBytes = await _fetchModel(srModelUrl, onProgress, 0.65, 0.75);
      srPath = '/models/sr.gguf';
      _writeToMemfs(module, srPath, srBytes);
    }
    onProgress?.(0.80);

    const ctxPtr = await _acall(module, 'wasm_ocr_pipeline_full_init', 'number',
      ['string', 'string', 'string', 'string', 'number', 'number', 'number'],
      [detPath, recPath, nafnetPath, srPath,
       cleanupEnabled ? 1 : 0, routerEnabled ? 1 : 0, nThreads]);
    if (!ctxPtr) throw new Error('wasm_ocr_pipeline_full_init failed');

    // Clean up MEMFS
    for (const p of [detPath, recPath, nafnetPath, srPath]) {
      if (p) try { module.FS.unlink(p); } catch (_) {}
    }
    onProgress?.(1.0);
    console.log('[CrispEmbedOCR] full pipeline initialized');

    return new CrispEmbedOCRPipeline(module, ctxPtr, 'full');
  }

  /**
   * Run OCR on an image. Returns structured results with bounding boxes.
   *
   * For 'basic' mode: returns { regions: [{x, y, w, h, confidence, text}] }
   * For 'full' mode:  returns { text, confidence, n_regions, regions: [...] }
   *
   * @param {*} source - image source (see CrispEmbedOCRWrapper.recognize)
   * @param {Object} [options] - { maxWidth, maxHeight }
   */
  async run(source, options = {}) {
    if (this.#disposed) throw new Error('disposed');
    const imageData = await _toImageData(source, options);

    // Write image to MEMFS as raw RGBA (we need a file path for the C API)
    const imgPath = '/tmp/input.png';
    const pngBytes = _encodeSimplePNG(imageData);
    _writeToMemfs(this.#module, imgPath, pngBytes);

    try {
      const full = this.#mode === 'full';
      let jsonStr;
      if (this.#module._wasm_ocr_pipeline_run_async) {
        // PROXY_TO_PTHREAD build: run the blocking pipeline on the runtime
        // pthread so the servicer never blocks — multithreaded, no deadlock.
        jsonStr = await _ocrRecognizeProxied(this.#module, this.#ctxPtr, imgPath, full);
      } else {
        const fnName = full ? 'wasm_ocr_pipeline_full_run' : 'wasm_ocr_pipeline_run';
        const jsonPtr = await _acall(this.#module, fnName, 'number',
          ['number', 'string'], [this.#ctxPtr, imgPath]);
        if (jsonPtr) {
          jsonStr = this.#module.UTF8ToString(jsonPtr);
          this.#module._free(jsonPtr);
        }
      }

      if (!jsonStr) return full
        ? { text: '', confidence: 0, n_regions: 0, regions: [] }
        : { regions: [] };

      const parsed = JSON.parse(jsonStr);
      return full ? parsed : { regions: parsed };
    } finally {
      try { this.#module.FS.unlink(imgPath); } catch (_) {}
    }
  }

  /**
   * Render the last pipeline results to a format string.
   * Only works with 'basic' mode pipeline.
   * @param {string} format - "text", "hocr", or "alto"
   * @param {number} pageWidth
   * @param {number} pageHeight
   * @returns {string}
   */
  render(imagePath, format, pageWidth, pageHeight) {
    if (this.#mode !== 'basic') {
      throw new Error('render() only supported on basic pipeline');
    }
    const ptr = this.#module.ccall('wasm_ocr_render', 'number',
      ['number', 'string', 'number', 'number', 'string'],
      [this.#ctxPtr, imagePath, pageWidth, pageHeight, format]);
    if (!ptr) return '';
    const result = this.#module.UTF8ToString(ptr);
    this.#module._free(ptr);
    return result;
  }

  dispose() {
    if (this.#disposed) return;
    this.#disposed = true;
    const fn = this.#mode === 'full' ? 'wasm_ocr_pipeline_full_free' : 'wasm_ocr_pipeline_free';
    _acall(this.#module, fn, null, ['number'], [this.#ctxPtr])
      .catch((e) => console.warn('[CrispEmbedOCR] dispose error:', e));
    this.#ctxPtr = 0;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// CrispEmbedScanCleanup — Document scan preprocessing
// ═══════════════════════════════════════════════════════════════════════════

class CrispEmbedScanCleanup {
  #module = null;
  #ctxPtr = 0;
  #disposed = false;

  constructor(module, ctxPtr) {
    this.#module = module;
    this.#ctxPtr = ctxPtr;
  }

  /**
   * Create a scan cleanup instance.
   * @param {Object} [options]
   * @param {string} [options.nafnetModelUrl] - NAFNet model for learned denoising (optional)
   * @param {number} [options.nThreads=1]
   * @param {function} [options.onProgress]
   */
  static async create({ nafnetModelUrl, nThreads = 1, onProgress } = {}) {
    onProgress?.(0);
    const module = await _initModule();
    onProgress?.(0.1);

    let modelPath = '';
    if (nafnetModelUrl) {
      const bytes = await _fetchModel(nafnetModelUrl, onProgress, 0.1, 0.8);
      modelPath = '/models/nafnet.gguf';
      _writeToMemfs(module, modelPath, bytes);
    }
    onProgress?.(0.85);

    const ctxPtr = await _acall(module, 'wasm_scan_cleanup_init', 'number',
      ['string', 'number'], [modelPath || '', nThreads]);
    if (!ctxPtr) throw new Error('wasm_scan_cleanup_init failed');

    if (modelPath) try { module.FS.unlink(modelPath); } catch (_) {}
    onProgress?.(1.0);

    return new CrispEmbedScanCleanup(module, ctxPtr);
  }

  /**
   * Clean up a scan image.
   * @param {*} source - image source
   * @param {Object} [options]
   * @param {boolean} [options.deskew=true]
   * @param {boolean} [options.cropBorders=true]
   * @param {boolean} [options.whitenBackground=false]
   * @param {boolean} [options.binarize=false]
   * @returns {Promise<ImageData>} cleaned image
   */
  async process(source, { deskew = true, cropBorders = true,
                          whitenBackground = false, binarize = false } = {}) {
    if (this.#disposed) throw new Error('disposed');
    const imageData = await _toImageData(source);

    const owPtr = this.#module._malloc(4);
    const ohPtr = this.#module._malloc(4);

    try {
      const resultPtr = await _withPixels(this.#module, imageData, async (pixelPtr, w, h, ch) => {
        return await _acall(this.#module, 'wasm_scan_cleanup_process', 'number',
          ['number', 'number', 'number', 'number', 'number',
           'number', 'number', 'number', 'number', 'number', 'number'],
          [this.#ctxPtr, pixelPtr, w, h, ch,
           deskew ? 1 : 0, cropBorders ? 1 : 0,
           whitenBackground ? 1 : 0, binarize ? 1 : 0, owPtr, ohPtr]);
      });

      if (!resultPtr) throw new Error('scan cleanup failed');

      const ow = this.#module.getValue(owPtr, 'i32');
      const oh = this.#module.getValue(ohPtr, 'i32');

      // Output is RGB (3 channels) — convert to RGBA for ImageData
      const rgbSize = ow * oh * 3;
      const rgb = new Uint8Array(_heapU8(this.#module).buffer, resultPtr, rgbSize);
      const rgba = new Uint8ClampedArray(ow * oh * 4);
      for (let i = 0, j = 0; i < ow * oh; i++, j += 3) {
        rgba[i * 4] = rgb[j];
        rgba[i * 4 + 1] = rgb[j + 1];
        rgba[i * 4 + 2] = rgb[j + 2];
        rgba[i * 4 + 3] = 255;
      }

      this.#module.ccall('wasm_scan_cleanup_free_image', null, ['number'], [resultPtr]);
      return new ImageData(rgba, ow, oh);
    } finally {
      this.#module._free(owPtr);
      this.#module._free(ohPtr);
    }
  }

  /**
   * Detect double-page spread gutter position.
   * @returns {Promise<number>} split column, or -1 for single page
   */
  async detectPageSplit(source) {
    const imageData = await _toImageData(source);
    return await _withPixels(this.#module, imageData, (pixelPtr, w, h, ch) => {
      return this.#module.ccall('wasm_scan_cleanup_detect_page_split', 'number',
        ['number', 'number', 'number', 'number'], [pixelPtr, w, h, ch]);
    });
  }

  /**
   * Detect content bounding box (trim blank margins).
   * @returns {Promise<{x0, y0, x1, y1}|null>} bbox or null for blank page
   */
  async contentBbox(source) {
    const imageData = await _toImageData(source);
    const ptrX0 = this.#module._malloc(4);
    const ptrY0 = this.#module._malloc(4);
    const ptrX1 = this.#module._malloc(4);
    const ptrY1 = this.#module._malloc(4);
    try {
      const rc = await _withPixels(this.#module, imageData, (pixelPtr, w, h, ch) => {
        return this.#module.ccall('wasm_scan_cleanup_content_bbox', 'number',
          ['number', 'number', 'number', 'number', 'number', 'number', 'number', 'number'],
          [pixelPtr, w, h, ch, ptrX0, ptrY0, ptrX1, ptrY1]);
      });
      if (rc !== 0) return null;
      return {
        x0: this.#module.getValue(ptrX0, 'i32'),
        y0: this.#module.getValue(ptrY0, 'i32'),
        x1: this.#module.getValue(ptrX1, 'i32'),
        y1: this.#module.getValue(ptrY1, 'i32'),
      };
    } finally {
      this.#module._free(ptrX0);
      this.#module._free(ptrY0);
      this.#module._free(ptrX1);
      this.#module._free(ptrY1);
    }
  }

  dispose() {
    if (this.#disposed) return;
    this.#disposed = true;
    _acall(this.#module, 'wasm_scan_cleanup_free', null, ['number'], [this.#ctxPtr])
      .catch((e) => console.warn('[CrispEmbedOCR] cleanup dispose error:', e));
    this.#ctxPtr = 0;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// CrispEmbedTextDetector — Standalone text detection
// ═══════════════════════════════════════════════════════════════════════════

class CrispEmbedTextDetector {
  #module = null;
  #ctxPtr = 0;
  #disposed = false;

  constructor(module, ctxPtr) {
    this.#module = module;
    this.#ctxPtr = ctxPtr;
  }

  static async create({ modelUrl, nThreads = 1, onProgress } = {}) {
    if (!modelUrl) throw new Error('modelUrl is required');
    onProgress?.(0);
    const module = await _initModule();
    onProgress?.(0.05);
    const bytes = await _fetchModel(modelUrl, onProgress, 0.05, 0.80);
    const modelPath = '/models/det.gguf';
    _writeToMemfs(module, modelPath, bytes);
    onProgress?.(0.85);
    const ctxPtr = await _acall(module, 'wasm_text_det_init', 'number',
      ['string', 'number'], [modelPath, nThreads]);
    if (!ctxPtr) throw new Error('wasm_text_det_init failed');
    try { module.FS.unlink(modelPath); } catch (_) {}
    onProgress?.(1.0);
    return new CrispEmbedTextDetector(module, ctxPtr);
  }

  /**
   * Detect text regions.
   * @param {*} source
   * @param {Object} [options]
   * @param {number} [options.textThreshold=0.3]
   * @param {number} [options.lowThreshold=0.2]
   * @returns {Promise<Array<{x0, y0, x1, y1, confidence}>>}
   */
  async detect(source, { textThreshold = 0.3, lowThreshold = 0.2 } = {}) {
    if (this.#disposed) throw new Error('disposed');
    const imageData = await _toImageData(source);
    const jsonStr = await _withPixels(this.#module, imageData, async (pixelPtr, w, h, ch) => {
      const ptr = await _acall(this.#module, 'wasm_text_det_run', 'number',
        ['number', 'number', 'number', 'number', 'number', 'number', 'number'],
        [this.#ctxPtr, pixelPtr, w, h, ch, textThreshold, lowThreshold]);
      if (!ptr) return '[]';
      const s = this.#module.UTF8ToString(ptr);
      this.#module._free(ptr);
      return s;
    });
    return JSON.parse(jsonStr);
  }

  dispose() {
    if (this.#disposed) return;
    this.#disposed = true;
    _acall(this.#module, 'wasm_text_det_free', null, ['number'], [this.#ctxPtr])
      .catch((e) => console.warn('[CrispEmbedOCR] detector dispose error:', e));
    this.#ctxPtr = 0;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// CrispEmbedLayoutDetector — Document layout analysis (17 classes)
// ═══════════════════════════════════════════════════════════════════════════

class CrispEmbedLayoutDetector {
  #module = null;
  #ctxPtr = 0;
  #disposed = false;

  constructor(module, ctxPtr) {
    this.#module = module;
    this.#ctxPtr = ctxPtr;
  }

  static async create({ modelUrl, nThreads = 1, onProgress } = {}) {
    if (!modelUrl) throw new Error('modelUrl is required');
    onProgress?.(0);
    const module = await _initModule();
    onProgress?.(0.05);
    const bytes = await _fetchModel(modelUrl, onProgress, 0.05, 0.80);
    const modelPath = '/models/layout.gguf';
    _writeToMemfs(module, modelPath, bytes);
    onProgress?.(0.85);
    const ctxPtr = await _acall(module, 'wasm_layout_init', 'number',
      ['string', 'number'], [modelPath, nThreads]);
    if (!ctxPtr) throw new Error('wasm_layout_init failed');
    // Keep model in MEMFS — layout_detect reads from file path
    onProgress?.(1.0);
    return new CrispEmbedLayoutDetector(module, ctxPtr);
  }

  /**
   * Detect document layout regions.
   * @param {*} source - image source
   * @param {Object} [options]
   * @param {number} [options.scoreThreshold=0.3]
   * @returns {Promise<Array<{x1, y1, x2, y2, score, label, label_name}>>}
   */
  async detect(source, { scoreThreshold = 0.3 } = {}) {
    if (this.#disposed) throw new Error('disposed');
    const imageData = await _toImageData(source);

    // Write to MEMFS as PNG
    const imgPath = '/tmp/layout_input.png';
    const pngBytes = _encodeSimplePNG(imageData);
    _writeToMemfs(this.#module, imgPath, pngBytes);

    try {
      const ptr = await _acall(this.#module, 'wasm_layout_detect', 'number',
        ['number', 'string', 'number'], [this.#ctxPtr, imgPath, scoreThreshold]);
      if (!ptr) return [];
      const jsonStr = this.#module.UTF8ToString(ptr);
      this.#module._free(ptr);
      return JSON.parse(jsonStr);
    } finally {
      try { this.#module.FS.unlink(imgPath); } catch (_) {}
    }
  }

  dispose() {
    if (this.#disposed) return;
    this.#disposed = true;
    _acall(this.#module, 'wasm_layout_free', null, ['number'], [this.#ctxPtr])
      .catch((e) => console.warn('[CrispEmbedOCR] layout dispose error:', e));
    this.#ctxPtr = 0;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Minimal PNG encoder (for writing ImageData to MEMFS as a file)
// ═══════════════════════════════════════════════════════════════════════════

function _encodeSimplePNG(imageData) {
  // Minimal uncompressed PNG: IHDR + IDAT (stored/no-compression) + IEND
  const { width, height, data } = imageData;
  const rowBytes = width * 4 + 1; // filter byte + RGBA
  const rawSize = rowBytes * height;

  // Deflate stored blocks (no compression) — each block max 65535 bytes
  const deflateBlocks = [];
  let remaining = rawSize;
  let offset = 0;
  const rawData = new Uint8Array(rawSize);

  // Build raw scanlines with filter byte 0 (None)
  for (let y = 0; y < height; y++) {
    rawData[y * rowBytes] = 0; // filter: None
    for (let x = 0; x < width; x++) {
      const si = (y * width + x) * 4;
      const di = y * rowBytes + 1 + x * 4;
      rawData[di] = data[si];
      rawData[di + 1] = data[si + 1];
      rawData[di + 2] = data[si + 2];
      rawData[di + 3] = data[si + 3];
    }
  }

  // Wrap in zlib (stored) — header: 0x78 0x01, then stored deflate blocks
  const maxBlock = 65535;
  const numBlocks = Math.ceil(rawSize / maxBlock) || 1;
  const zlibSize = 2 + rawSize + numBlocks * 5 + 4; // header + blocks + adler32

  const zlib = new Uint8Array(zlibSize);
  let zp = 0;
  zlib[zp++] = 0x78; // CMF
  zlib[zp++] = 0x01; // FLG

  let adlerA = 1, adlerB = 0;
  let rp = 0;
  while (rp < rawSize) {
    const blockSize = Math.min(maxBlock, rawSize - rp);
    const isLast = (rp + blockSize >= rawSize) ? 1 : 0;
    zlib[zp++] = isLast;
    zlib[zp++] = blockSize & 0xFF;
    zlib[zp++] = (blockSize >> 8) & 0xFF;
    zlib[zp++] = ~blockSize & 0xFF;
    zlib[zp++] = (~blockSize >> 8) & 0xFF;
    for (let i = 0; i < blockSize; i++) {
      const b = rawData[rp++];
      zlib[zp++] = b;
      adlerA = (adlerA + b) % 65521;
      adlerB = (adlerB + adlerA) % 65521;
    }
  }
  // Adler-32
  const adler = ((adlerB << 16) | adlerA) >>> 0;
  zlib[zp++] = (adler >> 24) & 0xFF;
  zlib[zp++] = (adler >> 16) & 0xFF;
  zlib[zp++] = (adler >> 8) & 0xFF;
  zlib[zp++] = adler & 0xFF;

  const zlibData = zlib.subarray(0, zp);

  // CRC32 table
  const crcTable = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    let c = i;
    for (let j = 0; j < 8; j++) c = (c & 1) ? (0xEDB88320 ^ (c >>> 1)) : (c >>> 1);
    crcTable[i] = c;
  }
  function crc32(data, start, len) {
    let crc = 0xFFFFFFFF;
    for (let i = start; i < start + len; i++) {
      crc = crcTable[(crc ^ data[i]) & 0xFF] ^ (crc >>> 8);
    }
    return (crc ^ 0xFFFFFFFF) >>> 0;
  }

  // Build PNG
  const ihdrData = new Uint8Array(13);
  const dv = new DataView(ihdrData.buffer);
  dv.setUint32(0, width);
  dv.setUint32(4, height);
  ihdrData[8] = 8;  // bit depth
  ihdrData[9] = 6;  // color type: RGBA
  ihdrData[10] = 0; // compression
  ihdrData[11] = 0; // filter
  ihdrData[12] = 0; // interlace

  const pngSize = 8 + (12 + 13) + (12 + zlibData.length) + 12;
  const png = new Uint8Array(pngSize);
  let pp = 0;

  // Signature
  const sig = [137, 80, 78, 71, 13, 10, 26, 10];
  for (const b of sig) png[pp++] = b;

  // IHDR chunk
  function writeChunk(type, payload) {
    const len = payload.length;
    png[pp++] = (len >> 24) & 0xFF;
    png[pp++] = (len >> 16) & 0xFF;
    png[pp++] = (len >> 8) & 0xFF;
    png[pp++] = len & 0xFF;
    const typeStart = pp;
    for (let i = 0; i < 4; i++) png[pp++] = type.charCodeAt(i);
    for (let i = 0; i < len; i++) png[pp++] = payload[i];
    const crc = crc32(png, typeStart, 4 + len);
    png[pp++] = (crc >> 24) & 0xFF;
    png[pp++] = (crc >> 16) & 0xFF;
    png[pp++] = (crc >> 8) & 0xFF;
    png[pp++] = crc & 0xFF;
  }

  writeChunk('IHDR', ihdrData);
  writeChunk('IDAT', zlibData);
  writeChunk('IEND', new Uint8Array(0));

  return png.subarray(0, pp);
}

// ═══════════════════════════════════════════════════════════════════════════
// Exports
// ═══════════════════════════════════════════════════════════════════════════

if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    CrispEmbedOCRWrapper,
    CrispEmbedOCRPipeline,
    CrispEmbedScanCleanup,
    CrispEmbedTextDetector,
    CrispEmbedLayoutDetector,
    CrispEmbedModelCache,
  };
}
