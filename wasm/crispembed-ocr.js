/**
 * crispembed-ocr.js — High-level JavaScript wrapper for CrispEmbed OCR WASM.
 *
 * Addresses the integration blockers from GitHub issue #31:
 *   1. TextDecoder crash with resizable ArrayBuffer (ALLOW_MEMORY_GROWTH=1)
 *   2. Raw pixel bytes via Canvas API (abstracts _malloc / RGBA extraction)
 *   3. One-shot API with automatic memory management
 *
 * Usage:
 *   <script src="crispembed_ocr.js"></script>
 *   <script src="crispembed-ocr.js"></script>
 *   <script>
 *     const ocr = await CrispEmbedOCRWrapper.create({
 *       modelUrl: 'https://example.com/model.gguf',
 *       onProgress: (p) => console.log(`${(p*100).toFixed(0)}%`),
 *     });
 *     const result = await ocr.recognize(imageElement);
 *     console.log(result.text, result.confidence);
 *     ocr.dispose();
 *   </script>
 *
 * The wrapper accepts any of: HTMLImageElement, HTMLCanvasElement,
 * HTMLVideoElement, ImageBitmap, ImageData, Blob, File, or a URL string.
 *
 * @license MIT
 */

// ---------------------------------------------------------------------------
// TextDecoder fix for resizable ArrayBuffer (Chrome/V8 bug with WASM growth)
// ---------------------------------------------------------------------------
// When Emscripten uses ALLOW_MEMORY_GROWTH=1, the backing ArrayBuffer is
// resizable. V8's TextDecoder.decode() throws:
//   "TypeError: The provided ArrayBuffer value must not be resizable."
// We monkey-patch TextDecoder to copy the buffer before decoding.
(function patchTextDecoder() {
  if (typeof TextDecoder === 'undefined') return;
  const OrigDecoder = TextDecoder;
  const origDecode = OrigDecoder.prototype.decode;
  OrigDecoder.prototype.decode = function (input, options) {
    if (input && input.buffer && input.buffer.resizable) {
      // Copy to a fixed-size ArrayBuffer
      input = new Uint8Array(input);
    }
    return origDecode.call(this, input, options);
  };
})();

// ---------------------------------------------------------------------------
// CrispEmbedOCRWrapper
// ---------------------------------------------------------------------------

class CrispEmbedOCRWrapper {
  /** @type {Object} Emscripten module instance */
  #module = null;
  /** @type {number} Opaque C pointer to OCR context */
  #ctxPtr = 0;
  /** @type {boolean} */
  #disposed = false;

  /**
   * Private constructor — use CrispEmbedOCRWrapper.create() instead.
   * @param {Object} module - initialized Emscripten module
   * @param {number} ctxPtr - pointer from wasm_ocr_init
   */
  constructor(module, ctxPtr) {
    this.#module = module;
    this.#ctxPtr = ctxPtr;
  }

  /**
   * Create and initialize a CrispEmbedOCR instance.
   *
   * @param {Object} options
   * @param {string}   options.modelUrl   - URL to a GGUF model file
   * @param {string}   [options.modelPath='/model.gguf'] - MEMFS path for the model
   * @param {number}   [options.nThreads=1] - number of threads (requires COOP/COEP)
   * @param {number}   [options.maxTokens] - maximum decode tokens
   * @param {function} [options.onProgress] - callback(fraction 0..1)
   * @returns {Promise<CrispEmbedOCRWrapper>}
   */
  static async create({
    modelUrl,
    modelPath = '/model.gguf',
    nThreads = 1,
    maxTokens,
    onProgress,
  } = {}) {
    if (!modelUrl) {
      throw new Error('CrispEmbedOCRWrapper.create() requires a modelUrl');
    }

    onProgress?.(0);

    // 1. Initialize the Emscripten module
    if (typeof CrispEmbedOCR === 'undefined') {
      throw new Error(
        'CrispEmbedOCR not found. Load crispembed_ocr.js before crispembed-ocr.js'
      );
    }
    const module = await CrispEmbedOCR();
    onProgress?.(0.05);

    // 2. Fetch the model with progress tracking
    const response = await fetch(modelUrl);
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
      if (contentLength > 0) {
        onProgress?.(0.05 + 0.75 * (received / contentLength));
      }
    }

    // Concatenate chunks into a single Uint8Array
    const modelBytes = new Uint8Array(received);
    let offset = 0;
    for (const chunk of chunks) {
      modelBytes.set(chunk, offset);
      offset += chunk.length;
    }
    onProgress?.(0.82);

    // 3. Write model to MEMFS
    module.FS.writeFile(modelPath, modelBytes);
    onProgress?.(0.85);

    // 4. Initialize OCR context
    const ctxPtr = module.ccall(
      'wasm_ocr_init', 'number',
      ['string', 'number'],
      [modelPath, nThreads]
    );
    if (!ctxPtr) {
      throw new Error('wasm_ocr_init failed — model may be corrupt or unsupported');
    }

    // 5. Optional: set max tokens
    if (maxTokens != null) {
      module.ccall(
        'wasm_ocr_set_max_tokens', null,
        ['number', 'number'],
        [ctxPtr, maxTokens]
      );
    }

    // Clean up MEMFS to free memory
    try { module.FS.unlink(modelPath); } catch (_) { /* ignore */ }

    const version = module.ccall('wasm_ocr_version', 'string', [], []);
    console.log(`[CrispEmbedOCR] initialized: ${version}`);
    onProgress?.(1.0);

    return new CrispEmbedOCRWrapper(module, ctxPtr);
  }

  /**
   * Recognize text from an image source.
   *
   * Accepts: HTMLImageElement, HTMLCanvasElement, HTMLVideoElement,
   *          ImageBitmap, ImageData, Blob, File, or a URL string.
   *
   * @param {*} source - image source
   * @param {Object} [options]
   * @param {number}  [options.maxWidth]  - downscale if wider (preserves aspect ratio)
   * @param {number}  [options.maxHeight] - downscale if taller
   * @returns {Promise<{text: string, confidence: number}>}
   */
  async recognize(source, options = {}) {
    if (this.#disposed) throw new Error('OCR instance already disposed');

    // Resolve source to ImageData
    const imageData = await this.#toImageData(source, options);
    const { width, height, data } = imageData;

    // Allocate WASM memory for pixel bytes (RGBA) and out_len
    const nBytes = data.length;
    const pixelPtr = this.#module._malloc(nBytes);
    const lenPtr = this.#module._malloc(4);

    try {
      // Copy RGBA pixel data into WASM heap
      this.#module.HEAPU8.set(data, pixelPtr);

      // Call wasm_ocr_recognize_copy (returns malloc'd string)
      const strPtr = this.#module.ccall(
        'wasm_ocr_recognize_copy', 'number',
        ['number', 'number', 'number', 'number', 'number', 'number'],
        [this.#ctxPtr, pixelPtr, width, height, 4, lenPtr]
      );

      if (!strPtr) {
        return { text: '', confidence: 0 };
      }

      // Read the result string
      const text = this.#module.UTF8ToString(strPtr);

      // Get confidence
      const confidence = this.#module.ccall(
        'wasm_ocr_mean_confidence', 'number',
        ['number'],
        [this.#ctxPtr]
      );

      // Free the malloc'd string copy
      this.#module._free(strPtr);

      return { text, confidence };
    } finally {
      this.#module._free(pixelPtr);
      this.#module._free(lenPtr);
    }
  }

  /**
   * Get the WASM module version string.
   * @returns {string}
   */
  get version() {
    return this.#module.ccall('wasm_ocr_version', 'string', [], []);
  }

  /**
   * Free all resources. Must be called when done.
   */
  dispose() {
    if (this.#disposed) return;
    this.#disposed = true;
    try {
      this.#module.ccall('wasm_ocr_free', null, ['number'], [this.#ctxPtr]);
    } catch (e) {
      console.warn('[CrispEmbedOCR] dispose error:', e);
    }
    this.#ctxPtr = 0;
    this.#module = null;
  }

  // ---- Private helpers ----------------------------------------------------

  /**
   * Convert any supported image source to ImageData (RGBA).
   * @param {*} source
   * @param {Object} options
   * @returns {Promise<ImageData>}
   */
  async #toImageData(source, options) {
    // String URL → fetch as blob → createImageBitmap
    if (typeof source === 'string') {
      const resp = await fetch(source);
      source = await resp.blob();
    }

    // Blob/File → ImageBitmap
    if (source instanceof Blob) {
      source = await createImageBitmap(source);
    }

    // ImageData — already done
    if (source instanceof ImageData) {
      return this.#maybeResize(source, options);
    }

    // Drawable sources: HTMLImageElement, HTMLCanvasElement, HTMLVideoElement, ImageBitmap
    let w, h;
    if (source instanceof HTMLVideoElement) {
      w = source.videoWidth;
      h = source.videoHeight;
    } else if (source instanceof HTMLCanvasElement) {
      w = source.width;
      h = source.height;
    } else {
      // HTMLImageElement, ImageBitmap
      w = source.naturalWidth || source.width;
      h = source.naturalHeight || source.height;
    }

    if (!w || !h) {
      throw new Error('Image source has zero dimensions — is it loaded?');
    }

    // Apply max dimensions
    const { drawW, drawH } = this.#calcDimensions(w, h, options);

    // Draw onto an offscreen canvas to extract RGBA
    const canvas = typeof OffscreenCanvas !== 'undefined'
      ? new OffscreenCanvas(drawW, drawH)
      : document.createElement('canvas');
    canvas.width = drawW;
    canvas.height = drawH;

    const ctx = canvas.getContext('2d');
    ctx.drawImage(source, 0, 0, drawW, drawH);
    return ctx.getImageData(0, 0, drawW, drawH);
  }

  /**
   * Optionally resize ImageData if it exceeds max dimensions.
   */
  #maybeResize(imageData, options) {
    const { maxWidth, maxHeight } = options;
    if (!maxWidth && !maxHeight) return imageData;
    const { width: w, height: h } = imageData;
    const { drawW, drawH } = this.#calcDimensions(w, h, options);
    if (drawW === w && drawH === h) return imageData;

    const canvas = typeof OffscreenCanvas !== 'undefined'
      ? new OffscreenCanvas(drawW, drawH)
      : document.createElement('canvas');
    canvas.width = drawW;
    canvas.height = drawH;

    // Put the ImageData on a temp canvas, then draw resized
    const tmp = typeof OffscreenCanvas !== 'undefined'
      ? new OffscreenCanvas(w, h)
      : document.createElement('canvas');
    tmp.width = w;
    tmp.height = h;
    tmp.getContext('2d').putImageData(imageData, 0, 0);

    const ctx = canvas.getContext('2d');
    ctx.drawImage(tmp, 0, 0, drawW, drawH);
    return ctx.getImageData(0, 0, drawW, drawH);
  }

  /**
   * Calculate draw dimensions respecting maxWidth/maxHeight.
   */
  #calcDimensions(w, h, { maxWidth, maxHeight } = {}) {
    let drawW = w, drawH = h;
    if (maxWidth && drawW > maxWidth) {
      drawH = Math.round(drawH * (maxWidth / drawW));
      drawW = maxWidth;
    }
    if (maxHeight && drawH > maxHeight) {
      drawW = Math.round(drawW * (maxHeight / drawH));
      drawH = maxHeight;
    }
    return { drawW, drawH };
  }
}

// Export for both module and global contexts
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { CrispEmbedOCRWrapper };
}
