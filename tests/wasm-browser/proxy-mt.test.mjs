// Runtime no-deadlock test for the PROXY_TO_PTHREAD OCR build.
//
// Loads the proxy build (build-wasm-proxy/) on a cross-origin-isolated page and
// runs the full DBNet+TrOCR pipeline with nThreads>1. If PROXY_TO_PTHREAD works,
// the multithreaded pipeline COMPLETES (the plain --threads build deadlocks on
// ggml's pthread_join and would hang until the timeout).
//
// Run:
//   cd tests/wasm-browser && npm install && npx playwright install chromium
//   CRISPEMBED_MODELS_DIR=/path/with/dbnet+trocr node proxy-mt.test.mjs
import { createRequire } from 'module';
const require = createRequire(import.meta.url);

process.env.PORT = process.env.PORT || '8790';
process.env.WASM_COI = '1';        // COOP/COEP → crossOriginIsolated → SharedArrayBuffer
process.env.WASM_E2E_PROXY = '1';  // expose /proxy/ route
const PORT = process.env.PORT;
const BASE = `http://127.0.0.1:${PORT}`;
const TIMEOUT_MS = parseInt(process.env.TIMEOUT_MS || '180000', 10);

const server = require('./server.js');

let chromium;
try { ({ chromium } = require('playwright')); }
catch { console.error('playwright not installed — npm install && npx playwright install chromium'); process.exit(2); }

const browser = await chromium.launch({ args: ['--no-sandbox'] });
const page = await browser.newPage();
page.setDefaultTimeout(TIMEOUT_MS);
page.on('console', m => console.log('  [page]', m.text()));
page.on('pageerror', e => console.log('  [pageerror]', e.message));

let failed = false;
try {
  await page.goto(`${BASE}/proxy-mt.html`, { waitUntil: 'domcontentloaded' });
  console.log('crossOriginIsolated:', await page.evaluate(() => self.crossOriginIsolated));
  await page.waitForFunction(() => window.__mt !== null, { timeout: TIMEOUT_MS });
  const mt = await page.evaluate(() => window.__mt);
  if (mt.ok) {
    console.log(`\n=== SUCCESS ✓ multithreaded pipeline completed (no deadlock) ===`);
    console.log(`  nThreads=${mt.nThreads}  isolated=${mt.isolated}  usesAsyncEntry=${mt.usesAsyncEntry}`);
    console.log(`  ${mt.ms} ms, ${mt.nRegions} regions`);
    console.log(`  result: ${mt.sample}`);
    if (!mt.usesAsyncEntry) { console.error('  ✗ did NOT use the proxied async entry — not the proxy build?'); failed = true; }
    if (mt.nThreads < 2) { console.error('  ✗ nThreads < 2 — not actually multithreaded'); failed = true; }
  } else {
    console.error(`\n=== FAIL ✗ ${mt.error}\n  ${mt.stack || ''}`);
    failed = true;
  }
} catch (e) {
  console.error(`\n=== FAIL ✗ ${e.message} (likely a DEADLOCK — pipeline never completed within ${TIMEOUT_MS}ms) ===`);
  console.error('  last page title:', await page.title().catch(() => '?'));
  failed = true;
} finally {
  await browser.close();
  server.close();
}
process.exit(failed ? 1 : 0);
