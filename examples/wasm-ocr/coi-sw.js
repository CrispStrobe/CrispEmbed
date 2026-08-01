/**
 * coi-sw.js — minimal cross-origin-isolation service worker.
 *
 * Injects COOP/COEP headers into same-origin responses so the page becomes
 * crossOriginIsolated on static hosts that cannot set headers (GitHub
 * Pages). Required for the multithreaded WASM build (SharedArrayBuffer).
 * The page registers this and reloads once after first install.
 */
'use strict';

self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (event) => event.waitUntil(self.clients.claim()));

self.addEventListener('fetch', (event) => {
  const req = event.request;
  // Pass through requests we must not intercept.
  if (req.cache === 'only-if-cached' && req.mode !== 'same-origin') return;
  // Only same-origin documents/scripts/wasm need the headers (COEP on the
  // document covers CORS-mode subresource fetches). Never proxy large model
  // downloads: WebKit terminates service workers mid-stream ("Service
  // Worker context closed"), killing the fetch.
  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;
  if (!/\.(html|js|mjs|wasm)$/.test(url.pathname) && url.pathname !== '/' && req.destination !== 'document') return;
  event.respondWith((async () => {
    const resp = await fetch(req);
    if (resp.status === 0 || resp.type === 'opaque') return resp;
    const headers = new Headers(resp.headers);
    headers.set('Cross-Origin-Opener-Policy', 'same-origin');
    headers.set('Cross-Origin-Embedder-Policy', 'require-corp');
    return new Response(resp.body, {
      status: resp.status, statusText: resp.statusText, headers,
    });
  })());
});
