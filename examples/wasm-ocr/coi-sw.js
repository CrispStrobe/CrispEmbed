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
