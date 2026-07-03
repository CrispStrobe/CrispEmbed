#!/usr/bin/env python3
"""Minimal HTTP server for testing the WASM OCR demo locally.

Serves the current directory with correct MIME types for .wasm files and
the COOP/COEP headers needed for multi-threaded WASM builds.

Usage:
    # 1. Copy (or symlink) the built WASM files into this directory:
    cp ../../build-wasm/crispembed_ocr.{js,wasm} .
    cp ../../wasm/crispembed-ocr.js .

    # 2. Run the server:
    python serve.py          # default port 8080
    python serve.py 3000     # custom port

    # 3. Open http://localhost:8080 in your browser.
"""

import http.server
import sys


class WasmHandler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        '.wasm': 'application/wasm',
        '.js': 'application/javascript',
    }

    def end_headers(self):
        # Required for SharedArrayBuffer (multi-threaded WASM)
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        super().end_headers()


port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
print(f'Serving on http://localhost:{port}')
print('Press Ctrl+C to stop.\n')
http.server.HTTPServer(('', port), WasmHandler).serve_forever()
