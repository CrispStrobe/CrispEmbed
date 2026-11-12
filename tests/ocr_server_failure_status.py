"""Hosted model-free HTTP controls using a compiled production server.

Usage: python tests/ocr_server_failure_status.py --server /path/crispembed-server
Missing local GGUF paths exercise lazy unified failure and legacy unknown blank
without downloading weights or running model inference.
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request


def request(base, route, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(base + route, data=data,
                                 headers={'Content-Type': 'application/json'})
    try:
        response = urllib.request.urlopen(req, timeout=10)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        return response.status, response.headers.get_content_type(), response.read().decode()


@contextmanager
def server(binary, directory, engine):
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    missing = directory / 'deliberately-missing.gguf'
    with (directory / (engine + '.log')).open('wb') as log:
        process = subprocess.Popen([str(binary), '--host', '127.0.0.1', '--port', str(port),
                                    '--image-root', str(directory), '--ocr-engine', engine,
                                    '--ocr-rec', str(missing), '-t', '1'],
                                   stdout=log, stderr=subprocess.STDOUT)
        try:
            base = 'http://127.0.0.1:' + str(port)
            deadline = time.monotonic() + 20
            while True:
                if process.poll() is not None:
                    raise AssertionError('server exited: ' + (directory / (engine + '.log')).read_text())
                try:
                    if request(base, '/health')[0] == 200:
                        break
                except (OSError, urllib.error.URLError):
                    pass
                if time.monotonic() >= deadline:
                    raise AssertionError('server startup timed out')
                time.sleep(.1)
            yield base
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--server', required=True)
    binary = Path(parser.parse_args().server).resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix='ocr-http-control-') as temp:
        directory = Path(temp)
        image = directory / 'blank.ppm'
        image.write_bytes(b'P6\n32 32\n255\n' + b'\xff' * (32 * 32 * 3))
        with server(binary, directory, 'unified') as base:
            # Repeat to ensure a failed request never creates a successful stale snapshot.
            for _ in range(2):
                for extra in [{}, {'response_format': 'review-json'}]:
                    status, kind, body = request(base, '/ocr/pipeline', {'image': str(image), **extra})
                    assert status == 500 and kind == 'application/json', (status, kind, body)
                    assert 'initialization failed' in json.loads(body)['error'], body
                status, kind, body = request(base, '/ocr/document', {'images': [str(image)], 'format': 'hocr'})
                assert status == 500 and kind == 'application/json', (status, kind, body)
                assert 'initialization failed' in json.loads(body)['error'], body
        # Legacy adapters without explicit status retain their existing non-NULL
        # blank output contract. HTTP must not infer failure from zero regions.
        with server(binary, directory, 'got') as base:
            status, kind, body = request(base, '/ocr/pipeline', {'image': str(image)})
            output = json.loads(body)
            assert status == 200 and output['text'] == '' and output['n_regions'] == 0, body
            status, kind, body = request(base, '/ocr/document', {'images': [str(image)], 'format': 'hocr'})
            assert status == 200 and kind == 'text/html', (status, kind, body)
    print(json.dumps({'lazy_failure_http_500': True, 'review_failure_http_500': True,
                      'document_failure_http_500': True, 'legacy_blank_http_200': True,
                      'model_inference_executed': False}, sort_keys=True))


if __name__ == '__main__':
    main()
