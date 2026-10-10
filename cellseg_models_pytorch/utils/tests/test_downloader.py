import gzip
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from cellseg_models_pytorch.utils import Downloader


@pytest.fixture
def download_server():
    payload = bytes(range(256)) * 4

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/redirect":
                self.send_response(302)
                self.send_header("Location", "/payload")
                self.end_headers()
                return
            body = gzip.compress(payload) if self.path == "/gzip" else payload
            self.send_response(200)
            self.send_header("Transfer-Encoding", "chunked")
            if self.path == "/gzip":
                self.send_header("Content-Encoding", "gzip")
            self.end_headers()
            for offset in range(0, len(body), 31):
                chunk = body[offset : offset + 31]
                self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
            self.wfile.write(b"0\r\n\r\n")

        def log_message(self, *args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}", payload
        finally:
            server.shutdown()
            thread.join()


@pytest.mark.parametrize("route", ["payload", "gzip", "redirect"])
def test_downloader_streams_complete_response(tmp_path, download_server, route):
    url, expected = download_server
    Downloader(str(tmp_path)).download(f"{url}/{route}", chunk_size=13)
    assert (tmp_path / route).read_bytes() == expected
