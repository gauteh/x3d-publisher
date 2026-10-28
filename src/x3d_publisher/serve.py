"""Static file server with CORS, Range and S3-like cache headers, standing in for the bucket locally (`x3d-publish serve`)."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import os
import re


class H(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Range")
        self.send_header("Access-Control-Expose-Headers", "Content-Range, Content-Length")
        self.send_header("Accept-Ranges", "bytes")
        # runs are immutable; only latest.json changes (same policy as for S3/CDN)
        if self.path.split("?")[0].endswith("latest.json"):
            self.send_header("Cache-Control", "public, max-age=60")
        else:
            self.send_header("Cache-Control", "public, max-age=31536000, immutable")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def do_GET(self):
        rng = self.headers.get("Range")
        path = self.translate_path(self.path)
        m = rng and re.match(r"bytes=(\d*)-(\d*)", rng)
        if not (m and os.path.isfile(path)):
            return super().do_GET()
        size = os.path.getsize(path)
        if m.group(1) == "":
            start, end = max(0, size - int(m.group(2))), size - 1
        else:
            start = int(m.group(1))
            end = min(int(m.group(2)) if m.group(2) else size - 1, size - 1)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            self.wfile.write(f.read(end - start + 1))

    def log_message(self, *a):
        pass


def serve(directory="store", port=8000, handler=H, host="0.0.0.0"):
    ThreadingHTTPServer((host, port), partial(handler, directory=str(directory))).serve_forever()
