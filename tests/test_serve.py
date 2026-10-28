import threading
from http.server import ThreadingHTTPServer
from functools import partial

import pytest
import requests

from x3d_publisher import serve


@pytest.fixture
def url(tmp_path):
    (tmp_path / "latest.json").write_text('{"run": "r"}')
    (tmp_path / "r").mkdir()
    (tmp_path / "r" / "blob.bin").write_bytes(bytes(range(256)))
    srv = ThreadingHTTPServer(("127.0.0.1", 0), partial(serve.H, directory=str(tmp_path)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_range_cors_and_cache(url):
    r = requests.get(f"{url}/r/blob.bin", headers={"Range": "bytes=10-19", "Origin": "http://x"})
    assert r.status_code == 206 and r.content == bytes(range(10, 20))
    assert r.headers["Content-Range"] == "bytes 10-19/256"
    assert r.headers["Access-Control-Allow-Origin"] == "*"
    assert "immutable" in r.headers["Cache-Control"]
    assert requests.get(f"{url}/r/blob.bin", headers={"Range": "bytes=-4"}).content == bytes([252, 253, 254, 255])
    # only latest.json is mutable
    assert "immutable" not in requests.get(f"{url}/latest.json").headers["Cache-Control"]
    assert requests.options(f"{url}/r/blob.bin").status_code == 204
