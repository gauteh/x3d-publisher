"""Build a small synthetic store for conformance tests of readers (x3d's `-m store` tests):

    python tests/make_store.py OUT [HOURS]     # -> OUT/ww3_4km/<run>/ + OUT/ww3_4km/latest.json
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from conftest import synthetic  # noqa: E402

from x3d_publisher import convert, model  # noqa: E402


def main(out, nt=72):
    m = model.load("ww3_4km")
    ds = synthetic(m, nt=3, ny=300, nx=200)
    ds = ds.reindex(time=np.datetime64("2026-01-01T00:00", "ns") + np.arange(nt) * np.timedelta64(1, "h"), method="nearest")
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "source.nc"
        ds.to_netcdf(src)
        return convert.convert(m, out, url=str(src), run="ci")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 72))
