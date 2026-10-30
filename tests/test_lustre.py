"""The lustre files and THREDDS must give the same values (and the same time axis), also across the file boundaries
(AN -> FC_0001 -> ... and between model runs). Needs the model files (a node that mounts /lustre) and THREDDS:
skipped otherwise. Run with `pytest -m lustre -v`."""
import glob

import numpy as np
import pytest
import xarray as xr

from x3d_publisher import convert as cv
from x3d_publisher import model as models

pytestmark = pytest.mark.lustre


@pytest.fixture(scope="module")
def sources():
    lm, tm = models.load("norkyst800_lustre"), models.load("norkyst800")
    if not glob.glob(lm.files):
        pytest.skip(f"no files matching {lm.files}")
    files, desc = cv.open_files(lm, tm.vars)
    try:
        thredds = xr.open_dataset(tm.url, decode_times=True)
    except OSError as e:
        pytest.skip(f"THREDDS unavailable: {e}")
    ref = tm.reference_time
    if cv.to_iso(np.atleast_1d(files[ref].values))[0] != cv.to_iso(np.atleast_1d(thredds[ref].values))[0]:
        pytest.skip("THREDDS is on another model run than the files")
    print(desc)
    return tm, files.sel(tm.select), thredds.sel(tm.select)


def common_window(tm, a, b, back_hours=72):
    ref = np.datetime64(cv.to_iso(np.atleast_1d(a[tm.reference_time].values))[0][:-1], "ns")
    t = np.intersect1d(a.time.values, b.time.values)
    return t[t >= ref - np.timedelta64(back_hours, "h")]


def test_time_axis_is_the_same(sources):
    tm, files, thredds = sources
    first = np.datetime64(cv.to_iso(np.atleast_1d(files[tm.reference_time].values))[0][:-1], "ns") - np.timedelta64(tm.back_hours, "h")
    a, b = files.time.values, thredds.time.values
    np.testing.assert_array_equal(a[a >= first], b[b >= first])


def test_point_series_over_days_match(sources):
    tm, files, thredds = sources
    t = common_window(tm, files, thredds)
    assert len(t) >= 72, "the window must cover several files"
    pt = {tm.y_dim: files.sizes[tm.y_dim] // 2, tm.x_dim: files.sizes[tm.x_dim] // 2}
    for v in tm.vars:
        a = files[v].sel(time=t).isel(pt).values
        b = thredds[v].sel(time=t).isel(pt).values
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-5, equal_nan=True, err_msg=v)


def test_fields_at_file_boundaries_match(sources):
    tm, files, thredds = sources
    t = common_window(tm, files, thredds)
    # a few steps spread over the window, so that AN, several FC pieces and the previous run are all hit
    steps = t[np.unique(np.linspace(0, len(t) - 1, 8).astype(int))]
    box = {tm.y_dim: slice(files.sizes[tm.y_dim] // 2, files.sizes[tm.y_dim] // 2 + 40),
           tm.x_dim: slice(files.sizes[tm.x_dim] // 2, files.sizes[tm.x_dim] // 2 + 40)}
    for s in steps:
        for v in ("u_eastward", "temperature"):
            a = files[v].sel(time=s).isel(box).values
            b = thredds[v].sel(time=s).isel(box).values
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-5, equal_nan=True, err_msg=f"{v} at {s}")
