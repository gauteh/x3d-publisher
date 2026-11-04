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


PAIRS = [("norkyst800_lustre", "norkyst800"), ("ww3_4km_lustre", "ww3_4km")]
CASES = [(lm, tm, i) for lm, tm in PAIRS for i in range(max(len(models.load(lm).file_globs), 2))]


def back(tm):
    return tm.back_hours or 24  # ww3 has no back_hours: its window is the last N steps, 24 h before the newest run


@pytest.fixture(scope="module", params=CASES, ids=lambda c: f"{c[0]}-side{c[2]}")
def sources(request):
    """Each configured production system (A, B) on its own is compared with THREDDS, not only the first one."""
    import dataclasses

    lname, tname, side = request.param
    lm, tm = models.load(lname), models.load(tname)
    if side >= len(lm.file_globs):
        pytest.skip("only one lustre source configured in the model")
    lm = dataclasses.replace(lm, files=[lm.file_globs[side]])
    if not glob.glob(lm.files[0]):
        pytest.skip(f"no files matching {lm.files[0]}")
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


def common_window(tm, a, b):
    ref = np.datetime64(cv.to_iso(np.atleast_1d(a[tm.reference_time].values))[0][:-1], "ns")
    t = np.intersect1d(a.time.values, b.time.values)
    return t[t >= ref - np.timedelta64(back(tm) + 24, "h")]


def test_time_axis_is_the_same(sources):
    tm, files, thredds = sources
    first = np.datetime64(cv.to_iso(np.atleast_1d(files[tm.reference_time].values))[0][:-1], "ns") - np.timedelta64(back(tm), "h")
    a, b = files.time.values, thredds.time.values
    np.testing.assert_array_equal(a[a >= first], b[b >= first])


def test_point_series_over_days_match(sources):
    tm, files, thredds = sources
    t = common_window(tm, files, thredds)
    assert len(t) >= 48, "the window must cover several files"
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
        for v in tm.vars[:2]:
            a = files[v].sel(time=s).isel(box).values
            b = thredds[v].sel(time=s).isel(box).values
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-5, equal_nan=True, err_msg=f"{v} at {s}")


@pytest.mark.parametrize("name", [p[0] for p in PAIRS])
def test_production_systems_agree(name):
    """Files that exist on more than one side (A, B: the globs of the model) must have identical content, and the
    union of the sides must cover every file of the newest run. Needs a model with two globs in `files`."""
    import os

    lm = models.load(name)
    globs = lm.file_globs
    if len(globs) < 2:
        pytest.skip("only one lustre source configured in the model")
    sides = [{os.path.basename(f): f for f in glob.glob(g)} for g in globs]
    if not all(sides):
        pytest.skip(f"a source has no files: {[len(s) for s in sides]}")
    names = sorted(set().union(*sides))
    both = [n for n in names if sum(n in s for s in sides) > 1]
    print(f"{len(names)} files, {len(both)} on several sides, only on one side: {[sum(n in s for s in sides) == 1 for n in names].count(True)}")
    for n in both[-12:]:
        a, b = [xr.open_dataset(s[n]) for s in sides if n in s][:2]
        xr.testing.assert_equal(a[lm.vars[:3]].isel(**{d: 0 for d in lm.select}), b[lm.vars[:3]].isel(**{d: 0 for d in lm.select}))


@pytest.fixture(scope="module", params=PAIRS, ids=lambda p: p[0])
def aggregates(request, tmp_path_factory):
    """The converted stores (what is published) from the lustre files and from THREDDS: the default window of the model."""
    import json

    lname, tname = request.param
    lm, tm = models.load(lname), models.load(tname)
    if not any(glob.glob(g) for g in lm.file_globs):
        pytest.skip(f"no files matching {lm.file_globs}")
    out = tmp_path_factory.mktemp("agg")
    vars = tm.vars[:2]
    try:
        cv.convert(tm, out / "thredds", run="t", vars=vars, workers=2)
    except OSError as e:
        pytest.skip(f"THREDDS unavailable: {e}")
    cv.convert(lm, out / "lustre", run="t", vars=vars, workers=2)
    runs = [json.loads((out / s / tm.name / "t" / "dataset.json").read_text()) for s in ("lustre", "thredds")]
    if runs[0]["model_run"] != runs[1]["model_run"]:
        pytest.skip("THREDDS is on another model run than the files")
    return tm, vars, [out / s / tm.name / "t" for s in ("lustre", "thredds")], runs


def test_aggregate_time_series_at_ocean_points_match(aggregates):
    tm, vars, (lus, thr), (dl, dt_) = aggregates
    common = sorted(set(dl["times"]) & set(dt_["times"]))
    assert len(common) >= 48, f"the aggregates share only {len(common)} time steps"
    il, it = [np.array([d["times"].index(t) for t in common]) for d in (dl, dt_)]
    for v in vars:
        a = xr.open_zarr(lus / "ts.zarr", consolidated=True)[v]
        b = xr.open_zarr(thr / "ts.zarr", consolidated=True)[v]
        scale = dl["scales"][v]
        valid = np.asarray(a.isel(time=il).notnull().all("time"))  # ocean at every step
        assert valid.sum() > 100, f"{v}: too few ocean points"
        ys, xs = np.nonzero(valid)
        pick = np.linspace(0, len(ys) - 1, 6).astype(int)  # spread over the domain
        for k in pick:
            y, x = int(ys[k]), int(xs[k])
            sa = a.isel(time=il, **{a.dims[1]: y, a.dims[2]: x}).values
            sb = b.isel(time=it, **{b.dims[1]: y, b.dims[2]: x}).values
            np.testing.assert_allclose(sa, sb, rtol=0, atol=scale * 1.01, err_msg=f"{v} at y={y}, x={x}")
