"""Offline tests of the converter on a small synthetic dataset."""
import json

import numpy as np
import pytest
import xarray as xr
import zarr

from conftest import synthetic
from x3d_publisher import convert as cv


def test_to_iso():
    assert cv.to_iso(np.array(["2026-10-27T06:00"], dtype="datetime64[ns]")) == ["2026-10-27T06:00:00Z"]


def test_overview_masks_land_and_wraps_direction(ww3):
    ov = cv.overview(synthetic(ww3), ww3)
    f = ww3.ov_factor
    assert ov.sizes["rlat"] == -(-48 // f) and ov.sizes["rlon"] == -(-40 // f)
    # the all-land corner stays masked, the rest is averaged
    assert bool(ov.hs.isnull().isel(time=0, rlat=0, rlon=0))
    assert bool(ov.hs.notnull().isel(time=0, rlat=-1, rlon=-1))
    # 359 deg is averaged as a vector, not arithmetically
    assert float(ov.pdir0.isel(time=0, rlat=-2, rlon=-2)) == pytest.approx(359.0, abs=0.01)
    assert ov.hs.dtype == np.float32


def test_write_roundtrip_is_standard_zarr_v3(tmp_path, ww3):
    sub = synthetic(ww3)[["hs"]]
    cv.write(sub, tmp_path / "map.zarr", {"time": 1, "rlat": 16, "rlon": 16}, ww3, shards={"time": 1, "rlat": 48, "rlon": 48})
    meta = json.loads((tmp_path / "map.zarr" / "hs" / "zarr.json").read_text())
    assert meta["zarr_format"] == 3 and meta["data_type"] == "int16"
    assert meta["codecs"][0]["name"] == "sharding_indexed"
    assert meta["codecs"][0]["configuration"]["index_location"] == "start"
    assert str(meta["fill_value"]) == "-32768"
    back = xr.open_zarr(tmp_path / "map.zarr", consolidated=True)
    # int16 packing: precision is the scale factor; land stays NaN
    np.testing.assert_allclose(back.hs.values, sub.hs.values, atol=ww3.scales["hs"], equal_nan=True)
    assert zarr.open_group(tmp_path / "map.zarr")["hs"].shape == (3, 48, 40)


def test_sidecars(tmp_path, ww3):
    out = tmp_path / "ww3_4km" / "20260101T000000Z"
    out.mkdir(parents=True)
    cv.write_sidecars(synthetic(ww3), out, ww3)
    info = json.loads((out / "dataset.json").read_text())
    assert info["format_version"] == 1 and info["dims"] == {"y": "rlat", "x": "rlon"}
    assert info["name"] == "ww3_4km" and info["run"] == out.name
    assert info["times"] == ["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z", "2026-01-01T02:00:00Z"]
    assert (info["ny"], info["nx"]) == (48, 40)
    # grid.bin = lon mesh then lat mesh, float32, reaching the grid edge
    mesh = np.frombuffer((out / "grid.bin").read_bytes(), "<f4")
    assert mesh.size == 2 * info["mesh_ny"] * info["mesh_nx"]
    lat = mesh[mesh.size // 2:].reshape(info["mesh_ny"], info["mesh_nx"])
    assert lat[-1, 0] == pytest.approx(70.0, abs=1e-4)


def test_convert_end_to_end(tmp_path, ww3, source):
    out = tmp_path / "store"
    assert cv.convert(ww3, out, url=source, run="r1") == 0
    run = out / "ww3_4km" / "r1"
    assert json.loads((out / "ww3_4km" / "latest.json").read_text()) == {"run": "r1"}
    for layout in ("map.zarr", "ov.zarr", "ts.zarr"):
        assert (run / layout / "zarr.json").exists()
    info = json.loads((run / "dataset.json").read_text())
    assert info["model_run"] == "2026-01-01T00:00:00Z" and len(info["times"]) == 3
    # the three layouts hold the same field
    src = xr.open_dataset(source).hs.values
    for layout in ("map.zarr", "ts.zarr"):
        got = xr.open_zarr(run / layout, consolidated=True).hs.values
        np.testing.assert_allclose(got, src, atol=ww3.scales["hs"], equal_nan=True)


def test_convert_skips_up_to_date(tmp_path, ww3, source):
    out = tmp_path / "store"
    cv.convert(ww3, out, url=source, run="r1")
    published = (out / "ww3_4km").as_uri()
    assert cv.convert(ww3, tmp_path / "again", url=source, published=published) == cv.UP_TO_DATE
    assert not (tmp_path / "again").exists()
    # a different time axis converts
    assert cv.convert(ww3, tmp_path / "again", url=source, published=published, hours=2) == 0


def test_x3d_reads_what_was_published(tmp_path, ww3, source):
    xr_x3d = pytest.importorskip("x3d")
    out = tmp_path / "store"
    cv.convert(ww3, out, url=source, run="r1")
    ds = xr.open_dataset(f"x3d+file://{out / 'ww3_4km'}")
    ref = xr.open_dataset(source)
    np.testing.assert_allclose(ds.hs.values, ref.hs.values, atol=ww3.scales["hs"], equal_nan=True)
