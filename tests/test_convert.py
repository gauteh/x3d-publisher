"""Offline tests of the converter on a small synthetic dataset."""
import dataclasses
import json

import numpy as np
import pytest
import xarray as xr
import zarr

from conftest import synthetic
from x3d_publisher import convert as cv
from x3d_publisher import model as models


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


def norkyst_source(path, nt=8, ny=40, nx=36, ref="2026-01-01T04:00"):
    """Like the NorKyst aggregate: X/Y + 2-D lat/lon, a depth dimension, a newest model run in the middle of the axis."""
    m = models.load("norkyst800")
    rng = np.random.default_rng(1)
    dims = ("time", "depth", "Y", "X")
    ds = xr.Dataset(
        {v: (dims, rng.uniform(-2, 12, (nt, 3, ny, nx)).astype("float32")) for v in ("u_eastward", "v_northward", "temperature")},
        coords={
            "time": np.datetime64("2026-01-01T00:00", "ns") + np.arange(nt) * np.timedelta64(1, "h"),
            "depth": [0.0, 1.0, 2.0], "Y": np.arange(ny) * 800, "X": np.arange(nx) * 800,
        },
    )
    for v in ("Uwind_eastward", "Vwind_northward"):
        ds[v] = (("time", "Y", "X"), rng.uniform(-10, 10, (nt, ny, nx)).astype("float32"))
    ds["temperature"].values[:, :, :6, :6] = np.nan  # land
    ds["temperature"].attrs.update(units="Celsius", grid_mapping="projection_stere", _ChunkSizes=np.array([1, 1, 4, 4]))
    ds["lon"] = (("Y", "X"), np.add.outer(np.zeros(ny), np.linspace(0, 10, nx)))
    ds["lat"] = (("Y", "X"), np.add.outer(np.linspace(60, 65, ny), np.zeros(nx)))
    ds["forecast_reference_time"] = np.datetime64(ref, "ns")
    ds.to_netcdf(path)
    return m, ds


def test_norkyst_surface_layer_and_window_from_model_run(tmp_path):
    src = tmp_path / "nk.nc"
    m, ds = norkyst_source(src)
    m = dataclasses.replace(m, back_hours=2, map_shard=(1, 64, 64))
    out = tmp_path / "store"
    assert cv.convert(m, out, url=str(src), run="r1") == 0
    run = out / "norkyst800" / "r1"
    info = json.loads((run / "dataset.json").read_text())
    # newest run at 04:00, two hours back: 02:00 .. end of the forecast
    assert info["times"][0] == "2026-01-01T02:00:00Z" and len(info["times"]) == 6
    assert info["dims"] == {"y": "Y", "x": "X"} and info["geo"] == ["lat", "lon"]
    for layout in ("map.zarr", "ts.zarr"):
        got = xr.open_zarr(run / layout, consolidated=True)
        assert "depth" not in got.variables and got.temperature.dims == ("time", "Y", "X")
        np.testing.assert_allclose(
            got.temperature.values, ds.temperature.isel(depth=0, time=slice(2, None)).values, atol=m.scales["temperature"], equal_nan=True
        )
        assert "grid_mapping" not in got.temperature.attrs and got.temperature.attrs["units"] == "Celsius"
    assert xr.open_zarr(run / "ov.zarr", consolidated=True).temperature.shape[0] == 6
    assert not list((out / "norkyst800").glob(".*spool*")), "the spool is removed"


def test_explicit_hours_override_the_model_window(tmp_path):
    src = tmp_path / "nk.nc"
    m, _ = norkyst_source(src)
    m = dataclasses.replace(m, back_hours=2, map_shard=(1, 64, 64))
    cv.convert(m, tmp_path / "s", url=str(src), run="r1", hours=3)
    assert len(json.loads((tmp_path / "s" / "norkyst800" / "r1" / "dataset.json").read_text())["times"]) == 3
