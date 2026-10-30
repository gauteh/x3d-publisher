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
    assert "rotated_pole" in ds.coords
    assert ds.rotated_pole.attrs["grid_mapping_name"] == ref.rotated_pole.attrs["grid_mapping_name"]
    assert ds.rotated_pole.attrs["proj4"] == ref.rotated_pole.attrs["proj4"]
    assert ds.hs.attrs["grid_mapping"] == "rotated_pole"
    assert ds.attrs["title"] == ref.attrs["title"]


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
    ds["projection_stere"] = ((), np.int32(0), {
        "grid_mapping_name": "polar_stereographic", "straight_vertical_longitude_from_pole": 70.0, "latitude_of_projection_origin": 90.0,
        "standard_parallel": 60.0, "false_easting": 3369600.0, "false_northing": 1844800.0,
        "proj4": "+proj=stere +lat_0=90 +lat_ts=60 +lon_0=70 +x_0=3369600 +y_0=1844800 +a=6378137 +b=6356752.3142 +units=m +no_defs +type=crs",
    })
    ds.attrs.update(title="Norkyst_v3-800m ... 2026-01-01T00:00:00 - 2026-01-01T23:00:00", title_no="norsk tittel", summary="ROMS", license="CC-BY-4.0")
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
        assert got.temperature.attrs["grid_mapping"] == "projection_stere" and got.temperature.attrs["units"] == "Celsius"
        assert got.projection_stere.attrs["grid_mapping_name"] == "polar_stereographic"
    assert xr.open_zarr(run / "ov.zarr", consolidated=True).temperature.shape[0] == 6
    assert xr.open_zarr(run / "ov.zarr", consolidated=True).projection_stere.attrs["proj4"].startswith("+proj=stere")
    assert info["crs"]["grid_mapping"] == "projection_stere" and info["crs"]["attrs"]["straight_vertical_longitude_from_pole"] == 70.0
    assert not list((out / "norkyst800").glob(".*spool*")), "the spool is removed"


def test_explicit_hours_override_the_model_window(tmp_path):
    src = tmp_path / "nk.nc"
    m, _ = norkyst_source(src)
    m = dataclasses.replace(m, back_hours=2, map_shard=(1, 64, 64))
    cv.convert(m, tmp_path / "s", url=str(src), run="r1", hours=3)
    assert len(json.loads((tmp_path / "s" / "norkyst800" / "r1" / "dataset.json").read_text())["times"]) == 3


def test_projection_and_attributes_are_published(tmp_path, ww3, source):
    out = tmp_path / "store"
    cv.convert(ww3, out, url=source, run="r1")
    run = out / "ww3_4km" / "r1"
    info = json.loads((run / "dataset.json").read_text())
    assert info["crs"]["grid_mapping"] == "rotated_pole" and info["crs"]["proj4"].startswith("+proj=ob_tran")
    assert info["crs"]["attrs"]["grid_north_pole_latitude"] == 22.0
    assert info["bbox"] == pytest.approx([0.0, 55.0, 20.0, 70.0])
    assert info["variables"]["hs"] == {"units": "m", "standard_name": "sea_surface_wave_significant_height", "grid_mapping": "rotated_pole"}
    # the same in every layout: the grid-mapping variable and the references to it
    for layout in ("map.zarr", "ts.zarr", "ov.zarr"):
        g = xr.open_zarr(run / layout, consolidated=True)
        assert g.rotated_pole.attrs["grid_mapping_name"] == "rotated_latitude_longitude" and g.rotated_pole.ndim == 0
        assert g.hs.attrs["grid_mapping"] == "rotated_pole" and "_ChunkSizes" not in g.hs.attrs
        assert g.attrs == info["attrs"]
    a = info["attrs"]
    # the source's descriptive attributes stay, the ones that describe the source file do not
    assert a["title"] == "synthetic wave forecast" and a["institution"] == "MET Norway" and a["license"] == "CC-BY-4.0"
    assert a["Conventions"] == "CF-1.6, ACDD-1.3"
    assert "id" not in a and "source file history" not in a["history"]
    assert a["source_url"] == source and a["model_run"] == "2026-01-01T00:00:00Z"
    assert (a["time_coverage_start"], a["time_coverage_end"]) == ("2026-01-01T00:00:00Z", "2026-01-01T02:00:00Z")
    assert a["geospatial_lat_min"] == 55.0 and a["geospatial_lon_max"] == 20.0
    assert a["date_created"] != "2025-01-01T00:00:00Z"


def test_model_overrides_and_drops_global_attributes(tmp_path):
    src = tmp_path / "nk.nc"
    m, _ = norkyst_source(src)
    m = dataclasses.replace(m, map_shard=(1, 64, 64))
    cv.convert(m, tmp_path / "s", url=str(src), run="r1")
    a = json.loads((tmp_path / "s" / "norkyst800" / "r1" / "dataset.json").read_text())["attrs"]
    assert a["title"] == m.attrs["title"] and "title_no" not in a and a["summary"] == "ROMS"


def test_no_grid_mapping_in_the_source_means_no_crs(tmp_path, ww3):
    ds = synthetic(ww3).drop_vars("rotated_pole")
    src = tmp_path / "plain.nc"
    ds.to_netcdf(src)
    cv.convert(ww3, tmp_path / "s", url=str(src), run="r1")
    run = tmp_path / "s" / "ww3_4km" / "r1"
    assert json.loads((run / "dataset.json").read_text())["crs"] is None
    g = xr.open_zarr(run / "ts.zarr", consolidated=True)
    assert "rotated_pole" not in g.variables and "grid_mapping" not in g.hs.attrs


def test_files_source_joins_runs_and_the_newest_wins(tmp_path):
    m, ds = norkyst_source(tmp_path / "full.nc")
    d = tmp_path / "files"
    d.mkdir()
    # run 1: re-analysis (AN) then forecast pieces; run 2 re-analyses steps 2..4 (changed values) and forecasts on
    pieces = {"20251231T00Z": {"AN": (0, 3), "FC_0001": (3, 6)}, "20260101T00Z": {"AN": (2, 5), "FC_0001": (5, 8)}}
    for run, parts in pieces.items():
        for part, (a, b) in parts.items():
            piece = ds.isel(time=slice(a, b))
            if run == "20260101T00Z":
                piece = piece.assign(temperature=piece.temperature + 10)
            piece.to_netcdf(d / f"nk_his_zdepth_{run}_m00_{part}.nc")
    (d / "nk_his_zdepth_20251230T00Z_m00_AN.nc").write_bytes(b"")  # too old: not among the newest runs
    m = dataclasses.replace(m, url=None, files=str(d / "nk_his_zdepth_*_m00_*.nc"), files_runs=2, files_min_age=0,
                            back_hours=None, hours=8, map_shard=(1, 64, 64))
    out = tmp_path / "store"
    assert cv.convert(m, out, run="r1") == 0
    info = json.loads((out / "norkyst800" / "r1" / "dataset.json").read_text())
    assert len(info["times"]) == 8 and info["model_run"] == "2026-01-01T00:00:00Z"
    assert "source_url" not in info["attrs"] and "nk_his_zdepth" in info["attrs"]["history"]
    got = xr.open_zarr(out / "norkyst800" / "r1" / "ts.zarr", consolidated=True).temperature.values
    # chunks never span the levels: only the surface is read from the files
    assert set(cv.open_files(m, m.vars)[0].temperature.chunksizes["depth"]) == {1}
    want = ds.temperature.isel(depth=0).values.copy()
    want[2:] += 10  # steps from the newest run: its AN (2..4) over the old forecast, then its forecast
    np.testing.assert_allclose(got, want, atol=m.scales["temperature"], equal_nan=True)
