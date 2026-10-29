import numpy as np
import pytest
import xarray as xr

from x3d_publisher import model as models


def synthetic(m, nt=3, ny=48, nx=40):
    """A small stand-in for the source dataset, with land in one corner."""
    rng = np.random.default_rng(0)
    hs = rng.uniform(0.2, 5.0, (nt, ny, nx)).astype("float32")
    hs[:, :8, :8] = np.nan
    names = [v for v in dict.fromkeys(m.vars + m.ov_vars) if not v.startswith("pdir")]
    dims = ("time", m.y_dim, m.x_dim)
    ds = xr.Dataset(
        {v: (dims, hs.copy()) for v in names},
        coords={
            "time": np.array(["2026-01-01T00:00", "2026-01-01T01:00", "2026-01-01T02:00"], dtype="datetime64[ns]")[:nt],
            m.y_dim: np.arange(ny, dtype="float64"),
            m.x_dim: np.arange(nx, dtype="float64"),
        },
    )
    ds["pdir0"] = (dims, np.full((nt, ny, nx), 359.0, "float32"))
    ds["pdir1"] = (dims, np.full((nt, ny, nx), 1.0, "float32"))
    ds[m.lon] = ((m.y_dim, m.x_dim), np.add.outer(np.zeros(ny), np.linspace(0, 20, nx)))
    ds[m.lat] = ((m.y_dim, m.x_dim), np.add.outer(np.linspace(55, 70, ny), np.zeros(nx)))
    ds["forecast_reference_time"] = np.datetime64("2026-01-01T00:00", "ns")
    # the projection as in the source (CF grid mapping), the way variables refer to it, and global attributes
    ds["rotated_pole"] = ((), np.int32(0), {
        "grid_mapping_name": "rotated_latitude_longitude", "grid_north_pole_latitude": 22.0,
        "grid_north_pole_longitude": 140.0, "proj4": "+proj=ob_tran +o_proj=longlat +lon_0=-40 +o_lat_p=22 +no_defs",
    })
    for v in ds.data_vars:
        if ds[v].ndim == 3:
            ds[v].attrs.update(grid_mapping="rotated_pole", units="m", standard_name="sea_surface_wave_significant_height", _ChunkSizes=[1, 4, 4])
    ds[m.lat].attrs.update(units="degrees_north", grid_mapping="rotated_pole")
    ds.attrs.update(
        title="synthetic wave forecast", institution="MET Norway", license="CC-BY-4.0", Conventions="CF-1.6, ACDD-1.3",
        history="source file history", date_created="2025-01-01T00:00:00Z", time_coverage_start="2025-01-01T00:00:00Z",
        id="uuid-of-the-source-file",
    )
    return ds


@pytest.fixture(scope="session")
def ww3():
    return models.load("ww3_4km")


@pytest.fixture
def source(tmp_path, ww3):
    """The synthetic dataset as a NetCDF file (anything xarray can open works as a source URL)."""
    path = tmp_path / "source.nc"
    synthetic(ww3).to_netcdf(path)
    return str(path)
