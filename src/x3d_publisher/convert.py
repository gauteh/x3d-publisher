"""Convert a source dataset (THREDDS OPeNDAP, or anything xarray opens) into the cloud-optimized Zarr v3 layouts of x3d.

  <out>/<name>/latest.json                  {"run": ...}; the only mutable object (short cache TTL)
  <out>/<name>/<run>/map.zarr   shards of one time step each (time=1, whole grid) holding 256x256
                                chunks -> one request per time step; all-fill (land) chunks are
                                absent from the shard index and cost no bytes
  <out>/<name>/<run>/ov.zarr    1/ov_factor resolution, (time=ov_tchunk, whole grid) chunks
                                -> a few requests give every time step: instant scrubbing
  <out>/<name>/<run>/ts.zarr    shards of (all time, 128x128) holding (all time, 16x16) chunks
                                -> a point time series is two small range requests
  <out>/<name>/<run>/dataset.json  time axis, layout parameters and viewer geometry
  <out>/<name>/<run>/grid.bin   decimated lon/lat (float32) mesh used to georeference the grid

Everything below <run>/ is immutable (cache forever); publish a new run, then swap latest.json.
Only standard Zarr v3 codecs are used (bytes + zstd; sharding_indexed with the index at the start), so
xarray/zarr-python, zarrs and others read it directly. See x3d's docs/format.md.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import shutil
import time
import urllib.request
from pathlib import Path

import dask
import dask.array as da
import numpy as np
import xarray as xr
import zarr
from zarr.codecs import BytesCodec, ShardingCodec, ShardingCodecIndexLocation, ZstdCodec

from .model import FORMAT_VERSION, Model

log = logging.getLogger("x3d_publisher.convert")

UP_TO_DATE = 3  # exit status of `convert` when the published store already has these time steps


def write(ds: xr.Dataset, path, chunks: dict, m: Model, shards: dict | None = None) -> None:
    """Write `ds` as a Zarr v3 group. `chunks`/`shards` map dimension name -> size."""
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    comp = ZstdCodec(level=m.zstd_level)
    ds = ds.chunk(shards or chunks)
    for v in ds.variables.values():
        v.encoding.clear()
    enc = {}
    for v in ds.data_vars:
        shape, dims = ds[v].shape, ds[v].dims
        ch = tuple(min(chunks[d], n) for d, n in zip(dims, shape))
        e = {"chunks": ch, "compressors": (comp,)}
        if shards:
            # index at the start of the shard: a client reads it with a plain `bytes=0-N` range
            # (CORS-safelisted, unlike suffix ranges), without knowing the object size
            e = {
                # shard = whole number of inner chunks covering the requested shard extent
                "chunks": tuple(-(-min(shards[d], -(-n // c) * c) // c) * c for d, n, c in zip(dims, shape, ch)),
                "compressors": None,
                "serializer": ShardingCodec(
                    chunk_shape=ch, codecs=[BytesCodec(), comp], index_location=ShardingCodecIndexLocation.start
                ),
            }
        if v in m.scales:
            # array fill_value must equal _FillValue so absent chunks decode as fill, not 0
            e.update(dtype="int16", scale_factor=m.scales[v], _FillValue=-32768, fill_value=-32768)
        enc[v] = e
    ds.to_zarr(path, mode="w", zarr_format=3, consolidated=True, encoding=enc)


def overview(ds: xr.Dataset, m: Model) -> xr.Dataset:
    """Block-mean to 1/ov_factor; a cell needs at least half valid (sea) samples.

    Directions (pdir*) are averaged as unit vectors so the 0/360 wrap is handled.
    """
    f = m.ov_factor
    win = {m.y_dim: f, m.x_dim: f}

    def count(da):
        return da.notnull().reset_coords(drop=True).coarsen(win, boundary="pad").sum()

    def coarse(da, n):
        s = da.reset_coords(drop=True).coarsen(win, boundary="pad").sum().astype("float64")
        return (s / n.where(n > 0)).where(n >= f * f // 2)

    out = {}
    for v in m.ov_vars:
        n = count(ds[v])
        if v.startswith("pdir"):
            r = np.deg2rad(ds[v].astype("float64"))
            out[v] = xr.apply_ufunc(
                lambda a, b: np.rad2deg(np.arctan2(a.astype("float64"), b.astype("float64"))) % 360,
                coarse(np.sin(r), n), coarse(np.cos(r), n),
            )
        else:
            out[v] = coarse(ds[v], n)
        out[v] = out[v].astype("float32")
        out[v].attrs = {k: ds[v].attrs[k] for k in ("standard_name", "long_name", "units") if k in ds[v].attrs}
    return xr.Dataset(out, coords={"time": ds.time}).drop_vars([m.y_dim, m.x_dim])


def to_iso(t) -> list[str]:
    return [str(x)[:19] + "Z" for x in np.asarray(t).astype("datetime64[s]")]


def published_times(base: str) -> list[str] | None:
    """Time axis of the run that latest.json points to, or None if unavailable."""

    def get(url):
        with urllib.request.urlopen(urllib.request.Request(url, headers={"Cache-Control": "no-cache"}), timeout=30) as r:
            return json.load(r)

    try:
        base = base.rstrip("/")
        return get(f"{base}/{get(base + '/latest.json')['run']}/dataset.json")["times"]
    except Exception as e:
        log.info("no published store to compare with (%s)", e)
        return None


def write_sidecars(ds: xr.Dataset, out: Path, m: Model) -> None:
    ny, nx = ds.sizes[m.y_dim], ds.sizes[m.x_dim]
    s = m.mesh_stride
    iy = np.minimum(np.arange(-(-(ny - 1) // s) + 1) * s, ny - 1)
    ix = np.minimum(np.arange(-(-(nx - 1) // s) + 1) * s, nx - 1)
    lon = ds[m.lon].values[np.ix_(iy, ix)].astype("<f4")
    lat = ds[m.lat].values[np.ix_(iy, ix)].astype("<f4")
    (out / "grid.bin").write_bytes(lon.tobytes() + lat.tobytes())
    (out / "dataset.json").write_text(json.dumps({
        "format_version": FORMAT_VERSION,
        "name": out.parent.name, "run": out.name, "model_run": ds.attrs.get("model_run"),
        "times": to_iso(ds.time.values), "ny": ny, "nx": nx, "dims": {"y": m.y_dim, "x": m.x_dim},
        "geo": [m.lat, m.lon], "mesh_stride": s, "mesh_ny": len(iy), "mesh_nx": len(ix),
        "ov_factor": m.ov_factor, "ov_tchunk": m.ov_tchunk, "ov_vars": m.ov_vars, "scales": m.scales,
    }))


def retry(fn, what="request", tries=4):
    """THREDDS (shared, rate-limited) fails transiently; back off and retry."""
    for i in range(tries):
        try:
            return fn()
        except OSError as e:
            if i == tries - 1:
                raise
            wait = 5 * 2**i
            log.warning("%s failed (%s); retry %d/%d in %ds", what, e, i + 1, tries - 1, wait)
            time.sleep(wait)


def batches(ds: xr.Dataset, step: int = 12):
    """The dataset a few time steps at a time (progress is visible, retries are cheap and memory is bounded)."""
    n, t0 = ds.sizes["time"], time.time()
    for i in range(0, n, step):
        yield i, retry(ds.isel(time=slice(i, i + step)).load, f"load steps {i}-{min(i + step, n) - 1}")
        done = min(i + step, n)
        log.info("downloaded %d/%d time steps (%.0f%%, %.0fs elapsed)", done, n, 100 * done / n, time.time() - t0)


FILL = -32768
# attributes that point at other variables of the source file, which are not in the store
DROP_ATTRS = ("grid_mapping", "coordinates", "bounds")


def pack(da: xr.DataArray, scale: float) -> np.ndarray:
    """int16 exactly as xarray would encode it when writing (scale_factor, _FillValue)."""
    v = xr.Variable(da.dims, da.values, {}, {"dtype": "int16", "scale_factor": scale, "_FillValue": FILL})
    return xr.conventions.encode_cf_variable(v).values


def unpack(a: np.ndarray, scale: float) -> np.ndarray:
    return np.where(a == FILL, np.float32("nan"), a.astype("float32") * np.float32(scale))


class Spool:
    """The packed (int16) time steps on disk while they are downloaded, so that memory stays bounded and each layout
    is written from local files rather than THREDDS."""

    def __init__(self, directory: Path, names: list[str], n: int, ny: int, nx: int, m: Model):
        directory.mkdir(parents=True, exist_ok=True)
        self.m, self.n, self.names = m, n, names
        self.arrays = {
            v: np.lib.format.open_memmap(directory / f"{v}.npy", mode="w+", dtype="int16", shape=(n, ny, nx)) for v in names
        }
        self.attrs: dict[str, dict] = {}
        self.coords: dict | None = None
        self.geo: dict = {}

    def add(self, i: int, part: xr.Dataset) -> None:
        m = self.m
        for v in self.names:
            self.arrays[v][i : i + part.sizes["time"]] = pack(part[v], self.m.scales[v])
            self.attrs.setdefault(v, {k: a for k, a in part[v].attrs.items() if k not in DROP_ATTRS and not k.startswith("_")})
        if self.coords is None:
            self.coords = {k: part[k].variable for k in (m.y_dim, m.x_dim) if k in part.coords}
            self.geo = {k: part[k].variable for k in (m.lat, m.lon)}

    def flush(self) -> None:
        for a in self.arrays.values():
            a.flush()

    def dataset(self, times: np.ndarray, chunks: dict) -> xr.Dataset:
        """Lazy float32 dataset (decoded on the fly) with dask chunks `chunks` (dimension -> size)."""
        m = self.m
        dims = ("time", m.y_dim, m.x_dim)
        data = {}
        for v, a in self.arrays.items():
            ch = tuple(min(chunks[d], n) for d, n in zip(dims, a.shape))
            lazy = da.from_array(a, chunks=ch, name=False).map_blocks(unpack, m.scales[v], dtype="float32")
            data[v] = xr.DataArray(lazy, dims=dims, attrs=self.attrs[v])
        for k, var in self.geo.items():
            data[k] = xr.DataArray(var)
        return xr.Dataset(data, coords={"time": times, **self.coords})


def timed(label, fn, *args, **kw):
    t0 = time.time()
    log.info("writing %s ...", label)
    fn(*args, **kw)
    log.info("wrote %s in %.1fs", label, time.time() - t0)


def convert(
    m: Model, out, *, url: str | None = None, name: str | None = None, start: str | None = None,
    hours: int | None = None, run: str | None = None, vars: list[str] | None = None,
    published: str | None = None, workers: int = 4,
) -> int:
    """Download and write `<out>/<name>/<run>/`; returns 0, or UP_TO_DATE if `published` has the same time steps."""
    # shared login nodes cap threads per user: keep dask/zarr/asyncio pools small
    dask.config.set(scheduler="threads", num_workers=workers)
    zarr.config.set({"async.concurrency": workers, "threading.max_workers": workers})
    name, url, vars = name or m.name, url or m.url, vars or m.vars
    log.info("opening %s", url)
    ds = retry(lambda: xr.open_dataset(url, decode_times=True), "open dataset")
    needed = list(dict.fromkeys(vars + m.ov_vars))
    log.info("source has %d time steps, last %s", ds.sizes["time"], ds.time.values[-1])
    log.debug("dataset dims %s; variables %s", dict(ds.sizes), list(ds.data_vars))
    ref = m.reference_time
    model_run = to_iso(np.atleast_1d(ds[ref].values))[0] if ref and ref in ds else None
    log.info("latest model run: %s", model_run)
    ds = ds[needed + [m.lat, m.lon]]
    if m.select:
        ds = ds.sel(m.select).drop_vars([d for d in m.select if d in ds.coords or d in ds.dims], errors="ignore")
    if start:
        ds = ds.sel(time=slice(start, None)).isel(time=slice(0, hours or m.hours))
    elif not hours and m.back_hours is not None and model_run:
        first = np.datetime64(model_run[:-1], "ns") - np.timedelta64(m.back_hours, "h")
        ds = ds.sel(time=slice(first, None))
    else:
        ds = ds.isel(time=slice(-(hours or m.hours), None))
    if published and published_times(published) == to_iso(ds.time.values):
        log.info("published store already has these %d time steps (%s .. %s); nothing to do",
                 ds.sizes["time"], ds.time.values[0], ds.time.values[-1])
        return UP_TO_DATE
    n, y, x = ds.sizes["time"], m.y_dim, m.x_dim
    log.info("downloading %d variables x %d time steps (%s .. %s), ~%.0f MB",
             len(needed), n, ds.time.values[0], ds.time.values[-1], ds.nbytes / 1e6)

    root = Path(out) / name
    run = run or dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = root / run
    path.mkdir(parents=True, exist_ok=True)
    spool_dir = root / f".{run}.spool"
    try:
        spool = Spool(spool_dir, vars, n, ds.sizes[y], ds.sizes[x], m)
        per_step = max(1, ds.nbytes // n)
        ov = []
        for i, part in batches(ds, step=max(1, min(12, int(400e6 // per_step)))):
            # the overview is made from the unquantised values, as the exact layouts are
            ov.append(overview(part, m))
            spool.add(i, part)
        spool.flush()
        times = ds.time.values
        sk = xr.Dataset(spool.geo, coords={"time": times, **spool.coords})
        attrs = dict(source=url, title=name, **({"model_run": model_run} if model_run else {}))

        def layout(chunks):
            d = spool.dataset(times, chunks)
            d.attrs.update(attrs)
            return d

        timed("map.zarr", lambda: write(layout(dict(zip(("time", y, x), m.map_shard))), path / "map.zarr",
                                        {"time": 1, y: m.map_chunk, x: m.map_chunk}, m, dict(zip(("time", y, x), m.map_shard))))
        ovo = xr.concat(ov, "time", data_vars="minimal", coords="minimal", compat="override")
        ovy, ovx = -(-ds.sizes[y] // m.ov_factor), -(-ds.sizes[x] // m.ov_factor)
        timed("ov.zarr", lambda: write(ovo, path / "ov.zarr", {"time": m.ov_tchunk, y: ovy, x: ovx}, m))
        write_sidecars(sk.assign_attrs(attrs), path, m)
        sh = {"time": n, y: m.ts_shard, x: m.ts_shard}
        timed("ts.zarr", lambda: write(layout(sh), path / "ts.zarr", {"time": n, y: m.ts_chunk, x: m.ts_chunk}, m, sh))
    finally:
        shutil.rmtree(spool_dir, ignore_errors=True)
    # published last: readers only ever see complete runs
    (root / "latest.json").write_text(json.dumps({"run": run}))
    size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    log.info("done: %d steps, vars=%s -> %s (%.0f MB)", n, vars, path, size / 1e6)
    return 0
