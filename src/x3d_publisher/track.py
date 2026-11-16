"""Track datasets: along-track time series (satellite altimeter) as one growing 1-D time series.

There is no grid, so none of the map/ts/ov layouts. The store is one Zarr v3 group of 1-D arrays along `time`, cut
into chunks of `track_chunk` samples (one object each, plain `bytes` + `zstd`, no sharding):

  <store>/latest.json                    {"run": "track", "kind": "track"}
  <store>/track/dataset.json             windows (start, file version, samples, offset), time span, scales, attributes
  <store>/track/track.zarr/              time (int64 seconds), latitude/longitude (int32), the variables (int16)

The source is a file per 3 h window that is re-issued (as a new file with a later creation stamp) while late data
arrive, so a window is always taken from its newest file. A window only changes in the last few updates, so the series
is "frozen" behind the newest `tail_windows` windows: an update uploads the chunks from the first sample of the first
changed window or of the `tail_windows`-th newest window, whichever is earlier, and the metadata.
With finite retention, the series is cut at the start only when it is older than `retention_days + trim_days` (all
samples shift then, so every chunk is rewritten). A `retention_days` of 0 retains all available history.

Unlike runs, the objects are overwritten in place: they are served with a short TTL (`max-age=cache_seconds`, 60 by default) and revalidated.
A reader that fetches during an update can see the new tail with the old shape; it never sees a chunk that is
half-written (an object PUT is atomic). Publishing order: chunks, then the arrays' and the group's `zarr.json`, then
`dataset.json`, then `latest.json`.
"""
from __future__ import annotations

import datetime as dt
import glob
import json
import logging
import os
import re
import shutil
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import xarray as xr
from zarr.codecs import ZstdCodec

from .convert import UP_TO_DATE, extent, global_attrs, jsonable, to_iso, var_attrs
from .model import FORMAT_VERSION, Model

# netCDF4 cannot lock files on lustre ("NetCDF: HDF error")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

log = logging.getLogger("x3d_publisher.track")

RUN = "track"  # the (only) run directory
ARRAYS = "track.zarr"
STAMP = "%Y%m%dT%H%M%S"
FILL16, FILL32 = -32768, int(np.iinfo("int32").min)


def cache_control(m: Model | None) -> str:
    return f"public, max-age={m.cache_seconds if m else 60}"


ATTRS = ("standard_name", "long_name", "units", "comment")


def is_track(store) -> bool:
    """Whether the local store directory `store` (containing latest.json) is a track dataset."""
    try:
        return json.loads((Path(store) / "latest.json").read_text()).get("kind") == "track"
    except (OSError, ValueError):
        return False


# --- source files ----------------------------------------------------------------------------------------------

def stamp(t: dt.datetime) -> str:
    return t.strftime(STAMP)


def iso(s: str) -> str:
    return dt.datetime.strptime(s, STAMP).strftime("%Y-%m-%dT%H:%M:%SZ")


def candidates(m: Model, lo: dt.datetime | None, hi: dt.datetime) -> list[tuple[int, str]]:
    """(index of the glob, file) of the files in the month directories of [lo, hi]; a month directory holds the
    windows that start in it."""
    pats: dict[str, int] = {}
    if lo is None:
        for i, g in enumerate(m.file_globs):
            pattern = g.replace("%Y", "[0-9][0-9][0-9][0-9]").replace("%m", "[0-9][0-9]")
            if "%" in pattern:
                raise ValueError("unlimited track retention supports only %Y and %m in file patterns")
            pats.setdefault(pattern, i)
        return [(i, f) for p, i in pats.items() for f in sorted(glob.glob(p))]
    cur = lo.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    while cur <= hi:
        for i, g in enumerate(m.file_globs):
            pats.setdefault(cur.strftime(g), i)
        cur = (cur + dt.timedelta(days=32)).replace(day=1)
    return [(i, f) for p, i in pats.items() for f in sorted(glob.glob(p))]


def windows_available(m: Model, lo: dt.datetime | None, hi: dt.datetime) -> dict[str, list[str]]:
    """Window start stamp -> usable files, newest version (creation stamp) first. A file is usable when it is not empty
    and was not modified in the last `files_min_age` seconds. Where several globs (production systems) have the same
    file, the first is preferred."""
    rx, wall = re.compile(m.files_window_regex), time.time()
    found: dict[str, dict[str, list[str]]] = {}
    for i, f in candidates(m, lo, hi):
        mt = rx.search(Path(f).name)
        if not mt:
            continue
        start, _, version = mt.groups()
        if (lo is not None and start < stamp(lo)) or start > stamp(hi):
            continue
        try:
            st = Path(f).stat()
        except OSError:
            continue
        if st.st_size > 0 and wall - st.st_mtime >= m.files_min_age:
            found.setdefault(start, {}).setdefault(version, []).append(f)
    return {s: [v[ver][0] for ver in sorted(v, reverse=True)] for s, v in sorted(found.items())}


def version_of(m: Model, path: str) -> str:
    return re.search(m.files_window_regex, Path(path).name).group(3)


def series_start(m: Model, now: dt.datetime, previous: dict | None) -> dt.datetime | None:
    """Where the series starts: at the first published window (so that nothing shifts) until that is older than
    retention + trim, then `retention_days` back. None (no lower bound) when retention_days is 0."""
    if m.retention_days == 0:
        return None
    lo = now - dt.timedelta(days=m.retention_days)
    if previous and previous.get("kind") == "track" and previous.get("windows"):
        first = dt.datetime.strptime(previous["windows"][0]["start"], STAMP)
        if first >= now - dt.timedelta(days=m.retention_days + m.trim_days):
            return first
    return lo


def first_changed(previous: list[dict] | None, windows: list[dict]) -> int:
    """Index of the first window that is not the published one (same start and file version)."""
    i = 0
    for p, w in zip(previous or [], windows):
        if (p["start"], p["version"]) != (w["start"], w["version"]):
            break
        i += 1
    return i


# --- conversion -------------------------------------------------------------------------------------------------

def read_window(m: Model, files: list[str]):
    """The first readable of `files` (newest first): (file, dataset loaded in memory)."""
    for f in files:
        try:
            with xr.open_dataset(f) as ds:
                return f, ds[[*m.vars, m.lat, m.lon]].load()
        except Exception as e:  # being written, or a lustre hiccup: the next older version will do
            log.warning("cannot read %s (%s)", f, str(e).splitlines()[0] if str(e) else type(e).__name__)
    return None, None


def read_windows(m: Model, available: dict[str, list[str]]):
    """Yield (start, file, source dataset, samples) of the newest readable file of every window, one window in memory at
    a time. The samples (float64 columns) are sorted and strictly increasing across windows (a window is disjoint from the
    next; this only guards the contract)."""
    last = -np.inf
    for start, files in available.items():
        f, ds = read_window(m, files)
        if ds is None:
            log.warning("window %s skipped: no readable file", start)
            continue
        t = ds.time.values.astype("datetime64[s]").astype("int64")
        order = np.argsort(t, kind="stable")
        keep = order[np.append(True, np.diff(t[order]) > 0) & (t[order] > last)]
        if len(keep) != len(t):
            log.warning("window %s: dropped %d of %d samples (duplicate or out of order times)", start, len(t) - len(keep), len(t))
        part = {"time": t[keep], **{v: ds[v].values.astype("float64")[keep] for v in (*m.vars, m.lat, m.lon)}}
        part[m.lon] = (part[m.lon] + 180) % 360 - 180  # the source has 0..360; publish -180..180
        if len(keep):
            last = t[keep[-1]]
        yield start, f, ds, part


class TrackWriter:
    """Writes the arrays of the store in batches of whole chunks as the windows come in, so that memory is bounded by one
    batch (`BATCH_CHUNKS` chunks) however long the series is. Group and variable attributes are set at the end."""

    BATCH_CHUNKS = 16

    def __init__(self, path: Path, m: Model):
        self.path, self.m, self.n, self.started = path, m, 0, False
        self.buf: dict[str, list[np.ndarray]] = {}
        self.buffered = 0
        if path.exists():
            shutil.rmtree(path)

    def add(self, part: dict) -> None:
        for k, v in part.items():
            self.buf.setdefault(k, []).append(v)
        self.buffered += len(part["time"])
        if self.buffered >= self.BATCH_CHUNKS * self.m.track_chunk:
            self.flush()

    def flush(self, final: bool = False) -> None:
        m = self.m
        n = self.buffered if final else self.buffered // m.track_chunk * m.track_chunk
        if n == 0:
            return
        cat = {k: np.concatenate(v) for k, v in self.buf.items()}
        self.buf = {k: [v[n:]] for k, v in cat.items()}
        self.buffered -= n
        ds = xr.Dataset(
            {k: ("time", v[:n]) for k, v in cat.items() if k != "time"},
            coords={"time": ("time", cat["time"][:n].astype("datetime64[s]").astype("datetime64[ns]"), {"standard_name": "time"})},
        )
        if not self.started:
            comp, c = ZstdCodec(level=m.zstd_level), (m.track_chunk,)
            enc = {"time": {"units": "seconds since 1970-01-01T00:00:00", "dtype": "int64", "chunks": c, "compressors": (comp,)}}
            for v in (*m.vars, m.lat, m.lon):
                fill, dtype = (FILL32, "int32") if v in (m.lat, m.lon) else (FILL16, "int16")
                enc[v] = {"dtype": dtype, "scale_factor": m.scales[v], "_FillValue": fill, "fill_value": fill, "chunks": c, "compressors": (comp,)}
            ds.to_zarr(self.path, mode="w", zarr_format=3, consolidated=False, encoding=enc)
            self.started = True
        else:  # the encoding of the existing arrays is used
            ds.to_zarr(self.path, mode="a", append_dim="time", zarr_format=3, consolidated=False)
        self.n += n

    def close(self, attrs: dict, vattrs: dict) -> None:
        import zarr

        self.flush(final=True)
        g = zarr.open_group(str(self.path), mode="r+", zarr_format=3)
        g.attrs.update(attrs)
        for v, a in vattrs.items():
            g[v].attrs.update(a)
        zarr.consolidate_metadata(str(self.path))


def convert_track(m: Model, out, *, name: str | None = None, now: dt.datetime | None = None, previous: dict | None = None,
                  check: bool = False, force: bool = False) -> int:
    """Write `<out>/<name>/{latest.json,track/}` from the newest file of every window. `previous` is the published
    dataset.json (decides where the series starts and whether anything changed); returns UP_TO_DATE if nothing did.
    The windows are read and written one at a time: memory does not grow with the length of the series."""
    name = name or m.name
    now = now or dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    lo = series_start(m, now, previous)
    available = windows_available(m, lo, now)
    log.info("%d windows from %s (%d files)", len(available), lo, sum(map(len, available.values())))
    if not available:
        raise SystemExit(f"no usable files matching {', '.join(m.file_globs)} for {lo} .. {now}")
    if m.retention_days == 0 and previous and previous.get("kind") == "track":
        missing = [w["start"] for w in previous["windows"] if w["start"] not in available]
        if missing:
            raise SystemExit(f"refusing to drop {len(missing)} previously published windows; first missing: {missing[0]}")
    newest = {s: version_of(m, f[0]) for s, f in available.items()}
    if not force and previous and previous.get("kind") == "track" and [(w["start"], w["version"]) for w in previous["windows"]] == list(newest.items()):
        log.info("published store already has these %d windows; nothing to do", len(newest))
        return UP_TO_DATE
    if check:
        log.info("the published store is out of date: %d windows to convert", len(newest))
        return 0

    root = Path(out) / name
    path = root / RUN
    path.mkdir(parents=True, exist_ok=True)
    writer = TrackWriter(path / ARRAYS, m)
    windows, offset, src, t0, t1 = [], 0, None, None, None
    box = {"lat": [np.inf, -np.inf], "lon": [np.inf, -np.inf]}
    for k, (start, f, ds, part) in enumerate(read_windows(m, available), 1):
        n = len(part["time"])
        windows.append({"start": start, "version": version_of(m, f), "n": n, "offset": offset})
        offset += n
        src = ds
        if n:
            t0 = part["time"][0] if t0 is None else t0
            t1 = part["time"][-1]
            for key, v in (("lat", m.lat), ("lon", m.lon)):
                x = part[v][np.isfinite(part[v])]
                if len(x):
                    box[key] = [min(box[key][0], float(x.min())), max(box[key][1], float(x.max()))]
            writer.add(part)
        if k % 1000 == 0:
            log.info("%d of %d windows, %d samples", k, len(available), offset)
    if not offset:
        raise SystemExit("no samples in the source files")
    skipped = sorted(set(available) - {w["start"] for w in windows})
    if skipped and m.retention_days == 0:
        raise SystemExit(f"refusing to omit {len(skipped)} unreadable windows; first: {skipped[0]}")

    n = offset
    times = to_iso(np.array([t0, t1]).astype("datetime64[s]"))
    bounds = {"geospatial_lat_min": box["lat"][0], "geospatial_lat_max": box["lat"][1],
              "geospatial_lon_min": box["lon"][0], "geospatial_lon_max": box["lon"][1]}
    stamp_now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    url = f"{Path(m.file_globs[0]).name} ({len(windows)} windows)"
    attrs = global_attrs({k: v for k, v in src.attrs.items()}, m, name=name, url=url, model_run=None, times=times, bounds=bounds, now=stamp_now, link=False)
    attrs.update(featureType="trajectory")
    vattrs = {v: var_attrs(src[v].attrs, None, ATTRS) for v in (*m.vars, m.lat, m.lon)}
    writer.close(attrs, vattrs)
    assert writer.n == n, (writer.n, n)
    (path / "dataset.json").write_text(json.dumps({
        "format_version": FORMAT_VERSION, "kind": "track", "name": name, "run": RUN, "n": n, "chunk": m.track_chunk,
        "dims": {"time": "time"}, "geo": [m.lat, m.lon], "vars": m.vars, "scales": m.scales,
        "tmin": times[0], "tmax": times[-1],
        "bbox": [bounds["geospatial_lon_min"], bounds["geospatial_lat_min"], bounds["geospatial_lon_max"], bounds["geospatial_lat_max"]],
        "windows": windows, "attrs": attrs, "variables": {k: {a: jsonable(b) for a, b in v.items()} for k, v in vattrs.items()},
    }))
    (root / "latest.json").write_text(json.dumps({"run": RUN, "kind": "track"}))
    size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    log.info("done: %d samples in %d windows (%s .. %s) -> %s (%.1f MB)", n, len(windows), times[0], times[-1], path, size / 1e6)
    return 0


# --- published state --------------------------------------------------------------------------------------------

def http_info(base: str) -> dict | None:
    """dataset.json of the published track store at the public URL `base`, or None."""
    def get(url):
        with urllib.request.urlopen(urllib.request.Request(url, headers={"Cache-Control": "no-cache"}), timeout=30) as r:
            return json.load(r)

    try:
        base = base.rstrip("/")
        return get(f"{base}/{get(base + '/latest.json')['run']}/dataset.json")
    except Exception as e:
        log.info("no published store to compare with (%s)", e)
        return None


def s3_info(s3, bucket: str, prefix: str) -> dict | None:
    """The same, read from the bucket itself (not through a cache)."""
    from botocore.exceptions import ClientError

    try:
        return json.loads(s3.get_object(Bucket=bucket, Key=f"{prefix}/{RUN}/dataset.json")["Body"].read())
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404", "NoSuchBucket"):
            return None
        raise


def chunk_index(key: str) -> int | None:
    mt = re.search(r"/c/(\d+)$", key)
    return int(mt.group(1)) if mt else None


def plan_upload(previous: dict | None, info: dict, tail_windows: int, remote_chunks: set[str], local_chunks: set[str]) -> int:
    """First chunk index to upload (0 = everything): the chunk of the first sample of the first changed window, or of
    the `tail_windows`-th newest window, whichever is first. Everything is uploaded if what is published does not match
    (other chunking, variables or scales; a chunk before that point is missing or should not exist)."""
    ws = info["windows"]
    if not previous or previous.get("kind") != "track" or any(previous.get(k) != info[k] for k in ("chunk", "vars", "scales")):
        return 0
    boundary = min(first_changed(previous["windows"], ws), max(0, len(ws) - tail_windows))
    c0 = (ws[boundary]["offset"] if boundary < len(ws) else info["n"]) // info["chunk"]
    before = lambda keys: {k for k in keys if (chunk_index(k) or 0) < c0}
    return c0 if before(remote_chunks) == before(local_chunks) else 0


def publish_track(src, dest: str, *, endpoint_url: str | None = None, configure: bool = True, workers: int = 8, s3=None,
                  validate: bool = True, model: Model | None = None, tail_windows: int | None = None) -> int:
    """Upload the local track store `src` (<out>/<name>) to `dest` (s3://bucket/prefix): only the chunks from the changed
    tail on. Returns the number of chunk objects uploaded."""
    import boto3

    from .publish import configure_bucket, split

    src = Path(src)
    bucket, prefix = split(dest)
    s3 = s3 or boto3.client("s3", endpoint_url=endpoint_url)
    info = json.loads((src / RUN / "dataset.json").read_text())
    previous = s3_info(s3, bucket, prefix)
    if validate:
        from .validate import validate_track

        for w in validate_track(src / RUN, model, previous):
            print(f"warning: {w}", flush=True)
    if configure:
        try:
            configure_bucket(s3, bucket)
        except Exception as e:
            print(f"warning: could not set the public-read policy/CORS on s3://{bucket}: {e}", flush=True)

    base = src / RUN
    local = {str(f.relative_to(src)): f for f in (base / ARRAYS).rglob("*") if f.is_file()}
    chunks = {k: f for k, f in local.items() if chunk_index(k) is not None}
    remote_prefix = f"{prefix}/{RUN}/{ARRAYS}/"
    remote = {}
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=remote_prefix):
        remote.update({o["Key"]: o["Size"] for o in page.get("Contents", [])})
    rel = lambda k: k[len(prefix) + 1:]
    tail = tail_windows or (model.tail_windows if model else 4)
    c0 = plan_upload(previous, info, tail, {rel(k) for k in remote if chunk_index(k) is not None}, set(chunks))
    todo = [k for k in chunks if (chunk_index(k) or 0) >= c0]
    log.info("uploading %d of %d chunks (from chunk %d) of %d windows", len(todo), len(chunks), c0, len(info["windows"]))

    def put(key: str, f: Path, ctype="application/octet-stream"):
        s3.upload_file(str(f), bucket, f"{prefix}/{key}", ExtraArgs={"CacheControl": cache_control(model), "ContentType": ctype})

    metadata = [k for k in local if k not in chunks]  # <array>/zarr.json, then the group's (shortest path) last
    metadata.sort(key=lambda k: -k.count("/"))
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(lambda k: put(k, chunks[k]), todo))
    for k in metadata:
        put(k, local[k], "application/json")
    put(f"{RUN}/dataset.json", base / "dataset.json", "application/json")
    listed = {}
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=remote_prefix):
        listed.update({o["Key"]: o["Size"] for o in page.get("Contents", [])})
    bad = [k for k in [*todo, *metadata] if listed.get(f"{prefix}/{k}") != local[k].stat().st_size]
    if bad:
        raise RuntimeError("upload is incomplete, latest.json was NOT updated:\n  " + "\n  ".join(bad[:20]))
    s3.put_object(Bucket=bucket, Key=f"{prefix}/latest.json", Body=(src / "latest.json").read_bytes(),
                  CacheControl=cache_control(model), ContentType="application/json")
    stale = [k for k in remote if chunk_index(k) is not None and rel(k) not in chunks]
    for i in range(0, len(stale), 1000):
        s3.delete_objects(Bucket=bucket, Delete={"Objects": [{"Key": k} for k in stale[i:i + 1000]], "Quiet": True})
    print(f"published {len(todo)} chunks + metadata, {info['n']} samples in {len(info['windows'])} windows ({info['tmin']} .. {info['tmax']})")
    return len(todo)


def run_cycle(m: Model, *, bucket_url: str, endpoint_url: str | None = None, name: str | None = None,
              now: dt.datetime | None = None, force: bool = False, check: bool = False, work: str | None = None,
              workers: int = 8, configure: bool = True) -> int:
    """One cycle: compare with the bucket, convert, validate, upload the changed tail. Returns 0 (done / needed) or UP_TO_DATE."""
    import boto3

    from .publish import split

    name = name or m.name
    dest = f"{bucket_url.rstrip('/')}/{name}"
    bucket, prefix = split(dest)
    s3 = boto3.client("s3", endpoint_url=endpoint_url)
    previous = s3_info(s3, bucket, prefix)
    Path(work or tempfile.gettempdir()).mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="run.", dir=work))
    try:
        rc = convert_track(m, tmp, name=name, now=now, previous=previous, check=check, force=force)
        if rc == UP_TO_DATE:
            return UP_TO_DATE if check else 0
        if check:
            return 0
        publish_track(tmp / name, dest, endpoint_url=endpoint_url, configure=configure, workers=workers, s3=s3, model=m)
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
