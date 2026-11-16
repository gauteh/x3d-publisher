"""Check that a converted run will work in the web client, before it is published.

The checks follow the contract in x3d/docs/format.md and what the client (wave-world `web/src/{lib,zarr}.rs`) needs:
the sidecars, the shapes and codecs of the arrays, the time axis, the grid mesh, and that the data are really there
(a time step whose source files were missing becomes all fill and would show as an empty map).
"""
import json
import math
import re
from pathlib import Path

import numpy as np

from .model import Model

FILL = -32768
ISO = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")
REQUIRED = ("times", "ny", "nx", "mesh_stride", "mesh_ny", "mesh_nx", "ov_factor", "ov_tchunk")


class ValidationError(Exception):
    def __init__(self, errors: list[str], warnings: list[str] | None = None):
        self.errors, self.warnings = errors, warnings or []
        super().__init__(f"{len(errors)} problem(s) make the run unsafe to publish:\n  " + "\n  ".join(errors))


def _json(p: Path, errs: list[str]):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError) as e:
        errs.append(f"{p.name}: cannot read: {e}")


def _array_meta(arr_dir: Path, errs: list[str], label: str, sharded: bool = True):
    """zarr.json of one array, checked for what the client parses."""
    meta = _json(arr_dir / "zarr.json", errs)
    if meta is None:
        return None
    if meta.get("zarr_format") != 3 or meta.get("node_type") != "array":
        errs.append(f"{label}: not a zarr v3 array")
        return None
    if meta.get("data_type") != "int16":
        errs.append(f"{label}: data_type {meta.get('data_type')!r}, the client decodes int16")
    if str(meta.get("fill_value")) != str(FILL):
        errs.append(f"{label}: fill_value {meta.get('fill_value')!r}, expected {FILL}")
    attrs = meta.get("attributes", {})
    sf = attrs.get("scale_factor")
    if not isinstance(sf, (int, float)) or not sf > 0 or not math.isfinite(sf):
        errs.append(f"{label}: missing or bad scale_factor {sf!r}")
    codecs = meta.get("codecs") or [{}]
    if not sharded:  # ov.zarr: plain chunks
        pass
    elif codecs[0].get("name") != "sharding_indexed":
        errs.append(f"{label}: first codec is {codecs[0].get('name')!r}, the client reads sharded arrays")
    elif codecs[0].get("configuration", {}).get("index_location") != "start":
        errs.append(f"{label}: shard index_location must be 'start'")
    return meta


def validate_run(run_dir, model: Model | None = None, previous: dict | None = None, deep: bool = True) -> list[str]:
    """Check the run directory (with dataset.json, grid.bin, map.zarr, ov.zarr, ts.zarr). Raises `ValidationError` if
    anything would break the client; returns the warnings. `previous` is the dataset.json of the run that is published
    now: a run that is much smaller than it is suspicious. `deep=False` skips reading the data chunks."""
    run_dir = Path(run_dir)
    errs: list[str] = []
    warns: list[str] = []
    ds = _json(run_dir / "dataset.json", errs)
    if ds is None:
        raise ValidationError(errs)
    missing = [k for k in REQUIRED if k not in ds]
    if missing:
        raise ValidationError([f"dataset.json: missing {missing}"])
    ny, nx, times = ds["ny"], ds["nx"], ds["times"]

    # time axis
    if not times:
        errs.append("dataset.json: empty time axis")
    bad = [t for t in times if not isinstance(t, str) or not ISO.match(t)]
    if bad:
        errs.append(f"dataset.json: times not ISO-8601 UTC ('...Z'), e.g. {bad[0]!r}")
    elif any(b <= a for a, b in zip(times, times[1:])):
        errs.append("dataset.json: times are not strictly increasing")
    elif model and model.max_gap_hours and len(times) > 1:
        t = np.array([np.datetime64(x[:-1]) for x in times])
        gap = float(np.max(np.diff(t)) / np.timedelta64(1, "h"))
        if gap > model.max_gap_hours:
            errs.append(f"dataset.json: gap of {gap:g} h in the time axis (max {model.max_gap_hours:g} h)")
    for k in ("ny", "nx", "mesh_stride", "ov_factor", "ov_tchunk"):
        if not isinstance(ds[k], int) or ds[k] < 1:
            errs.append(f"dataset.json: bad {k} {ds[k]!r}")
    if errs:
        raise ValidationError(errs, warns)
    nt = len(times)
    f = ds["ov_factor"]
    cny, cnx = -(-ny // f), -(-nx // f)
    if model and model.name != ds.get("name"):
        warns.append(f"dataset.json name {ds.get('name')!r} differs from the model {model.name!r}")

    # grid mesh
    grid = run_dir / "grid.bin"
    want = ds["mesh_ny"] * ds["mesh_nx"] * 2 * 4
    if not grid.is_file():
        errs.append("grid.bin is missing")
    elif grid.stat().st_size != want:
        errs.append(f"grid.bin is {grid.stat().st_size} bytes, expected {want} ({ds['mesh_ny']}x{ds['mesh_nx']} lon+lat float32)")
    else:
        g = np.fromfile(grid, "<f4").reshape(2, ds["mesh_ny"], ds["mesh_nx"])
        if not np.isfinite(g).all():
            errs.append("grid.bin: non-finite coordinates")
        elif np.abs(g[0]).max() > 360 or np.abs(g[1]).max() > 90:
            errs.append("grid.bin: lon/lat out of range (degrees expected)")
    ms = ds["mesh_stride"]  # the mesh always includes the last row and column
    if ds["mesh_ny"] != -(-(ny - 1) // ms) + 1 or ds["mesh_nx"] != -(-(nx - 1) // ms) + 1:
        errs.append(f"dataset.json: mesh {ds['mesh_ny']}x{ds['mesh_nx']} does not match grid {ny}x{nx} with stride {ms}")

    # arrays
    variables = ds.get("variables") or []
    want_vars = list(model.vars) if model else list(variables)
    ov_vars = list(ds.get("ov_vars") or (model.ov_vars if model else []))
    try:
        import zarr
    except ImportError:  # pragma: no cover
        zarr = None
    stores = {"map.zarr": (want_vars, (nt, ny, nx)), "ts.zarr": (want_vars, (nt, ny, nx)), "ov.zarr": (ov_vars, (nt, cny, cnx))}
    for sname, (names, shape) in stores.items():
        if not (run_dir / sname).is_dir():
            if sname == "ov.zarr" and not ov_vars:
                continue
            errs.append(f"{sname} is missing")
            continue
        if not (run_dir / sname / "zarr.json").is_file():
            errs.append(f"{sname}/zarr.json is missing")
        for v in names:
            label = f"{sname}/{v}"
            meta = _array_meta(run_dir / sname / v, errs, label, sharded=sname != "ov.zarr")
            if meta is None:
                continue
            if tuple(meta.get("shape", ())) != shape:
                errs.append(f"{label}: shape {tuple(meta.get('shape', ()))}, dataset.json says {shape}")
                continue
            sf = meta["attributes"].get("scale_factor")
            exp = ds.get("ov_scales" if sname == "ov.zarr" else "scales", {}).get(v)
            if sf is not None and exp is not None and not math.isclose(sf, exp, rel_tol=1e-6):
                errs.append(f"{label}: scale_factor {sf} differs from dataset.json ({exp})")
    if errs or not deep or zarr is None:
        if errs:
            raise ValidationError(errs, warns)
        return warns

    # the data
    min_valid = model.min_valid if model else 0.0
    ranges = model.ranges if model else {}
    step = max(1, min(ny, nx) // 128)  # about 128 samples across the smaller side
    for sname in ("map.zarr", "ov.zarr"):
        if not (run_dir / sname).is_dir():
            continue
        g = zarr.open_group(run_dir / sname, mode="r")
        for v in (want_vars if sname == "map.zarr" else ov_vars):
            if v not in g:
                continue
            a = g[v]
            sf = a.attrs["scale_factor"]
            s = step if sname == "map.zarr" else 1
            empty = []
            for t in range(nt):
                x = np.asarray(a[t, ::s, ::s])
                ok = x != FILL
                if ok.mean() < min_valid:
                    empty.append(t)
                if ok.any() and v in ranges and sname == "map.zarr":
                    lo, hi = ranges[v]
                    u = x[ok] * sf
                    if u.min() < lo or u.max() > hi:
                        errs.append(f"{sname}/{v} at {times[t]}: values {u.min():g}..{u.max():g} outside the plausible range [{lo:g}, {hi:g}]")
                        break
            if empty:
                shown = ", ".join(times[t] for t in empty[:3]) + (" ..." if len(empty) > 3 else "")
                errs.append(f"{sname}/{v}: {len(empty)} of {nt} time steps have (almost) no valid data: {shown}")
    if previous:
        pn = len(previous.get("times", []))
        if pn and nt < 0.5 * pn:
            errs.append(f"only {nt} time steps, the published run has {pn}")
        if (previous.get("ny"), previous.get("nx")) != (ny, nx):
            warns.append(f"grid {ny}x{nx} differs from the published run ({previous.get('ny')}x{previous.get('nx')})")
    if errs:
        raise ValidationError(errs, warns)
    return warns


def validate_store(store, run: str | None = None, model: Model | None = None, published: dict | None = None, deep: bool = True) -> list[str]:
    """`store` is <out>/<name>: latest.json plus the run directories."""
    store = Path(store)
    errs: list[str] = []
    latest = _json(store / "latest.json", errs)
    if latest is None:
        raise ValidationError(errs)
    run = run or latest.get("run")
    if latest.get("kind") == "track":
        if not (store / run).is_dir():
            raise ValidationError([f"latest.json points to {run!r}, which is not a directory in the store"])
        return validate_track(store / run, model, published, deep)
    if not run or not (store / run).is_dir():
        raise ValidationError([f"latest.json points to run {run!r}, which is not a directory in the store"])
    return validate_run(store / run, model, published, deep)


def validate_track(path, model: Model | None = None, published: dict | None = None, deep: bool = True) -> list[str]:
    """A track store (<store>/track): dataset.json against the arrays, then the data (strictly increasing time, coordinates,
    value ranges, not mostly fill). `published` is the dataset.json now on S3: a series that lost most of its samples
    without being cut at the start fails."""
    import zarr

    path = Path(path)
    errs: list[str] = []
    warns: list[str] = []
    info = _json(path / "dataset.json", errs)
    meta = _json(path / "track.zarr" / "zarr.json", errs)
    if info is None or meta is None:
        raise ValidationError(errs)
    n, chunk = info.get("n"), info.get("chunk")
    ws = info.get("windows", [])
    if info.get("kind") != "track" or not isinstance(n, int) or n < 1 or not isinstance(chunk, int) or chunk < 1:
        errs.append("dataset.json: kind, n and chunk are required")
    if not ws or any(w["offset"] != sum(x["n"] for x in ws[:i]) for i, w in enumerate(ws)) or sum(w["n"] for w in ws) != n:
        errs.append("dataset.json: windows are empty or their offsets/samples do not add up to n")
    if [w["start"] for w in ws] != sorted({w["start"] for w in ws}):
        errs.append("dataset.json: windows are not strictly increasing")
    nodes = meta.get("consolidated_metadata", {}).get("metadata", {})
    lat, lon = info.get("geo", ["latitude", "longitude"])
    expect = {"time": "int64", lat: "int32", lon: "int32", **{v: "int16" for v in info.get("vars", [])}}
    for name, dtype in expect.items():
        a = nodes.get(name)
        if a is None:
            errs.append(f"track.zarr: array {name} is missing")
            continue
        if a["shape"] != [n] or a.get("data_type") != dtype or a["chunk_grid"]["configuration"]["chunk_shape"] != [chunk]:
            errs.append(f"track.zarr/{name}: expected shape [{n}], {dtype}, chunk [{chunk}]; got {a['shape']}, {a.get('data_type')}, {a['chunk_grid']['configuration']['chunk_shape']}")
        if name != "time" and (a.get("attributes", {}).get("scale_factor") != info["scales"].get(name) or "_FillValue" not in a.get("attributes", {})):
            errs.append(f"track.zarr/{name}: scale_factor/_FillValue do not match dataset.json")
    if published and published.get("kind") == "track" and published["windows"] and ws and published["windows"][0]["start"] == ws[0]["start"] \
            and n < published["n"] // 2:
        errs.append(f"the series has {n} samples, the published one {published['n']}: more than half lost")
    if errs:
        raise ValidationError(errs, warns)
    if deep:  # in slices of whole chunks, so that memory does not grow with the series
        g = zarr.open_group(str(path / "track.zarr"), mode="r")
        step = max(1, (1 << 20) // chunk) * chunk
        rng = model.ranges if model else {}
        lat_fill = np.iinfo("int32").min
        prev_t, unordered = None, False
        bounds = {v: [np.inf, -np.inf] for v in (lat, lon)}
        missing = {lat: 0, lon: 0}
        vmin, vmax, valid = {v: np.inf for v in info["vars"]}, {v: -np.inf for v in info["vars"]}, {v: 0 for v in info["vars"]}
        for i in range(0, n, step):
            t = g["time"][i:i + step]
            unordered |= bool((np.diff(t) <= 0).any()) or (prev_t is not None and t[0] <= prev_t)
            prev_t = t[-1]
            for v in (lat, lon):
                raw = g[v][i:i + step]
                x = raw[raw != lat_fill] * info["scales"][v]
                missing[v] += len(raw) - len(x)
                if len(x):
                    bounds[v] = [min(bounds[v][0], x.min()), max(bounds[v][1], x.max())]
            for v in info["vars"]:
                raw = g[v][i:i + step]
                ok = raw != FILL
                valid[v] += int(ok.sum())
                if ok.any() and v in rng:
                    x = raw[ok] * np.float32(info["scales"][v])
                    vmin[v], vmax[v] = min(vmin[v], x.min()), max(vmax[v], x.max())
        if unordered:
            errs.append("time is not strictly increasing")
        for v, (lo, hi) in {lat: (-90, 90), lon: (-180, 180)}.items():
            if missing[v] or bounds[v][0] < lo or bounds[v][1] > hi:
                errs.append(f"{v}: {missing[v]} missing or outside [{lo}, {hi}]")
        for v in info["vars"]:
            if valid[v] / n < (model.min_valid if model else 0.005):
                errs.append(f"{v}: only {valid[v] / n:.1%} of the samples have a value")
            if v in rng and valid[v] and (vmin[v] < rng[v][0] or vmax[v] > rng[v][1]):
                errs.append(f"{v}: values {vmin[v]:.3g} .. {vmax[v]:.3g} outside the plausible range {list(rng[v])}")
    if errs:
        raise ValidationError(errs, warns)
    return warns
