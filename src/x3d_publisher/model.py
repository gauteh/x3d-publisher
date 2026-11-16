"""Model configuration: what to read, how to pack it and how to lay it out (a TOML file per model)."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

FORMAT_VERSION = 1  # of the published layout, see x3d's docs/format.md


@dataclass(frozen=True)
class Model:
    name: str
    vars: list[str]
    ov_vars: list[str]
    scales: dict[str, float]
    # coarser packing of the overview only (ov.zarr): variable -> scale_factor; the others use `scales`
    ov_scales: dict[str, float] = field(default_factory=dict)
    # the source: an OPeNDAP/zarr/netCDF URL, or `files` (a glob of netCDF files, e.g. on a shared file system) holding
    # the newest `files_runs` model runs, each split in files that are joined along time; later runs win where they overlap
    # `files` may be a list of globs for redundant copies of the same output (the production systems A and B): a file
    # (by its name) is taken from the first copy that has it complete, so a file missing on one side is read from the other
    url: str | None = None
    files: str | list[str] | None = None
    files_runs: int = 3
    # the model run (reference time) in a file name, as a regex with one group in the form %Y%m%dT%HZ
    files_run_regex: str = r"_(\d{8}T\d{2}Z)_"
    # a run is only used when it is complete: it has `files_per_run` files (if set; the last piece is written last)
    # and none of them was modified within the last `files_min_age` seconds (it may still be written).
    # An incomplete run is skipped as a whole, so the store never contains a half-written run.
    files_per_run: int | None = None
    files_min_age: float = 300
    y_dim: str = "rlat"
    x_dim: str = "rlon"
    lat: str = "latitude"
    lon: str = "longitude"
    reference_time: str | None = None
    hours: int = 72
    # forecast sources: start this many hours before the newest model run (reference_time) and keep everything after it;
    # overrides `hours` unless --hours/--start is given
    back_hours: int | None = None
    # fixed selections on other dimensions, e.g. {depth = 0.0} for the surface layer
    select: dict = field(default_factory=dict)
    # global attributes: `attrs` overrides/extends those of the source, `drop_attrs` leaves some of the source's out
    # (the time-dependent ones and the history of the source file are always left out)
    attrs: dict = field(default_factory=dict)
    drop_attrs: tuple[str, ...] = ()
    zstd_level: int = 5
    map_chunk: int = 256
    map_shard: tuple[int, int, int] = (1, 1280, 768)
    ts_chunk: int = 16
    ts_shard: int = 128
    ov_factor: int = 4
    ov_tchunk: int = 24
    mesh_stride: int = 4
    # validation before publishing (see validate.py): the plausible range [lo, hi] of each variable in its unpacked unit,
    # the fraction of valid (non-fill) cells a variable must have at every time step, and the longest gap in the time axis
    ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    min_valid: float = 0.005
    max_gap_hours: float | None = 1.5
    # track datasets (kind = "track", see track.py): along-track time series (satellite altimeter) kept as one growing
    # 1-D time series. `files` are strftime patterns (%Y/%m: the directory of the window start); each file holds one
    # window and a window is re-issued as late data arrives, so the newest file of a window (the last of the three
    # stamps in `files_window_regex`: window start, window end, creation) is used.
    kind: str = "grid"
    files_window_regex: str = r"_(\d{8}T\d{6})_(\d{8}T\d{6})_(\d{8}T\d{6})\.nc$"
    retention_days: float = 30
    # the series starts at the first published window until it is older than retention_days + trim_days; dropping the
    # head shifts every sample, so it is done in one go (all chunks rewritten) instead of at every window
    trim_days: float = 7
    # the newest windows are always uploaded again (late revisions); older ones only when their file changed
    tail_windows: int = 4
    cache_seconds: int = 60  # max-age of the (mutable) track objects
    track_chunk: int = 16384  # samples per chunk object
    extra: dict = field(default_factory=dict)

    @property
    def file_globs(self) -> list[str]:
        return [self.files] if isinstance(self.files, str) else list(self.files or [])

    @classmethod
    def from_dict(cls, d: dict) -> "Model":
        d = dict(d)
        if d.get("kind") == "track":
            d.setdefault("ov_vars", [])
        layout = d.pop("layout", {})
        d["scales"] = d.pop("scales", {})
        d["ov_scales"] = d.pop("ov_scales", {})
        d["ranges"] = {k: tuple(v) for k, v in d.pop("ranges", {}).items()}
        known = set(cls.__dataclass_fields__) - {"extra"}
        params = {k: v for k, v in {**d, **layout}.items() if k in known}
        if isinstance(params.get("files"), list):
            params["files"] = list(params["files"])
        if "drop_attrs" in params:
            params["drop_attrs"] = tuple(params["drop_attrs"])
        if "map_shard" in params:
            params["map_shard"] = tuple(params["map_shard"])
        m = cls(**params, extra={k: v for k, v in {**d, **layout}.items() if k not in known})
        if bool(m.url) == bool(m.files):
            raise ValueError(f"model {m.name}: give exactly one of `url` and `files`")
        if m.kind not in ("grid", "track"):
            raise ValueError(f"model {m.name}: kind must be \"grid\" or \"track\"")
        if m.kind == "track":
            if not m.files or m.tail_windows < 1 or m.track_chunk < 1:
                raise ValueError(f"model {m.name}: a track model needs `files`, tail_windows >= 1 and track_chunk >= 1")
            if m.lat not in m.scales or m.lon not in m.scales:
                raise ValueError(f"model {m.name}: a track model needs scales for {m.lat} and {m.lon}")
        missing = [v for v in dict.fromkeys(m.vars + m.ov_vars) if v not in m.scales]
        if missing:
            raise ValueError(f"model {m.name}: no scale for {missing}")
        bad = [v for v, r in m.ranges.items() if v not in m.scales or len(r) != 2 or not r[0] < r[1]]
        if bad:
            raise ValueError(f"model {m.name}: bad `ranges` for {bad}: need a variable with a scale and [lo, hi]")
        stray = [v for v in m.ov_scales if v not in m.ov_vars]
        if stray:
            raise ValueError(f"model {m.name}: ov_scales for variables that are not in ov_vars: {stray}")
        return m


def bundled() -> list[str]:
    return sorted(p.name[:-5] for p in resources.files("x3d_publisher.models").iterdir() if p.name.endswith(".toml"))


def load(name_or_path: str | Path) -> Model:
    """A bundled model by name (`ww3_4km`), or a TOML file."""
    p = Path(name_or_path)
    if p.suffix == ".toml" and p.exists():
        return Model.from_dict(tomllib.loads(p.read_text()))
    if str(name_or_path) not in bundled():
        raise SystemExit(f"unknown model {name_or_path!r}; bundled: {', '.join(bundled())} (or give a .toml file)")
    return Model.from_dict(tomllib.loads(resources.files("x3d_publisher.models").joinpath(f"{name_or_path}.toml").read_text()))
