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
    # the source: an OPeNDAP/zarr/netCDF URL, or `files` (a glob of netCDF files, e.g. on a shared file system) holding
    # the newest `files_runs` model runs, each split in files that are joined along time; later runs win where they overlap
    url: str | None = None
    files: str | None = None
    files_runs: int = 3
    # the model run (reference time) in a file name, as a regex with one group in the form %Y%m%dT%HZ
    files_run_regex: str = r"_(\d{8}T\d{2}Z)_"
    # files modified more recently than this (seconds) may still be written and are skipped
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
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "Model":
        d = dict(d)
        layout = d.pop("layout", {})
        d["scales"] = d.pop("scales", {})
        known = set(cls.__dataclass_fields__) - {"extra"}
        params = {k: v for k, v in {**d, **layout}.items() if k in known}
        if "drop_attrs" in params:
            params["drop_attrs"] = tuple(params["drop_attrs"])
        if "map_shard" in params:
            params["map_shard"] = tuple(params["map_shard"])
        m = cls(**params, extra={k: v for k, v in {**d, **layout}.items() if k not in known})
        if bool(m.url) == bool(m.files):
            raise ValueError(f"model {m.name}: give exactly one of `url` and `files`")
        missing = [v for v in dict.fromkeys(m.vars + m.ov_vars) if v not in m.scales]
        if missing:
            raise ValueError(f"model {m.name}: no scale for {missing}")
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
