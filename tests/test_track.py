"""Track datasets (satellite altimeter): newest file per window, incremental upload of the changed tail, reading back."""
import dataclasses
import datetime as dt
import json
from pathlib import Path

import boto3
import numpy as np
import pytest
import xarray as xr
from moto import mock_aws

from x3d_publisher import model as models
from x3d_publisher import track
from x3d_publisher.convert import UP_TO_DATE
from x3d_publisher.validate import ValidationError, validate_store

T0 = dt.datetime(2026, 9, 1)
NOW = dt.datetime(2026, 9, 3)
NW = 8  # windows in the first round


@pytest.fixture
def m(tmp_path):
    base = models.load("altimeter_s3a")
    return dataclasses.replace(base, files=[str(tmp_path / "src/%Y/%m/s3a_*.nc")], files_min_age=0, track_chunk=64, track_time_chunk=256,
                               tail_windows=2, retention_days=5, trim_days=1)


def window(tmp_path, k, version, n):
    """The file of window k (3 h), created at `version` (hours after the window start) with n samples."""
    start = T0 + dt.timedelta(hours=3 * k)
    t = start + np.arange(n) * dt.timedelta(seconds=100)
    ds = xr.Dataset(
        {"VAVH": ("time", np.full(n, 1.0 + k + n / 1000)), "VAVH_UNFILTERED": ("time", np.full(n, 2.0)),
         "WIND_SPEED": ("time", np.r_[np.nan, np.full(n - 1, 5.0)])},
        coords={"time": t, "latitude": ("time", np.linspace(-80, 80, n)), "longitude": ("time", np.linspace(0, 359, n))},
        attrs={"title": "x", "first_meas_time": "x"},
    )
    d = tmp_path / "src" / start.strftime("%Y/%m")
    d.mkdir(parents=True, exist_ok=True)
    stamps = f"{start:%Y%m%dT%H%M%S}_{start + dt.timedelta(hours=3):%Y%m%dT%H%M%S}_{start + dt.timedelta(hours=version):%Y%m%dT%H%M%S}"
    ds.to_netcdf(d / f"s3a_{stamps}.nc")


@pytest.fixture
def s3(monkeypatch):
    for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        monkeypatch.setenv(k, "test")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    with mock_aws():
        c = boto3.client("s3")
        c.create_bucket(Bucket="bkt")
        yield c


def etags(s3):
    return {o["Key"]: o["ETag"] for o in s3.list_objects_v2(Bucket="bkt")["Contents"]}


def test_newest_file_of_each_window_is_used(tmp_path, m):
    window(tmp_path, 0, 4, 30)
    window(tmp_path, 0, 6, 50)  # re-issued with more samples
    window(tmp_path, 0, 5, 40)
    window(tmp_path, 1, 4, 20)
    assert track.convert_track(m, tmp_path / "out", now=NOW) == 0
    info = json.loads((tmp_path / "out/altimeter_s3a/track/dataset.json").read_text())
    assert [(w["version"], w["n"], w["offset"]) for w in info["windows"]] == [("20260901T060000", 50, 0), ("20260901T070000", 20, 50)]
    assert validate_store(tmp_path / "out/altimeter_s3a", model=m) == []
    # nothing new: up to date
    assert track.convert_track(m, tmp_path / "out2", now=NOW, previous=info) == UP_TO_DATE


def test_incremental_publish_rewrites_only_the_tail(tmp_path, m, s3):
    for k in range(NW):
        window(tmp_path, k, 4, 60)
    kw = dict(bucket_url="s3://bkt", name="alt", now=NOW, work=str(tmp_path / "work"), configure=False)
    assert track.run_cycle(m, **kw) == 0
    first = etags(s3)
    chunks = [k for k in first if "/VAVH/c/" in k]
    assert len([k for k in first if "/time/c/" in k]) == -(-NW * 60 // 256)  # `time` has its own, larger chunk
    assert len(chunks) == -(-NW * 60 // 64) and json.loads(s3.get_object(Bucket="bkt", Key="alt/latest.json")["Body"].read())["kind"] == "track"
    assert track.run_cycle(m, check=True, **kw) == UP_TO_DATE
    assert track.run_cycle(m, **kw) == 0 and etags(s3) == first  # nothing to publish

    # the last window is revised and one is added: only the chunks from the first changed window on are uploaded
    window(tmp_path, NW - 1, 6, 90)
    window(tmp_path, NW, 4, 60)
    assert track.run_cycle(m, **kw) == 0
    second = etags(s3)
    c0 = (NW - 1) * 60 // 64  # the revised window is the first changed one (and the 2nd newest of the 9)
    same = [k for k in chunks if int(k.rsplit("/", 1)[1]) < c0]
    assert same and all(first[k] == second[k] for k in same)  # the frozen prefix was not touched
    changed = [k for k in second if "/VAVH/c/" in k and first.get(k) != second[k]]
    assert changed and len(changed) < len(second) // 2

    info = json.loads(s3.get_object(Bucket="bkt", Key="alt/track/dataset.json")["Body"].read())
    assert info["n"] == (NW - 1) * 60 + 90 + 60 and len(info["windows"]) == NW + 1


def test_a_store_without_the_current_encoding_is_uploaded_again(m):
    ws = [{"start": f"2026082{k}T000000", "version": "v", "n": 100, "offset": k * 100} for k in range(10)]
    info = dict(chunk=64, time_chunk=256, vars=m.vars, scales=m.scales, delta=track.delta_arrays(m), windows=ws, n=1000)
    prev = {**info, "kind": "track"}
    assert track.plan_upload(prev, info, 2, set(), set()) == 800  # only the tail
    assert track.plan_upload({k: v for k, v in prev.items() if k != "delta"}, info, 2, set(), set()) == 0


def test_x3d_reads_the_track_store(tmp_path, m):
    pytest.importorskip("x3d")
    for k in range(3):
        window(tmp_path, k, 4, 100)
    track.convert_track(m, tmp_path / "out", now=NOW)
    for a in ("time", "latitude", "longitude"):  # delta-coded; the reader undoes it
        codecs = json.loads((tmp_path / f"out/altimeter_s3a/track/track.zarr/{a}/zarr.json").read_text())["codecs"]
        assert [c["name"] for c in codecs] == ["numcodecs.delta", "bytes", "zstd"]
    ds = xr.open_dataset(f"x3d+file://{tmp_path / 'out/altimeter_s3a'}")  # the layout is detected
    assert ds.sizes == {"time": 300} and {"latitude", "longitude"} <= set(ds.coords)
    assert -180 <= float(ds.longitude.min()) and float(ds.longitude.max()) < 180
    assert ds.attrs["x3d_layout"] == "track"
    np.testing.assert_allclose(ds.VAVH.values[:100], 1.1, atol=1e-6)
    assert np.isnan(ds.WIND_SPEED.values[0]) and (ds.WIND_SPEED.values[1:100] == 5).all()
    np.testing.assert_allclose(ds.latitude.values[:100], np.linspace(-80, 80, 100), atol=1e-6)
    assert (np.diff(ds.time.values.astype("int64")) > 0).all()
    raw = xr.open_dataset(f"x3d+file://{tmp_path / 'out/altimeter_s3a'}", decode_times=False)
    assert raw.time.dtype == "int64" and raw.time.attrs["units"] == "seconds since 1970-01-01"
    np.testing.assert_array_equal(raw.time.values * 10**9, ds.time.values.astype("int64"))


def test_series_is_cut_at_the_start_only_after_retention_and_trim(m):
    prev = {"kind": "track", "windows": [{"start": "20260827T000000", "version": "v"}]}
    assert track.series_start(m, NOW, prev) == dt.datetime(2026, 8, 29)  # 27th is older than 5 + 1 days: cut
    prev["windows"][0]["start"] = "20260828T000000"
    assert track.series_start(m, NOW, prev) == dt.datetime(2026, 8, 28)  # still within retention + trim: kept
    assert track.series_start(m, NOW, None) == dt.datetime(2026, 8, 29)


def test_unlimited_history_backfills_and_rejects_missing_published_windows(tmp_path, m):
    m = dataclasses.replace(m, retention_days=0)
    window(tmp_path, 0, 4, 30)
    window(tmp_path, 8 * 40, 4, 30)
    now = T0 + dt.timedelta(days=45)
    assert track.convert_track(m, tmp_path / "out", now=now) == 0
    info = json.loads((tmp_path / "out/altimeter_s3a/track/dataset.json").read_text())
    assert [w["start"] for w in info["windows"]] == ["20260901T000000", "20261011T000000"]
    assert track.convert_track(m, tmp_path / "out", now=now, previous=info, check=True) == UP_TO_DATE
    info["windows"].append({"start": "20260902T000000", "version": "v"})
    with pytest.raises(SystemExit, match="refusing to drop 1 previously published windows"):
        track.convert_track(m, tmp_path / "out", now=now, previous=info)


def test_check_only_looks_at_the_recent_windows(tmp_path, m):
    m = dataclasses.replace(m, retention_days=0, check_days=1)
    for k in (0, 8 * 40, 8 * 40 + 1):
        window(tmp_path, k, 4, 30)
    now = T0 + dt.timedelta(days=40, hours=5)
    assert track.convert_track(m, tmp_path / "out", now=now) == 0
    info = json.loads((tmp_path / "out/altimeter_s3a/track/dataset.json").read_text())
    assert track.convert_track(m, tmp_path / "out", now=now, previous=info, check=True) == UP_TO_DATE
    window(tmp_path, 0, 6, 30)  # an old window is re-issued: only a full run notices
    assert track.convert_track(m, tmp_path / "out", now=now, previous=info, check=True) == UP_TO_DATE
    window(tmp_path, 8 * 40 + 1, 6, 30)  # the newest window is re-issued
    assert track.convert_track(m, tmp_path / "out", now=now, previous=info, check=True) == 0


def test_check_does_not_trust_a_source_without_recent_files(tmp_path, m):
    m = dataclasses.replace(m, retention_days=0, check_days=1)
    window(tmp_path, 0, 4, 30)
    window(tmp_path, 8 * 40, 4, 30)
    now = T0 + dt.timedelta(days=40, hours=5)
    assert track.convert_track(m, tmp_path / "out", now=now) == 0
    info = json.loads((tmp_path / "out/altimeter_s3a/track/dataset.json").read_text())
    later = now + dt.timedelta(days=30)  # nothing newer than the published windows: the full comparison runs, still current
    assert track.convert_track(m, tmp_path / "out", now=later, previous=info, check=True) == UP_TO_DATE
    gone = dataclasses.replace(m, files=[str(tmp_path / "missing/%Y/%m/s3a_*.nc")])  # an unmounted source is an error
    with pytest.raises(SystemExit, match="no usable files"):
        track.convert_track(gone, tmp_path / "out", now=later, previous=info, check=True)


def test_only_recent_files_are_checked_for_being_empty(tmp_path, m):
    for k in (0, 8 * 40):
        window(tmp_path, k, 4, 30)
        window(tmp_path, k, 6, 30)
    now = T0 + dt.timedelta(days=40, hours=5)
    files = track.windows_available(m, None, now)
    for f in (files["20260901T000000"][0], files["20261011T000000"][0]):
        Path(f).write_bytes(b"")
    again = track.windows_available(m, None, now)
    assert again["20260901T000000"][0] == files["20260901T000000"][0]  # old: trusted without a stat
    assert again["20261011T000000"] == files["20261011T000000"][1:]  # recent: empty, the older version is used


def test_unlimited_history_rejects_unreadable_windows(tmp_path, m):
    m = dataclasses.replace(m, retention_days=0)
    window(tmp_path, 0, 4, 30)
    window(tmp_path, 1, 4, 30)
    files = track.windows_available(m, None, NOW)
    Path(files["20260901T030000"][0]).write_bytes(b"invalid netcdf")
    with pytest.raises(SystemExit, match="refusing to omit 1 unreadable windows"):
        track.convert_track(m, tmp_path / "out", now=NOW)


def test_a_series_that_lost_most_samples_is_rejected(tmp_path, m):
    for k in range(3):
        window(tmp_path, k, 4, 100)
    track.convert_track(m, tmp_path / "out", now=NOW)
    prev = json.loads((tmp_path / "out/altimeter_s3a/track/dataset.json").read_text())
    prev["n"] *= 3
    with pytest.raises(ValidationError, match="more than half"):
        from x3d_publisher.validate import validate_track

        validate_track(tmp_path / "out/altimeter_s3a/track", m, prev)
