"""Validation before publishing, and the redundant lustre sources (A/B)."""
import json
import os
import shutil
import time

import boto3
import pytest
import xarray as xr
from moto import mock_aws

from conftest import synthetic
from x3d_publisher import cli, publish
from x3d_publisher import convert as cv
from x3d_publisher import model as models
from x3d_publisher.validate import ValidationError, validate_store


@pytest.fixture
def store(tmp_path, ww3, source):
    cv.convert(ww3, tmp_path / "store", url=source, run="r1")
    return tmp_path / "store" / "ww3_4km"


def run_dir(store):
    return store / "r1"


def test_good_store_passes(store, ww3):
    assert validate_store(store, model=ww3) == []


def edit_json(p, fn):
    d = json.loads(p.read_text())
    fn(d)
    p.write_text(json.dumps(d))


@pytest.mark.parametrize("name,corrupt,expect", [
    ("missing grid", lambda r: (r / "grid.bin").unlink(), "grid.bin"),
    ("short grid", lambda r: (r / "grid.bin").write_bytes(b"\0" * 8), "grid.bin"),
    ("time order", lambda r: edit_json(r / "dataset.json", lambda d: d["times"].reverse()), "increasing"),
    ("time format", lambda r: edit_json(r / "dataset.json", lambda d: d.update(times=["2026-01-01"] + d["times"][1:])), "ISO"),
    ("shape", lambda r: edit_json(r / "dataset.json", lambda d: d.update(nx=d["nx"] + 1)), "shape"),
    ("missing array", lambda r: shutil.rmtree(r / "map.zarr" / "hs"), "map.zarr/hs"),
    ("no sharding", lambda r: edit_json(r / "map.zarr" / "hs" / "zarr.json", lambda d: d["codecs"].pop(0)), "shard"),
    ("dtype", lambda r: edit_json(r / "ts.zarr" / "hs" / "zarr.json", lambda d: d.update(data_type="float32")), "int16"),
])
def test_corrupt_metadata_is_rejected(store, ww3, name, corrupt, expect):
    corrupt(run_dir(store))
    with pytest.raises(ValidationError, match=expect):
        validate_store(store, model=ww3)


def test_all_fill_time_step_is_rejected(store, ww3):
    import zarr

    a = zarr.open_group(run_dir(store) / "map.zarr", mode="r+")["hs"]
    a[1] = -32768  # a step whose source files were missing
    with pytest.raises(ValidationError, match="no valid data"):
        validate_store(store, model=ww3)


def test_implausible_values_are_rejected(store, ww3):
    import zarr

    a = zarr.open_group(run_dir(store) / "map.zarr", mode="r+")["hs"]
    a[0, 20:30, 20:30] = 25000  # 250 m
    with pytest.raises(ValidationError, match="plausible"):
        validate_store(store, model=ww3)


def test_much_shorter_run_than_published_is_rejected(store, ww3):
    published = json.loads((run_dir(store) / "dataset.json").read_text())
    published["times"] = published["times"] * 10
    with pytest.raises(ValidationError, match="published run"):
        validate_store(store, model=ww3, published=published)


def test_cli_validate_exit_code(store, capsys):
    assert cli.main(["validate", str(store), "--model", "ww3_4km"]) == 0
    (run_dir(store) / "grid.bin").unlink()
    assert cli.main(["validate", str(store), "--model", "ww3_4km"]) == 1


@pytest.fixture
def s3(monkeypatch):
    for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        monkeypatch.setenv(k, "test")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    with mock_aws():
        c = boto3.client("s3")
        c.create_bucket(Bucket="bkt")
        yield c


def test_publish_uploads_nothing_when_invalid(store, s3):
    (run_dir(store) / "grid.bin").unlink()
    with pytest.raises(ValidationError):
        publish.publish(store, "s3://bkt/ww3_4km")
    assert "Contents" not in s3.list_objects_v2(Bucket="bkt")


def test_run_does_not_publish_an_invalid_store(tmp_path, ww3, source, s3, monkeypatch):
    real = cv.write_sidecars

    def broken(*a, **kw):
        real(*a, **kw)
        (a[1] / "grid.bin").unlink()

    monkeypatch.setattr(cv, "write_sidecars", broken)
    with pytest.raises(ValidationError):
        cli.main(["run", "ww3_4km", "--url", source, "--bucket-url", "s3://bkt", "--force", "--work", str(tmp_path / "w")])
    assert "Contents" not in s3.list_objects_v2(Bucket="bkt")


def test_latest_is_not_moved_when_an_upload_is_incomplete(store, s3, monkeypatch):
    publish.publish(store, "s3://bkt/ww3_4km")  # r1 is live
    cv_dir = store.parent
    cv.convert(models.load("ww3_4km"), cv_dir, url=str(cv_dir.parent / "source.nc"), run="r2")
    real = s3.upload_file

    def lossy(path, bucket, key, ExtraArgs=None):
        if key.endswith("/grid.bin") and "r2" in key:
            return  # silently lost
        return real(path, bucket, key, ExtraArgs=ExtraArgs)

    monkeypatch.setattr(s3, "upload_file", lossy)
    with pytest.raises(RuntimeError, match="NOT updated"):
        publish.publish(store, "s3://bkt/ww3_4km", run="r2", s3=s3)
    assert json.loads(s3.get_object(Bucket="bkt", Key="ww3_4km/latest.json")["Body"].read()) == {"run": "r1"}


# --- redundant sources: the production systems A and B write the same files

def nk_files(tmp_path, side, names, age=3600):
    d = tmp_path / side
    d.mkdir(exist_ok=True)
    m = models.load("norkyst800_lustre")
    ds = synthetic(models.load("ww3_4km"), nt=1)[["hs"]].rename(hs="x")
    for n in names:
        ds.to_netcdf(d / n)
        os.utime(d / n, (time.time() - age, time.time() - age))
    return d


def lustre_model(tmp_path, *sides, per_run=3):
    import dataclasses

    m = models.load("norkyst800_lustre")
    return dataclasses.replace(
        m, files=[str(tmp_path / s / "norkyst800_his_zdepth_*_m00_*.nc") for s in sides], files_runs=1, files_per_run=per_run,
        files_min_age=60, vars=["x"], select={},
    )


def test_file_missing_on_one_side_is_taken_from_the_other(tmp_path):
    n = [f"norkyst800_his_zdepth_20260101T00Z_m00_{k}.nc" for k in ("AN", "FC_0001", "FC_0002")]
    nk_files(tmp_path, "A", n[:2])
    nk_files(tmp_path, "B", n[1:])
    m = lustre_model(tmp_path, "A", "B")
    found = {}
    import glob, re
    for g in m.file_globs:
        for f in glob.glob(g):
            found.setdefault(os.path.basename(f), []).append(f)
    assert len(found) == 3  # A and B together cover the run
    _, desc = cv.open_files(m, ["x"])
    assert "3 files" in desc


def test_incomplete_run_on_both_sides_is_skipped(tmp_path):
    n = [f"norkyst800_his_zdepth_20260101T00Z_m00_{k}.nc" for k in ("AN", "FC_0001", "FC_0002")]
    nk_files(tmp_path, "A", n[:2])
    nk_files(tmp_path, "B", n[:1])
    with pytest.raises(SystemExit, match="no complete model run"):
        cv.open_files(lustre_model(tmp_path, "A", "B"), ["x"])


def test_empty_file_on_the_first_side_falls_back_to_the_second(tmp_path):
    n = [f"norkyst800_his_zdepth_20260101T00Z_m00_{k}.nc" for k in ("AN", "FC_0001", "FC_0002")]
    nk_files(tmp_path, "A", n)
    nk_files(tmp_path, "B", n)
    (tmp_path / "A" / n[1]).write_bytes(b"")
    _, desc = cv.open_files(lustre_model(tmp_path, "A", "B"), ["x"])
    assert "3 files" in desc


def test_ww3_lustre_names_and_runs(tmp_path):
    import dataclasses

    wm = models.load("ww3_4km_lustre")
    runs = ("20260101T00Z", "20260101T06Z")
    names = [f"{v}_ww3_4km_b{r}.nc" for r in runs for v in ("20260101T00Z", "20260101T06Z")]
    nk_files(tmp_path, "A", names + ["ww3_20260101T06Z.nc"])  # the whole-run file is not a piece
    m = dataclasses.replace(wm, files=[str(tmp_path / "A" / "*_ww3_4km_b*.nc")], files_runs=2, files_per_run=2, files_min_age=60)
    _, desc = cv.open_files(m, ["x"])
    assert "4 files, runs 20260101T00Z .. 20260101T06Z" in desc
