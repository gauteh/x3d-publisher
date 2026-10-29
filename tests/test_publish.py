"""S3 pruning, publish/fetch round trip and the `run` cycle, against a mocked S3 (moto)."""
import datetime as dt
import json

import boto3
import pytest
from moto import mock_aws

from x3d_publisher import cli, fetch, publish


class FakeS3:
    """Minimal stand-in for the boto3 calls used by publish.prune."""

    def __init__(self, runs):
        now = dt.datetime.now(dt.timezone.utc)
        self.objs = {f"p/{r}/ts.zarr/zarr.json": now - dt.timedelta(hours=age) for r, age in runs.items()}
        self.objs["p/latest.json"] = now
        self.deleted = []

    def get_paginator(self, _):
        outer = self

        class P:
            def paginate(self, **_):
                yield {"Contents": [{"Key": k, "LastModified": t} for k, t in outer.objs.items()]}

        return P()

    def delete_objects(self, Bucket, Delete):
        self.deleted += [o["Key"] for o in Delete["Objects"]]


def test_prune_keeps_newest_and_current():
    runs = {"20260101T000000Z": 100, "20260102T000000Z": 90, "20260103T000000Z": 80, "20260104T000000Z": 70, "20260105T000000Z": 60}
    s3 = FakeS3(runs)
    publish.prune(s3, "b", "p", current="20260102T000000Z", keep=2, grace_hours=6, dry=False)
    # kept: newest two (04, 05) and the current run (02)
    assert sorted(s3.deleted) == ["p/20260101T000000Z/ts.zarr/zarr.json", "p/20260103T000000Z/ts.zarr/zarr.json"]


def test_prune_respects_grace_period_and_dry_run():
    s3 = FakeS3({"20260101T000000Z": 100, "20260102T000000Z": 2, "20260103T000000Z": 1})
    publish.prune(s3, "b", "p", current="20260103T000000Z", keep=1, grace_hours=6, dry=False)
    assert s3.deleted == ["p/20260101T000000Z/ts.zarr/zarr.json"]  # 02 is younger than the grace period
    s3 = FakeS3({"20260101T000000Z": 100, "20260103T000000Z": 1})
    publish.prune(s3, "b", "p", current="20260103T000000Z", keep=1, grace_hours=6, dry=True)
    assert s3.deleted == []


@pytest.fixture
def s3(monkeypatch):
    for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        monkeypatch.setenv(k, "test")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    with mock_aws():
        c = boto3.client("s3")
        c.create_bucket(Bucket="bkt")
        yield c


def converted(tmp_path, ww3, source, run="r1"):
    from x3d_publisher.convert import convert

    out = tmp_path / "store"
    convert(ww3, out, url=source, run=run)
    return out / "ww3_4km"


def test_publish_headers_and_fetch_roundtrip(tmp_path, ww3, source, s3):
    local = converted(tmp_path, ww3, source)
    publish.publish(local, "s3://bkt/ww3_4km")
    latest = s3.head_object(Bucket="bkt", Key="ww3_4km/latest.json")
    assert "max-age=60" in latest["CacheControl"]
    assert "immutable" in s3.head_object(Bucket="bkt", Key="ww3_4km/r1/dataset.json")["CacheControl"]
    assert json.loads(s3.get_object(Bucket="bkt", Key="ww3_4km/latest.json")["Body"].read()) == {"run": "r1"}
    fetch.fetch("s3://bkt/ww3_4km", tmp_path / "back", s3=s3)
    mine = sorted(p.relative_to(local) for p in local.rglob("*") if p.is_file())
    assert mine == sorted(p.relative_to(tmp_path / "back") for p in (tmp_path / "back").rglob("*") if p.is_file())


def test_publish_sets_public_read_only_policy_by_default(tmp_path, ww3, source, s3):
    publish.publish(converted(tmp_path, ww3, source), "s3://bkt/ww3_4km")
    actions = {st["Action"] for st in json.loads(s3.get_bucket_policy(Bucket="bkt")["Policy"])["Statement"]}
    assert actions == {"s3:GetObject", "s3:ListBucket"}
    assert s3.get_bucket_cors(Bucket="bkt")["CORSRules"][0]["AllowedMethods"] == ["GET", "HEAD"]


def test_publish_can_skip_bucket_configuration(tmp_path, ww3, source, s3):
    publish.publish(converted(tmp_path, ww3, source), "s3://bkt/ww3_4km", configure=False)
    with pytest.raises(Exception):
        s3.get_bucket_policy(Bucket="bkt")


def test_publish_prunes_old_runs(tmp_path, ww3, source, s3):
    for run in ("r1", "r2", "r3"):
        local = converted(tmp_path, ww3, source, run)
        publish.publish(local, "s3://bkt/ww3_4km", run=run, keep=2, grace_hours=0)
    keys = {o["Key"].split("/")[1] for o in s3.list_objects_v2(Bucket="bkt")["Contents"]}
    assert keys == {"latest.json", "r2", "r3"}


def test_run_cycle(tmp_path, ww3, source, s3):
    work = tmp_path / "work"
    rc = cli.main(["run", "ww3_4km", "--url", source, "--bucket-url", "s3://bkt", "--force", "--work", str(work)])
    assert rc == 0
    run = json.loads(s3.get_object(Bucket="bkt", Key="ww3_4km/latest.json")["Body"].read())["run"]
    assert s3.head_object(Bucket="bkt", Key=f"ww3_4km/{run}/ts.zarr/zarr.json")
    assert list(work.iterdir()) == []  # the scratch directory is removed
