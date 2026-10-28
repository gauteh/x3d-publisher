"""Download the current run of a store from S3 into a local directory (the inverse of publish).

Anonymous by default (the bucket allows public GetObject/ListBucket; `endpoint_url` names the server).
With an s3cmd-style `s3cfg` ([default] access_key / secret_key / host_base / use_https) or `signed`
(AWS env credentials) requests are signed instead.
"""
from __future__ import annotations

import configparser
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .publish import split


def client(endpoint, s3cfg=None, signed=False):
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    if not s3cfg:
        return boto3.client("s3", endpoint_url=endpoint, **({} if signed else {"config": Config(signature_version=UNSIGNED)}))
    c = configparser.ConfigParser()
    c.read(s3cfg)
    d = c["default"]
    if not endpoint and d.get("host_base"):
        endpoint = ("https://" if d.getboolean("use_https", True) else "http://") + d["host_base"]
    return boto3.client("s3", endpoint_url=endpoint, aws_access_key_id=d["access_key"], aws_secret_access_key=d["secret_key"])


def fetch(src: str, dest, *, endpoint_url=None, s3cfg=None, signed=False, workers=16, s3=None) -> str:
    """Download the run `src` (s3://bucket/prefix, containing latest.json) points to into `dest`."""
    bucket, prefix = split(src)
    s3 = s3 or client(endpoint_url, s3cfg, signed)
    dest = Path(dest)
    latest = s3.get_object(Bucket=bucket, Key=f"{prefix}/latest.json")["Body"].read()
    run = json.loads(latest)["run"]
    keys = [o["Key"] for pg in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=f"{prefix}/{run}/") for o in pg.get("Contents", [])]
    if not keys:
        raise SystemExit(f"run {run} has no objects under {src}")

    def get(k):
        f = dest / k[len(prefix) + 1:]
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(s3.get_object(Bucket=bucket, Key=k)["Body"].read())

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(get, keys))
    (dest / "latest.json").write_bytes(latest)  # last: the store is only complete once this exists
    print(f"fetched {len(keys)} objects, run {run} -> {dest}")
    return run
