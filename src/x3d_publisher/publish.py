"""Publish a converted run to S3 (or any S3-compatible store) with CDN-friendly cache headers.

Uploads <dir>/<run>/ (immutable, cached forever), data objects first and metadata last, then
swaps <dir>/latest.json (short TTL). A reader therefore never sees a half-written run.
`prune` then removes old runs: the newest N stay, as does the run latest.json points to; runs younger
than the grace period are never removed (clients may still hold a cached latest.json).

Credentials come from the usual AWS environment/config. The bucket needs public read (GetObject,
ListBucket) and a CORS rule allowing GET/HEAD with the Range header; `configure_bucket` sets both
(or do it in your own IaC). `publish` and `run` do this by default (idempotent; `--no-configure-bucket` to skip),
but the bucket itself must exist: create it first, e.g. `s3cmd -c s3cfg mb s3://BUCKET`.
"""
from __future__ import annotations

import datetime as dt
import json
import mimetypes
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

IMMUTABLE = "public, max-age=31536000, immutable"
LATEST = "public, max-age=60"
METADATA = ("zarr.json", "dataset.json", "grid.bin")


def split(dest: str) -> tuple[str, str]:
    u = urlparse(dest)
    return u.netloc, u.path.strip("/")


def configure_bucket(s3, bucket: str) -> None:
    s3.put_bucket_policy(Bucket=bucket, Policy=json.dumps({
        "Version": "2012-10-17",
        "Statement": [
            {"Effect": "Allow", "Principal": "*", "Action": "s3:GetObject", "Resource": f"arn:aws:s3:::{bucket}/*"},
            {"Effect": "Allow", "Principal": "*", "Action": "s3:ListBucket", "Resource": f"arn:aws:s3:::{bucket}"},
        ],
    }))
    s3.put_bucket_cors(Bucket=bucket, CORSConfiguration={"CORSRules": [{
        "AllowedOrigins": ["*"], "AllowedMethods": ["GET", "HEAD"], "AllowedHeaders": ["Range"],
        "ExposeHeaders": ["Content-Range", "Content-Length", "ETag"], "MaxAgeSeconds": 86400,
    }]})


def publish(
    src, dest: str, *, endpoint_url: str | None = None, keep: int | None = None, grace_hours: float = 6,
    dry_run: bool = False, run: str | None = None, configure: bool = True, workers: int = 8, s3=None,
) -> str:
    """Upload run `run` (default: the one in <src>/latest.json) of the local store `src` to `dest` (s3://bucket/prefix)."""
    import boto3

    src = Path(src)
    run = run or json.loads((src / "latest.json").read_text())["run"]
    bucket, prefix = split(dest)
    s3 = s3 or boto3.client("s3", endpoint_url=endpoint_url)
    if configure:
        try:
            configure_bucket(s3, bucket)
        except Exception as e:  # e.g. credentials without bucket-policy rights; the bucket may be set up already
            print(f"warning: could not set the public-read policy/CORS on s3://{bucket}: {e}", flush=True)

    def put(path: Path, key: str, cache: str):
        ctype = "application/json" if path.suffix == ".json" else mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        s3.upload_file(str(path), bucket, key, ExtraArgs={"CacheControl": cache, "ContentType": ctype})

    files = [f for f in (src / run).rglob("*") if f.is_file()]
    key = lambda f: f"{prefix}/{f.relative_to(src)}"
    data = [f for f in files if f.name not in METADATA]
    meta = [f for f in files if f.name in METADATA]
    total, t0, done = len(files), time.time(), [0]
    lock = threading.Lock()

    def put_counted(f):
        put(f, key(f), IMMUTABLE)
        with lock:
            done[0] += 1
            if done[0] % max(1, total // 20) == 0 or done[0] == total:
                print(f"{time.strftime('%H:%M:%S')} uploaded {done[0]}/{total} objects ({100 * done[0] // total}%, {time.time() - t0:.0f}s)", flush=True)

    print(f"{time.strftime('%H:%M:%S')} uploading run {run}: {total} objects to s3://{bucket}/{prefix}/ ({endpoint_url or 'default endpoint'})", flush=True)
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(put_counted, data))
        list(ex.map(put_counted, meta))
    s3.put_object(Bucket=bucket, Key=f"{prefix}/latest.json", Body=json.dumps({"run": run}).encode(),
                  CacheControl=LATEST, ContentType="application/json")
    print(f"published {len(files)} objects, latest -> {run}")
    if keep:
        prune(s3, bucket, prefix, run, keep, grace_hours, dry_run)
    return run


def prune(s3, bucket, prefix, current, keep, grace_hours, dry):
    runs = {}  # run id -> (keys, newest modification)
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix + "/"):
        for o in page.get("Contents", []):
            run = o["Key"][len(prefix) + 1:].split("/", 1)[0]
            if run == "latest.json":
                continue
            keys, t = runs.get(run, ([], o["LastModified"]))
            keys.append(o["Key"])
            runs[run] = (keys, max(t, o["LastModified"]))
    # run ids are UTC timestamps, so lexical order is chronological
    protected = set(sorted(runs, reverse=True)[:keep]) | {current}
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=grace_hours)
    for run in sorted(set(runs) - protected):
        keys, newest = runs[run]
        if newest > cutoff:
            continue
        print(f"{'would delete' if dry else 'deleting'} run {run} ({len(keys)} objects)")
        if not dry:
            for i in range(0, len(keys), 1000):
                s3.delete_objects(Bucket=bucket, Delete={"Objects": [{"Key": k} for k in keys[i:i + 1000]], "Quiet": True})
