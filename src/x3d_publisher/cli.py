"""Command line: x3d-publish {models,convert,publish,fetch,run}."""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

from . import model as models

env = os.environ.get


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("zarr", "fsspec", "asyncio", "urllib3", "botocore", "boto3", "s3transfer"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if not verbose:
        warnings.filterwarnings("ignore")


def add_convert_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("model", help="bundled model name (see `models`) or a model .toml file")
    p.add_argument("--url", help="override the source dataset URL")
    p.add_argument("--name", help="store name (default: the model)")
    p.add_argument("--start", help="ISO start time (default: the last --hours)")
    p.add_argument("--hours", type=int, default=int(env("HOURS", 0)) or None, help="time steps to keep (default: the model's, or $HOURS)")
    p.add_argument("--run", help="run id (default: UTC conversion time)")
    p.add_argument("--vars", nargs="+", help="variables of the field layouts (default: the model's)")
    p.add_argument("--published", help="base URL of the published store (<base>/latest.json); exit 3 without converting if it already has these time steps")
    p.add_argument("--workers", type=int, default=int(env("WORKERS", 4)), help="threads for dask/zarr (default 4, or $WORKERS)")


def cmd_serve(a) -> int:
    from . import serve

    print(f"serving {a.dir} on http://{a.host}:{a.port}", file=sys.stderr)
    try:
        serve.serve(a.dir, a.port, host=a.host)
    except KeyboardInterrupt:
        pass
    return 0


def cmd_models(a) -> int:
    for n in models.bundled():
        m = models.load(n)
        print(f"{n}: {m.url}")
    return 0


def cmd_convert(a) -> int:
    from .convert import convert

    return convert(models.load(a.model), a.out, url=a.url, name=a.name, start=a.start, hours=a.hours, run=a.run,
                   vars=a.vars, published=a.published, workers=a.workers)


def cmd_publish(a) -> int:
    from .publish import publish

    if not a.dest:
        raise SystemExit("destination required (argument or $S3_DEST)")
    if a.keep is not None and a.keep < 1:
        raise SystemExit("--keep must be >= 1")
    publish(a.src, a.dest, endpoint_url=a.endpoint_url, keep=a.keep, grace_hours=a.grace_hours, dry_run=a.dry_run,
            run=a.run, configure=a.configure_bucket, workers=a.workers)
    return 0


def cmd_fetch(a) -> int:
    from .fetch import fetch

    fetch(a.src, a.dest, endpoint_url=a.endpoint_url, s3cfg=a.s3cfg, signed=a.signed, workers=a.workers)
    return 0


def cmd_run(a) -> int:
    """One cycle: convert -> publish -> prune, in a private scratch directory that is always removed."""
    from .convert import UP_TO_DATE, convert
    from .publish import publish

    log = logging.getLogger("x3d_publisher.run")
    if not a.bucket_url:
        raise SystemExit("set --bucket-url (or $S3_BUCKET_URL) to s3://bucket[/prefix]")
    m = models.load(a.model)
    name = a.name or m.name
    bucket_url = a.bucket_url.rstrip("/")
    pub = None
    if not a.force:
        pub = a.published_url or (a.endpoint_url and f"{a.endpoint_url.rstrip('/')}/{bucket_url[len('s3://'):]}")
    Path(a.work).mkdir(parents=True, exist_ok=True)
    # always a fresh private subdirectory: only that is ever deleted, and overlapping runs cannot clash
    work = Path(tempfile.mkdtemp(prefix="run.", dir=a.work))
    try:
        log.info("converting %s (%s h) -> %s", name, a.hours or m.hours, work)
        rc = convert(m, work, url=a.url, name=name, hours=a.hours, vars=a.vars, workers=a.workers,
                     published=f"{pub.rstrip('/')}/{name}" if pub else None)
        if rc == UP_TO_DATE:
            log.info("up to date, nothing to publish")
            return 0
        log.info("publishing to %s/%s", bucket_url, name)
        publish(work / name, f"{bucket_url}/{name}", endpoint_url=a.endpoint_url, keep=a.keep,
                configure=a.configure_bucket, workers=a.upload_workers)
        log.info("done")
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="x3d-publish", description=__doc__)
    p.add_argument("-v", "--verbose", action="store_true", help="debug output")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("models", help="list the bundled model configs").set_defaults(fn=cmd_models)

    v = sub.add_parser("serve", help="serve a local store over HTTP (CORS, Range, cache headers) as an S3 stand-in")
    v.add_argument("dir", nargs="?", default="store")
    v.add_argument("port", nargs="?", type=int, default=8000)
    v.add_argument("--host", default="127.0.0.1")
    v.set_defaults(fn=cmd_serve)

    c = sub.add_parser("convert", help="source dataset -> local store <out>/<name>/<run>/")
    add_convert_args(c)
    c.add_argument("--out", default="store")
    c.set_defaults(fn=cmd_convert)

    u = sub.add_parser("publish", help="upload a local store to S3 (data, then metadata, then latest.json), optionally prune")
    u.add_argument("src", help="local store directory containing latest.json and run directories")
    u.add_argument("dest", nargs="?", default=env("S3_DEST"), help="s3://bucket/prefix (default: $S3_DEST)")
    u.add_argument("--endpoint-url", default=env("S3_ENDPOINT_URL"))
    u.add_argument("--keep", type=int, help="after publishing, delete all but the newest N runs")
    u.add_argument("--grace-hours", type=float, default=6, help="never prune runs younger than this")
    u.add_argument("--dry-run", action="store_true", help="only list what --keep would delete")
    u.add_argument("--run", help="run to publish (default: the one in latest.json)")
    u.add_argument("--configure-bucket", action="store_true", help="set public-read policy and CORS (GET/HEAD, Range) on the bucket")
    u.add_argument("--workers", type=int, default=8)
    u.set_defaults(fn=cmd_publish)

    f = sub.add_parser("fetch", help="download the current run of a published store")
    f.add_argument("src", help="s3://bucket/prefix of one store (contains latest.json)")
    f.add_argument("dest", help="local directory")
    f.add_argument("--endpoint-url", default=env("S3_ENDPOINT_URL"))
    f.add_argument("--s3cfg")
    f.add_argument("--signed", action="store_true", help="sign requests with AWS env credentials")
    f.add_argument("--workers", type=int, default=16)
    f.set_defaults(fn=cmd_fetch)

    r = sub.add_parser("run", help="one cycle: convert -> publish -> prune (cron / CronJob entrypoint)")
    add_convert_args(r)
    r.add_argument("--bucket-url", default=env("S3_BUCKET_URL"), help="s3://bucket[/prefix]; the store goes to <that>/<name> (default: $S3_BUCKET_URL)")
    r.add_argument("--endpoint-url", default=env("S3_ENDPOINT_URL"))
    r.add_argument("--published-url", default=env("PUBLISHED_URL"), help="public base URL of the bucket, to skip up-to-date runs (default: <endpoint>/<bucket>)")
    r.add_argument("--force", action="store_true", default=bool(env("FORCE")), help="convert even if the published store is up to date ($FORCE)")
    r.add_argument("--keep", type=int, default=int(env("KEEP_RUNS", 3)), help="runs to keep (default 3, or $KEEP_RUNS)")
    r.add_argument("--configure-bucket", action="store_true", default=bool(env("CONFIGURE_BUCKET")))
    r.add_argument("--work", default=env("WORK", tempfile.gettempdir() + "/x3d-publisher"), help="scratch parent directory ($WORK)")
    r.add_argument("--upload-workers", type=int, default=8)
    r.set_defaults(fn=cmd_run)
    return p


def main(argv=None) -> int:
    a = parser().parse_args(argv)
    setup_logging(a.verbose)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
