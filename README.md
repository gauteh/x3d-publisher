# x3d-publisher

Producer of the [x3d](https://github.com/gauteh/x3d) data store for [waves.met.no](https://github.com/gauteh/waves.met.no)
(optimized for both Python and wasm/web access): converts a source dataset (THREDDS OPeNDAP, or anything xarray
opens) into the Zarr v3 layouts described in x3d's `docs/format.md`, publishes them to S3 (data first,
metadata later, `latest.json` last) and prunes old runs. x3d itself is a strict client; this package is the
only writer. It contains no credentials or bucket names: those are deployment settings.

    mamba activate x3d                       # the x3d environment (`environment.yml` in x3d) has everything
    pip install -e . --no-deps               # x3d-publish on PATH
    pytest

## Use

    x3d-publish models                                   # bundled model configs
    x3d-publish convert ww3_4km --out store --hours 72   # -> store/ww3_4km/<run>/ + latest.json
    x3d-publish publish store/ww3_4km s3://BUCKET/ww3_4km --endpoint-url URL --keep 3 [--no-configure-bucket]
    x3d-publish fetch s3://BUCKET/ww3_4km local/ww3_4km  # the inverse (anonymous by default)
    x3d-publish serve store 8000                         # local S3 stand-in over HTTP (CORS, Range, cache headers)
    x3d-publish run ww3_4km --bucket-url s3://BUCKET     # convert -> publish -> prune, in a scratch dir

`run` is the cron/CronJob entrypoint. It exits 0 without publishing when the published store (checked over
`--published-url`, default `<endpoint>/<bucket>`) already has the same time steps; `--force` converts anyway.
Credentials are the usual AWS environment (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, ...); how they are
found is up to the caller. Env equivalents: `S3_BUCKET_URL`, `S3_ENDPOINT_URL`,
`PUBLISHED_URL`, `KEEP_RUNS`, `HOURS`, `WORKERS`, `WORK`, `FORCE`, `CONFIGURE_BUCKET` (default on).

## Creating a bucket

`publish` and `run` need an existing bucket; they do not create it. With s3cmd (credentials in `s3cfg`; bucket names must be DNS-compatible: lowercase, dashes, no underscores):

    s3cmd -c s3cfg mb s3://norkyst800m-v3
    x3d-publish run norkyst800 --bucket-url s3://norkyst800m-v3 --endpoint-url https://rgw.met.no

On every publish the bucket is made readable by everyone and read-only: the policy allows only anonymous
`s3:GetObject` and `s3:ListBucket`, writes need credentials. CORS allows `GET`/`HEAD` with `Range` from any origin
(needed for browsers). `--no-configure-bucket` (or `CONFIGURE_BUCKET=0`) skips this, and a failure to set it
(credentials without policy rights) is a warning, not an error. Check it with
`s3cmd -c s3cfg info s3://norkyst800m-v3` (policy, CORS) or `curl <endpoint>/<project>:<bucket>/norkyst800/latest.json`.
The public URL is `<endpoint>/<project-id>:<bucket>/<prefix>`, which is what x3d opens:
`xr.open_dataset("x3d+https://rgw.met.no/<project-id>:norkyst800m-v3/norkyst800")`.

## Models

A model is one TOML file in `src/x3d_publisher/models/` (or any path given instead of the name): source URL,
dimension names, variables, overview variables, `int16` scale factors and the layout parameters. `ww3_4km`
and `norkyst800` (NorKyst v3 surface layer, latest run plus 48 h back) are bundled; adding a model means adding a file. The output of `ww3_4km` is byte-identical to what the original waves.met.no pipeline wrote, plus `format_version`, `dims` and `geo` in `dataset.json`.

## Tests

`pytest` runs offline on a synthetic dataset: converter, up-to-date check, publish/fetch/prune against a mocked
S3 (moto), the `run` cycle, and (`x3d` is in the same environment) reading the result with `xr.open_dataset("x3d+file://...")`.

## Not done yet

- NorKyst-800 (`norkyst800` prefix, surface layer): a model file plus int16 packing while downloading.
- `Dockerfile` / CronJob for k8s.
- CI.
