# Switchyard artifact tools

This crate connects the generic `switchyard-artifacts` library to S3-compatible
storage for local preparation and publication. It does not depend on the simulator,
scheduler, live fleet runtime, or inference engine.

The `switchyard artifacts` commands build, pack, verify, publish, and hydrate filesystem
trees. The runtime builder runs in Docker and exports a tar stream into a local
staging directory before publishing the tree. Publication uses immutable object identities and conditional writes; it
publishes the manifest after every chunk has been accepted. Credential files are
read locally, and download capabilities are never printed or stored in receipts.

Library callers use `publish_directory(&storage, directory, namespace, concurrency)`
with an existing `switchyard_provider::storage::Storage`. The function validates
the packed directory's manifest, captures its exact bytes, uploads verified chunks,
and publishes the manifest last under `<namespace>/<artifact-id>`. It returns a
serializable `PublicationReceipt` with the immutable reference and per-object
creation flags. Existing matching objects are verified and reused; different
objects are refused. The CLI invokes this same function and writes its receipt.

## Raw extent preparation and publication

`pack-extents` preserves filesystem bytes in independently downloadable raw objects
and writes a `switchyard-extents-v1` manifest. This generic packer divides files at
the requested size; model-specific tensor and header planning belongs to the weight
preparation layer. Extent layout identity and filesystem content identity are
separate fields, and neither is interchangeable with an archive identity.

```sh
cargo run --features artifacts -- artifacts pack-extents \
  --source model-files --out packed-weights --metadata metadata.json \
  --extent-bytes 8388608
cargo run --features artifacts -- artifacts publish-extents \
  --config fleet.json --env-file credentials.env \
  --directory packed-weights --prefix switchyard/weights \
  --concurrency 20 --receipt publication.json
```

The publisher also accepts extent directories produced by model-specific packers.
Their manifest must be named `manifest.json`. Library callers use
`publish_extent_directory(&storage, directory, namespace, concurrency)` and receive
an `ExtentPublicationReceipt` containing the format, layout and content identities,
immutable prefix, and per-object creation flags. Shared objects are uploaded once,
even when multiple extents refer to different ranges within the object.

Before upload, publication validates the manifest, each local object, and every
declared range digest. These checks run during preparation, outside endpoint
startup. The captured manifest is published after all unique objects have been
accepted. A corrupt range prevents publication; failed or cancelled uploads can
leave reusable objects without publishing an incomplete manifest. Object files
must remain stable throughout the operation.

A [live R2 receipt](../../experiments/extent-publication-r2.json) records a 4 KiB
fixture packed into four extents that share one identical content-addressed object.
Remote reads matched the local object and manifest bytes. A second publication
reused both objects without creating replacements. This validates the storage
boundary and deduplication; it does not establish model startup performance.

Hydration defaults to verified installation. For trusted, pinned publications,
`hydrate --trusted-streaming --concurrency 20 --extraction-concurrency 20`
extracts response bytes as they arrive without content hashing, a tree rescan,
or compressed-cache writes. This path preserves private staging, path and entry
constraints, and declared bounds. The `--cache` directory is only used by verified
hydration.

## Download measurements

`switchyard artifacts probe --config fleet.json --env-file credentials.env
--prefix switchyard/artifacts/<id> --id <id> --concurrency 20 --receipt probe.json`
measures the compressed publication's download rate. It resolves and validates
the pinned manifest, issues up to twenty concurrent GETs, counts each body frame,
and immediately discards it. Body lengths must match their declarations. The probe
does not decompress, hash, extract, cache, or write chunk bytes, and does not retry.
Its memory use includes concurrent HTTP buffers, with no zstd decoder histories.
Only the requested JSON receipt is written.

Library callers use `probe_directory(&storage, prefix, id, ProbeOptions)` and receive
a serializable `ProbeReceipt`. Manifest resolution and signing time are recorded
separately from transfer elapsed time. The receipt records each shard's header
latency, first and last body-frame arrival, and response completion, all against
one monotonic transfer clock. Aggregate bytes per second includes request and
header time; bytes are HTTP response bodies, excluding TLS and protocol framing.
This is one measurement on the caller's host and connection. It does not establish
a provider throughput guarantee or measure decompression performance.

Hydration accepts `--receipt hydrate.json` to record controller time, complete
hydration time, their sum, and progress callback observations. For fresh trusted streaming,
add `--retries 0` to record the callback that first sees all declared wire bytes
and the remaining time until publication. Progress is throttled, so this callback
can occur after the actual last network byte: its timestamp is an upper bound,
and the reported remaining tail is a lower bound. Receipts mark that qualification.
Retry-enabled, verified/cache-assisted, and already-present installations omit
that completion/tail measurement to avoid attributing repeated or cached bytes
to a network completion time. First positive network progress is also an
observation rather than an exact first-byte timestamp. The generic hydration
library and its verification policies are unchanged.

`--attempt-timeout-s` bounds each full GET including headers and body;
`--stall-timeout-s` bounds waits for initial headers or the next body frame. Receipts and transport
errors omit signed capabilities and credentials.
