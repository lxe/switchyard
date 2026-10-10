# Switchyard provider adapters

`switchyard-provider` discovers GPU capacity, allocates generic containers,
manages their lifecycle, and signs object-storage requests. It does not
interpret models, serving engines, bootstrap programs, or scheduling policy.

## Provider interface

`Provider` exposes `offers()`, `create(Launch)`, `get`, `list`, `terminate`, and
`endpoint_url`. Each adapter owns its cloud configuration. `RunPod::with_config`
accepts `ComputeConfig` and a credential; `RunPod::with_key` creates a REST-only
client for inventory, lifecycle reads, and billing. Capacity discovery and
container allocation require the configured constructor.

`Launch` selects an `Offering`, a unique name, and a `ContainerSpec`. The
container specification contains an image, disk allocation, entrypoint,
command arguments, environment variables, and ports. Empty entrypoint or command
vectors retain the image's defaults. `Port { number, http }`
identifies an HTTP proxy port or a TCP port. RunPod translates these generic
fields into its REST v2 request. Another provider can implement the same
interface without understanding what runs inside the container.

RunPod checks inventory before allocation to reject reused names. When
allocation returns a transport failure, a server error, or malformed success
metadata, the adapter checks inventory several times without repeating
`POST /pods`. An unresolved or ambiguous result returns
`Error::UnknownLaunch { name }`. The caller must preserve its ownership
journal, reconcile that exact name, and release recovered pods before retrying.
A missing pod's read returns `None`; termination treats an explicit HTTP 404
as success.

## Capacity and storage configuration

`ComputeConfig` contains RunPod API access, cloud selection, GPU and data-center
filters, minimum CUDA version, disk allocation, and disk pricing assumptions.
`StorageConfig` contains S3 endpoint, region, bucket, credential variable names,
and URL expiration. Both types validate their own fields. Endpoint URLs cannot
contain credentials or query parameters. Local HTTP endpoints remain available
for contract tests.

`Offering` records the catalog's GPU hourly quote and a separate disk estimate.
The disk estimate is `disk_gb * disk_usd_per_gb_month /
billing_hours_per_month`. RunPod rejects a launch whose disk differs from the
configured allocation used for that quote. `Pod::cost` retains the
provider-reported quote; its meaning must be checked against the provider's
billing rules before combining it with disk estimates. Neither field is a
settled charge. `RunPod::billing` exposes billing data for later reconciliation;
callers should retain only records belonging to their experiment.

## Object storage

`Storage` signs S3-compatible requests with AWS SigV4. Cloudflare R2 uses the
signing region `auto`; geographic bucket placement is a separate concept.
JSON uploads and reads use signed headers. Expiring GET URLs are capabilities
and must stay out of logs and persisted reports. Object-key segments are
encoded according to RFC 3986 before signing, including literal `+`, `%`, and
spaces. Keys containing empty, `.` or `..` components are rejected so URL
processing cannot select a different object.

Errors contain an operation and an HTTP status, when available. They do not
include response bodies, credentials, or signed URLs. `ContainerSpec`,
`Launch`, `RunPod`, and `Storage` intentionally omit `Debug`; persisted `Pod`
metadata does not retain environment variables or startup commands.

## Validation

```sh
cargo test -p switchyard-provider
cargo clippy -p switchyard-provider --all-targets -- -D warnings
```

Tests use local HTTP servers to exercise allocation recovery, opaque container
translation, catalog filtering, lifecycle behavior, sanitized errors, and
signed payloads. A fixed signature vector was calculated independently of the
AWS signer to check S3 path encoding and credential scope. Tests do not allocate
GPUs or use live credentials.

See [RunPod live execution](../../docs/RUNPOD.md) for the inference runtime's
fleet configuration, workloads, receipts, and recovery commands.
