#!/bin/sh
# Build a static agent for CUDA containers without relying on the host's glibc.
set -eu
project=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
mkdir -p "$project/target/musl" "$project/target/cargo-musl" "$project/target/artifacts"
docker run --rm --name switchyard-agent-build \
  -v "$project:/source:ro" \
  -v "$project/target/musl:/build" \
  -v "$project/target/cargo-musl:/usr/local/cargo/registry" \
  -e CARGO_TARGET_DIR=/build -w /source \
  rust:1.98.1-alpine@sha256:7cc1c22d77d9432f7fe012a70e6d3e555af54c2a6832700ed7d553f1769ae89f \
  sh -c 'apk add --no-cache build-base cmake perl && cargo build --locked --release -p switchyard-agent --target x86_64-unknown-linux-musl'
cp "$project/target/musl/x86_64-unknown-linux-musl/release/switchyard-agent" \
  "$project/target/artifacts/switchyard-agent-x86_64-linux-musl"
sha256sum "$project/target/artifacts/switchyard-agent-x86_64-linux-musl"
