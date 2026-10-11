# syntax=docker/dockerfile:1
# This image is a build tool. Only the export target leaves the build machine.
FROM --platform=linux/amd64 rust:1.98.0-slim-bookworm@sha256:1469a27c125cb5a3aebfa4f4e4665d935b02fb72cc093b2c974b3d740e43f157 AS rust-builder
WORKDIR /source
COPY Cargo.toml Cargo.lock ./
COPY src/ src/
COPY crates/ crates/
RUN cargo build --locked --release -p switchyard-runtime-build

FROM --platform=linux/amd64 ubuntu:24.04@sha256:f610ab94648195aa356059f5b41d6085c9d4d903c072430cdd1af7bdb646106b AS native-builder
LABEL io.switchyard.component="hydration-builder"
ENV DEBIAN_FRONTEND=noninteractive
# Raw Ubuntu has no CA bundle. BuildKit verifies this pinned package over TLS.
ADD --checksum=sha256:6bac2a01979e210d9eac1d4d56747ec709ea60654744d66705dc3c36e7629e50 https://snapshot.ubuntu.com/ubuntu/20261010T000000Z/pool/main/c/ca-certificates/ca-certificates_20260601~24.04.1_all.deb /tmp/bootstrap-ca.deb
RUN dpkg-deb -x /tmp/bootstrap-ca.deb /tmp/bootstrap-ca \
 && mkdir -p /etc/ssl/certs \
 && cat /tmp/bootstrap-ca/usr/share/ca-certificates/mozilla/*.crt > /etc/ssl/certs/ca-certificates.crt \
 && rm -rf /tmp/bootstrap-ca /tmp/bootstrap-ca.deb
RUN apt-get update --snapshot 20261010T000000Z \
 && apt-get install --snapshot 20261010T000000Z --no-install-recommends -y \
      ca-certificates gcc-13 g++-13 binutils libc6-dev libnuma1 libavdevice60 xz-utils \
 && rm -rf /var/lib/apt/lists/*

FROM native-builder AS builder
WORKDIR /build
COPY deploy/hydration/sglang/sources.lock.json deploy/hydration/sglang/requirements.in deploy/hydration/sglang/requirements.lock deploy/hydration/sglang/smoke.py deploy/hydration/sglang/compiler-wrapper.sh deploy/hydration/sglang/runtime-env.sh ./
COPY deploy/hydration/patches/ /build/patches/
COPY --from=rust-builder /source/target/release/switchyard-runtime-build /build/switchyard-runtime-build
RUN --mount=type=cache,id=switchyard-hydration-downloads,target=/build/downloads,sharing=locked \
    --mount=type=cache,id=switchyard-hydration-uv,target=/build/uv-cache,sharing=locked \
    /build/switchyard-runtime-build

FROM scratch AS export
COPY --from=builder /opt/switchyard/runtime/ /runtime/
