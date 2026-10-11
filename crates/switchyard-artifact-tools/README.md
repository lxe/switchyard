# Switchyard artifact tools

This crate connects the generic `switchyard-artifacts` library to S3-compatible
storage for local preparation and publication. It does not depend on the simulator,
scheduler, live fleet runtime, or inference engine.

The `switchyard artifacts` commands build, pack, verify, publish, and hydrate filesystem
trees. The runtime builder runs in Docker and exports a tar stream into a local
staging directory before publishing the tree. Publication uses immutable object identities and conditional writes; it
publishes the manifest after every chunk has been accepted. Credential files are
read locally, and download capabilities are never printed or stored in receipts.
