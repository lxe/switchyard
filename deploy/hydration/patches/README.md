# Runtime source patches

Runtime source patches change an installed Python package during the artifact bake. They are optional: a source lock without `source_patches` leaves every installed source file unchanged. Patches are applied before Python bytecode compilation and are recorded in `build-info.json`. The resulting runtime files and source lock participate in the artifact identity.

Add a descriptor to this directory and pin its file hash in `sources.lock.json`:

```json
"source_patches": [
  {"path": "patches/example.json", "sha256": "<descriptor SHA-256>"}
]
```

A descriptor has this structure:

```json
{
  "schema_version": 1,
  "id": "engine-version-change-name",
  "package": {"name": "engine", "version": "1.0"},
  "target": "venv/lib/python3.12/site-packages/engine/module.py",
  "before_sha256": "<original source SHA-256>",
  "after_sha256": "<patched source SHA-256>",
  "replacements": [{"before": "exact original text", "after": "replacement text"}]
}
```

Descriptor and target paths must be normalized relative paths to regular files, without symlink components. Package versions must match the installed package inventory. Each replacement must match exactly once, in sequence; the complete source and result hashes must also match. Identifiers and targets must be distinct within a bake. The builder validates the entire patch set before writing any target. An I/O failure aborts the bake; a failed bake must not be published.

Keep a unified diff next to each descriptor for review. A patch must have semantic checks and a matched serving test before it is enabled in a published runtime. Faster imports alone do not establish a lower time to first token. Rebuild compatible GPU caches and verify inference against the resulting artifact; a source patch can change compiler identities even when it preserves outputs.

This mechanism belongs to runtime preparation. Provider adapters, scheduler policies, and simulator behavior do not apply library patches at startup.
