# Native merged decode

The dense Qwen adapter decodes requests with different prompt lengths in one tensor batch after completing their equal-length prefill cohorts. `MixedModelSession` identifies rows by request ID and accepts one input token and one absolute position per active row. The existing engine uses this adapter when an admitted request group contains different prompt lengths. Single requests, equal-length groups and factories without this adapter retain their original path.

## Sequence state

`SequenceState` contains one row's attention KV tensors, GDN convolution history, recurrent state and next absolute position. Its fields are private, and its immutable model owner is checked before restoration. Model clones share that owner and weights; an independently constructed model cannot restore the state even when its shapes match.

Prefill exports individual rows from its cohort. Before the first merged decode, restoration gathers recurrent and convolution tensors along the batch dimension. It left-pads each KV history to the longest active position, aligning the end of every history. For positions 8 and 32, the first row has 24 masked leading entries and eight valid entries; the second has 32 valid entries. The next token appends at the end of both rows. RoPE gathers separate cosine and sine tables at absolute positions 8 and 32 rather than using the shared storage width as a position.

Each decode checks the supplied positions against the owned state before modifying tensors. Selection rejects duplicate and invalid row indices. Restoration assembles all replacement tensors before updating the model. State exported after merged decode strips left padding so each row can resume independently.

## Request ownership and release

`QwenMixedSession` admits distinct request IDs up to its configured row capacity. Released IDs cannot re-enter that session. All remaining prefilled rows join the first decode together, and later decode calls preserve active order. New prefill is rejected once decode begins. These rules keep the current admission boundary while allowing varied prompt lengths to share decode work.

The engine transfers sampled token IDs to the host before releasing EOS, cancelled or output-budget-complete rows. That transfer completes the CUDA step that references the state. Release gathers surviving state rows and crops leading KV storage when the longest row leaves. Releasing every row clears state. Tensor ownership is dropped at these boundaries; the backend allocator may retain freed allocations for reuse, so logical reclamation does not imply that the driver reports an immediate drop in reserved VRAM.

The admitted row count and context limit bound state sizes. Left padding costs storage proportional to active rows times the longest active context, and joining or compacting state can temporarily retain old and replacement tensors. This implementation has no paged KV pool or continuous admission.

## CPU verification

`cargo test -p switchyard-engine --all-targets` passes fourteen tests, including seven mixed decode fixtures. `cargo clippy -p switchyard-engine --all-targets -- -D warnings` passes. Tiny deterministic F32 models exercise prompt lengths 8/32/96, GDN value/key-head ratios 1/2/3, a shared prefill cohort, reordered state restoration, scatter/resume and removal of the longest row. They compare each decode with an independent session at a maximum absolute-logit tolerance of `0.00002`.

The engine fixtures check output ordering, EOS, cancellation, different output budgets, observer row IDs across bounded cohorts, fresh calls and context rejection. The offline teacher fixture checks identical next-token choices and expected-token log-probability error at most `0.0001`. Ownership fixtures reject foreign model state, stale or unknown IDs, duplicate selection, position drift and late prefill. These are CPU synthetic results; CUDA model parity and throughput have not yet been measured.

## CUDA reference handoff

`switchyard-engine-merged-probe` is an offline binary that runs teacher forcing through the same mixed session as generation. It accepts the existing trace format, a JSON array of `prompt` and `continuation` token arrays. `--independent-output` also runs the existing independent-session probe using the same loaded model. Neither diagnostic is called by production readiness.

```sh
CUDA_COMPUTE_CAP=120 CUDA_ROOT=/opt/cuda CARGO_BUILD_JOBS=4 \
  cargo build --release -p switchyard-engine --features cuda \
  --bin switchyard-engine-merged-probe
target/release/switchyard-engine-merged-probe \
  --config f32-config.json --input traces.json \
  --output teacher-merged.json --independent-output teacher-independent.json
```

Run the pinned published Transformers oracle separately on the identical trace file. Compare merged results both with that oracle and with independent F32 sessions, requiring identical argmax IDs and maximum expected-token log-probability error of `0.0001`. The handoff helper prepares repeated and unique 8/32/96-token prompts with differing continuation lengths and verifies every row and step. GPU execution must use the coordinating chat's serialized handoff before any throughput claim.

After numerical acceptance, compare single sessions, the existing equal-length-cohort baseline and merged decode on the same request sequences, device, precision and transport. Record total output throughput, each request's first-token time, decode intervals, active row counts and memory. CPU projection measurements and this implementation alone cannot establish a serving speedup.
