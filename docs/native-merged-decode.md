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

`cargo test -p switchyard-engine --all-targets` includes seven mixed decode fixtures alongside the existing generation, model, transport and CLI tests. `cargo clippy -p switchyard-engine --all-targets -- -D warnings` passes. Tiny deterministic F32 models exercise prompt lengths 8/32/96, GDN value/key-head ratios 1/2/3, a shared prefill cohort, reordered state restoration, scatter/resume and removal of the longest row. They compare each decode with an independent session at a maximum absolute-logit tolerance of `0.00002`.

The engine fixtures check output ordering, EOS, cancellation, different output budgets, observer row IDs across bounded cohorts, fresh calls and context rejection. The offline teacher fixture checks identical next-token choices and expected-token log-probability error at most `0.0001`. Ownership fixtures reject foreign model state, stale or unknown IDs, duplicate selection, position drift and late prefill. These are CPU synthetic results; the released-model CUDA comparison is recorded separately below.

## Measured CUDA numerical gate

The [RTX 5090 receipt](receipts/native-merged-decode-rtx5090.json) records dense F32 Qwen3.5 0.8B, 2B and 4B comparisons against independent native sessions and the pinned published Hugging Face implementation. Each model consumes the same six literal histories with prompt lengths 8/32/96 repeated and continuation lengths 16/12/4/10/8/2. The longest rows finish first, exercising CUDA row selection and KV cropping while shorter rows continue. All 52 teacher-forced next-token decisions per model match both references, giving 156/156 matches per reference across the three models.

| Model | Maximum log-probability error vs independent | Maximum log-probability error vs published |
| --- | ---: | ---: |
| 0.8B | 0.0000191922 | 0.0000178740 |
| 2B | 0.0000162125 | 0.0000161429 |
| 4B | 0.0000643729 | 0.0000355862 |

Every comparison passes the unchanged `0.0001` maximum expected-token log-probability error and exact argmax contract. The oracle uses the accepted PyTorch convolution and GDN fallbacks, eager attention, TF32 disabled and the validated rotary reconstruction. Binary, model revision, config, literal trace, model shard and reference hashes are preserved. All processes ran serially through existing tmux session 0, pane `%3`; no production inference readiness request was added.

The first native GPU attempt rejected strided KV input to CUDA `index_select`. Compaction now makes K/V contiguous before selecting surviving rows, matching the explicit layout handling for GDN state. The original gate then passed. An oracle cache write failure was resolved by directing writable caches into a private output directory while preserving the shared cache read-only; its mathematical script was unchanged. Owned reference containers had a 32 GiB memory cap and no swap beyond that cap. Sampled cgroup high-water memory was 1.84 GB for 0.8B and 3.27 GB for 4B; the 2B container high-water value was not collected.

These tests establish the recorded F32 numerical behavior. Full generated-sequence equality and warm throughput are separate acceptance steps; neither is claimed by the teacher-forcing receipt.

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

`switchyard-engine-mixed-bench` performs that comparison offline with one loaded F32 model. It requires a passing gate receipt bound to the identical request histories and output budgets. Before timing it warms each mode and requires complete generated sequences to match; every measured repetition repeats that check. It alternates measurement order and records raw samples, medians and token timestamps from one outer clock, including each row's wait for earlier work. Driver VRAM and host RSS must be recorded separately. Its times exclude loading and HTTP, so they cannot replace an endpoint or cold-start receipt.

```sh
target/release/switchyard-engine-mixed-bench \
  --config f32-config.json --requests requests.json \
  --gate gate/gate.json --samples 5 --output offline-timing.json
```
