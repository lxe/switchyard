# Paged CUDA attention

This crate provides the native engine's CUDA decode operation for compact,
immutable KV pages. Each sequence supplies an ordered page list with its true
length. A kernel reads those allocations through a device pointer table and
writes one output token per query head. It does not concatenate KV history or
pad short sequences to match longer rows.

The operation supports F32 and BF16 queries and pages, grouped-query attention,
and head dimensions up to 256. Scores, maxima, the softmax denominator, and the
value accumulator use F32. For BF16, unnormalized exponentials are rounded to
BF16 before the value product; the denominator uses their original F32 values.
The kernel uses a temporary F32 score array of
`batch_size * query_heads * longest_sequence_length` elements. It does not
allocate another tensor containing the attention history.

CPU builds have no CUDA compiler dependency and return `None` from the public
operation. CUDA builds compile `src/paged_decode.cu` to embedded PTX with `nvcc`.
`CUDA_ROOT`, `NVCC`, and `CUDA_COMPUTE_CAP` select the compiler and target. The
default target is compute_80. The driver can still compile embedded PTX when
loading the module; the platform bake must exercise this path to populate the
existing CUDA driver cache. There is no runtime NVRTC compilation.

## CUDA launch ownership

The engine itself keeps its workspace prohibition on unsafe Rust. This crate
uses `unsafe_code = "deny"` and permits unsafe operations only in the CUDA launch
module. Its safe public function validates shape, dtype, device, contiguous
layout, non-final page fullness, total lengths, and kernel indexing bounds.

During launch, read locks retain each page's storage. Cudarc device-pointer
guards order reads after earlier writes, then record a read-completion event
when the guards are dropped after launch. Page allocation release or reuse is
ordered after that event. Pointer tables, scratch space, and output use Candle's
stream; writable launch arguments establish write ownership. No host device
synchronization is inserted into the serving operation.

## Focused validation

The GPU test requires an available GPU slot:

```sh
cargo test -p switchyard-engine-kernels --features cuda --test paged_decode -- --ignored --nocapture
```

It compares paged outputs with dense CUDA attention for F32/BF16, head dimensions
128/256, mixed lengths, full pages and partial tails. Tests that run by default
verify the CPU fallback contract. Full model accuracy and throughput gates remain
in `switchyard-engine`.

## Performance limits

A warm RTX 5090 microbenchmark measured approximately 18 microseconds per paged
operation at context 128, compared with 53 microseconds for a one-row dense
reference and 85 microseconds for an eight-row dense reference. At context 1024,
eight-row paged attention took 119 microseconds versus 357 microseconds dense.
One-row context 1024 took 109 microseconds versus 80 microseconds dense; this
kernel does not improve that case. These are single-sample averages of 100
serialized release-mode invocations, including host table preparation and GPU
completion. Page append and model projection are outside this measurement.

The kernel uses one block per sequence/query head. Small batches with long
histories therefore have limited GPU occupancy, and pointer-table/event work
grows with page count. A future long-context optimization needs measured
partitioned attention or reusable pointer-table ownership. The current
implementation makes no full-model throughput claim from this microbenchmark.
