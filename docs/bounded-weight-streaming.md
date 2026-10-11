# Bounded final-tensor streaming

The weight loader downloads byte chunks and copies their intersections with baked tensor ranges into the model's final GPU parameter allocations. It never assembles a complete CPU tensor, and it creates no temporary GPU tensor. Environment hydration, metadata extraction, weight reception, and GPU consumption run concurrently. Serving requires all final parameter bytes, but reception does not wait for model-wide download completion before delivering chunks.

## Tensor destinations

`metadata.attributes.tensor_destinations_v1` contains a serialized `switchyard-tensor-destinations-v1` object. Its worker partition identifies rank, tensor parallel size, and pipeline parallel size. Each tensor identifies its final name, scalar dtype, shape, byte count, layout, strides, and source-to-destination byte ranges. A range gives the artifact file entry, source file offset, destination offset relative to that parameter, and byte length. For example, a range with source offset 256, destination offset 0, and length 8192 copies those 8192 artifact bytes to the beginning of the named parameter.

The first supported layout is `dense_c`: strides must equal the canonical C layout, and destination ranges must cover the tensor exactly without overlap. Rust validates the artifact mappings against file boundaries. The vLLM adapter checks every final parameter's name, dtype, shape, strides, device, and contiguity before issuing any copy. The native receipt also binds the runtime, vLLM version, model class, and completed bake transformations. The serving adapter currently supports dense, unquantized rank-zero TP=1/PP=1 state. Noncontiguous export or destination layouts are rejected explicitly; they are never flattened silently.

The bake exports the engine's final parameter representation, and `pack_weights` derives source offsets from its safetensors headers. Final serving bypasses HF mapping/fusion and the native post-load transformations already completed by that bake. Artifacts without a final destination index cannot enter the final-state serving path.

## Module responsibilities

- `switchyard-artifacts/tensors.rs` defines tensor destinations and validity rules without engine or provider dependencies.
- `range_stream/transport.rs` receives concurrent HTTP objects or ranges into reservations. It interprets no tensor metadata and enumerates requests lazily.
- `range_stream/buffer.rs` owns RAM memfds, private disk files, aligned I/O, capacity accounting, reader leases, and reclamation.
- `switchyard-weights/chunk_bridge.rs` delivers descriptors and processes individual completed-copy acknowledgements through a private Unix socket.
- `switchyard_chunk_loader.py` resolves final CUDA parameter views, stages bounded chunks, submits copies, and polls CUDA events.
- `reception.rs` coordinates these modules with metadata extraction and the existing agent startup gate. Provisioning and storage credentials remain in their existing adapters.

## Buffer limits and ownership

The default limits are 4 MiB chunks, 64 MiB of RAM backing, 4 GiB of transient disk backing, 16 simultaneous downloads with 64 KiB writer buffers, and 20 MiB of pinned staging shared by four transfer slots. These limits are explicit fields in `VllmLoaderConfig.buffers`. Runtime imports, tensor metadata, HTTP/TLS state, kernel allocations, and persistent cache pages contribute additional process/container memory; the RAM backing limit is not a claim about total container memory.

Each normal chunk follows `reserved → downloading → ready → copying → consumed → reclaimed`. Reservation happens before opening its HTTP request. A chunk becomes ready after exact body completion and storage writes. A whole extent also checks its baked digest; subranges of a larger extent have exact HTTP bounds but no separate digest unless baked as individual extents. The production bake uses extents no larger than the chunk size.

The consumer reads into an aligned pinned slot and closes the backing descriptor. It submits copies from slices of that slot directly into slices of final GPU parameter byte views. A CUDA completion event protects each slot. The consumer sends an acknowledgement only after that event reports completion, then reuses the slot. The server releases the chunk lease on that acknowledgement; its backing is reclaimed after the final reader lease drops. No device-wide synchronization is required per chunk. On cancellation, the consumer settles outstanding events before releasing pinned buffers, and the receiver disposes of its private backlog.

Reception prefers RAM reservations while capacity is available. Overflow uses individually reserved, physically allocated private files with aligned `O_DIRECT` writes and reads. Files are unlinked after consumption; no shared cached artifact is punched, truncated, or removed. Tmpfs and ramfs are rejected as overflow locations. The bounded local launch uses `/opt/switchyard/stream-models` on the owned NVMe bind mount. Once both RAM and disk capacities are exhausted, new requests wait for reclamation. Finite storage cannot preserve network saturation indefinitely.

Linux requires alignment for direct I/O and cautions against concurrent fork with private direct-I/O buffers. The Rust receiver performs its writes in a separate process from vLLM's worker creation and uses aligned buffers. The measured container tests also observe dirty pages rather than treating disk files as proof of bounded memory. See the [Linux `open` documentation](https://man7.org/linux/man-pages/man2/open.2.html) and [PyTorch CUDA stream documentation](https://docs.pytorch.org/docs/stable/notes/cuda).

## Telemetry and verification

`SWITCHYARD_WEIGHT_STREAM` events use monotonic timestamps. Lifecycle events and periodic buffer progress report received bytes, RAM and disk occupancy and peaks, reclaimed bytes, consumer availability, body wait, spool write time, capacity wait, and consumer wait. CUDA submission/completion events identify chunks and final-destination bytes. Byte-only progress is throttled to 100 ms while lifecycle transitions remain visible. Wait counters sum across workers; they are not elapsed wall-clock durations. The CUDA adapter applies the CPU snapshot clock offset when running in a restored time namespace.

The synthetic GPU harness runs the Rust receiver and CUDA consumer together in one container. It tests a 64 MiB tensor with 1 MiB RAM backing, 2 MiB disk backing, 256 KiB chunks, and four transfer slots. Delayed consumer startup, deliberately slow CUDA work, out-of-order arrivals, retained readers, cancellation, and incompatible strides are covered. Correctness is checked in the test harness by reading final GPU slices back in bounded chunks. Production startup performs no extra correctness inference.

The measured synthetic receipt is [bounded-weight-acceptance.json](bounded-weight-acceptance.json). Peak container memory was 358,400,000 bytes, both data budgets were respected, four transfers were in flight, and the loader allocated zero temporary GPU bytes. All normal backing regions were reclaimed. The 20.57 MiB/s reception rate includes intentional startup and GPU delays; it is not a measurement of network link capacity. Event-measured DMA throughput was 22.00 GiB/s. The full native cold-start timing is reported separately after the coordinating launch succeeds.
