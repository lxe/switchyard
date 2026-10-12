# Virtual hardware profiles

This catalog defines three hypothetical serving tiers for `demo-model`. The tiers vary aggregate GPU work capacity, concurrency, KV capacity, client networking, hydration bandwidth, startup time, and rental price. Every value is an assumption for sensitivity testing; the catalog makes no claim about a physical GPU or cloud provider.

| Profile | Single-request work/s | Maximum-batch work/s | Slots | KV tokens | Client link, GB/s | Hydration, GB/s | Assumed USD/hour |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `virtual-baseline` | 80 | 600 | 16 | 65,536 | 0.10 | 0.10 | 0.50 |
| `virtual-fast` | 160 | 3,600 | 64 | 262,144 | 1.25 | 1.00 | 3.00 |
| `virtual-wide` | 240 | 9,600 | 128 | 1,048,576 | 6.25 | 2.00 | 8.00 |

GB/s uses decimal bytes. Work/s includes input processing and output generation, so it is not directly equivalent to delivered output tokens/s. Each tier has the same input-work coefficient. Higher `kv_tokens` allows more input-plus-output reservation; it is a serving budget rather than raw VRAM in bytes.

Each cold host downloads an assumed 4 GB model with no cache hit. This transfer is added to the non-transfer startup baseline: six seconds for baseline, four for fast, and three for wide. Warm hosts bypass the cold transfer. All profiles set `weights_s` to zero because hydration is represented by explicit bytes and bandwidth. Profile names are labels; all behavior comes from the other fields.

Run identical requests on one instance of each tier:

```sh
cargo run --locked --release --example virtual_hardware
```

The example compares a cold single call, a cold 1,000-call burst, a warm 1,000-call burst, and a warm burst with 64 MiB of transport padding per call. The padding isolates upload cost without claiming that payload bytes determine token counts. Each request has 512 input tokens, an estimated 64 output tokens, a 128-token hard bound, and a private actual output length of 96. Full warm slot capacity is enabled consistently for all tiers; the cold controller provisions one instance.

Results under `results/virtual-hardware` include the exact profile catalog, generated scenario and detailed timings for each case, and comparison summaries. To edit the catalog or select another output directory:

```sh
cargo run --locked --release --example virtual_hardware -- \
  configs/virtual-hardware/profiles.json results/my-hardware-comparison
```

Each case isolates one tier to compare its physical behavior. Use all three entries with `run --profiles configs/virtual-hardware/profiles.json --scenario SCENARIO` or `serve --profiles ...` to let a scheduler choose among tiers. A workload must use a matching model identity; the [quickstart burst](../quickstart/burst.json) already uses `demo-model`. Scenario `warm` profile IDs must also match the chosen catalog.

## Hypothetical outcomes

The example completed all 12 cases and 9,003 calls with zero invalid decisions. These deterministic simulation results use the checked-in assumptions and request sizes:

| Profile | Cold single-call TTFT, s | Warm 1,000-call output tok/s |
| --- | ---: | ---: |
| `virtual-baseline` | 46.233 | 508.6 |
| `virtual-fast` | 8.096 | 3,035.9 |
| `virtual-wide` | 5.062 | 8,116.7 |

Cold TTFT includes startup, the assumed 4 GB transfer, input processing, and first-token delivery. Warm throughput divides 96,000 completed output tokens by the complete burst duration. The full generated comparison also includes cold bursts and upload-heavy workloads. Detailed costs include the configured 600-second observation horizon for retained warm hosts; they are not job-only rental costs.

## Network approximation

The network model gives each request an independent link and each instance its own hydration bandwidth. It does not enforce a shared NIC or object-store bandwidth budget across connections or hosts. Raising bandwidth here changes the existing link approximation; it does not simulate a physical switch or a twenty-connection aggregate pipe. See [configuration](../../docs/CONFIGURATION.md) and [cloud observations](../../docs/EMPIRICAL_SIMULATION.md) for fields and additional environmental controls.
