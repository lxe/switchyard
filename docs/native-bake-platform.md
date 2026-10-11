# Native bake platform controller

The native bake platform controller allocates one RunPod GPU directly, observes its hardware, and runs fixed native cache-preparation commands. It reads an independent platform configuration and a list of model configurations and prepared artifacts. It does not use inference scheduling to select or operate the bake pod.

## Platform configuration

`configs/runpod/native/bake-platform-sm89-32g.json` permits the 0.8B and 2B models with a 32 GB host-RAM minimum and a 12 GiB GPU-admission floor. `bake-platform-sm89-64g.json` permits all three models with a 64 GB RAM minimum and a 20 GiB GPU-admission floor. Both request an RTX 4090, compute capability 8.9, eight vCPUs, 30 GB disk, and a minimum compatible host CUDA capability of 13.0. Their price ceiling is $1.00 per hour, their owned-pod limit is one, and their pod deadline is 600 seconds.

The provider's minimum host CUDA capability is an allocation requirement. The serving runtime retains its own toolkit, compiler, binary, and source identities. A newer compatible host driver does not change those artifact identities; its cache component records the actual observed driver separately.

`driver_target` is an exact observed driver version, or `null` for discovery. `excluded_drivers` prevents selecting unwanted versions before weights transfer. These fields select a bake host; they do not establish cache validity. A new engine or model configuration still requires new components for every desired driver. The current profiles target 580.173.02, observed on a Romania RTX 4090 during the [driver discovery attempt](receipts/native-bake-driver-discovery-01.json).

Model configurations supply the model revision, native request bounds, and binary provenance. Before allocation, the controller checks that the platform admits every requested model and meets its fixed RAM, CPU, disk, and GPU-admission requirements. Every input must refer to the same shared runtime. After generation, the controller compares the bake receipt's complete native configuration, exact binary hash, and hardware identity before packing a component.

## Offline control and cleanup

The HTTP bake control surface authenticates every request with a temporary token. It exposes status, a bounded diagnostic log, and bundles for known model names. Its only write operation releases the fixed bake program with an empty `POST /start`; callers cannot submit commands. The controller observes and records hardware before releasing that gate, so unsupported or unwanted hardware causes no weights transfer.

Python and zstd are installed only inside the temporary offline bake container. Its token and signed download script stay in memory and are removed from the environment before native subprocesses run. These dependencies are not published into the serving runtime or serving image. Native generation uses the existing Rust bake executable and engine binary.

The bake controller polls the authenticated HTTP surface, retains sanitized provider logs, and downloads each cache bundle with a 32 MiB compressed-byte limit. The diagnostic tail is capped at 64 KiB and redacts URLs and the control token. It downloads the finished components, then terminates the exact journaled pod before local packing and object-store publication. Each component is verified and published independently after cleanup succeeds. A provider GET returning 404 confirms cleanup. Generation failure, control failure, and timeout follow the same cleanup path. Publication failure occurs after GPU ownership has ended. The 600-second deadline bounds the remote-work loop; bounded diagnostic reads and provider cleanup calls run afterward.

## Controller invocation

Run this example after the engine source and prepared artifacts are frozen. Replace the configuration and preparation paths together when using a newer engine freeze; the paths below refer to the recorded binary from source `1c966bb`. Every output directory must be fresh.

```bash
python deploy/native/run-model-bakes.py \
  --platform-config configs/runpod/native/bake-platform-sm89-64g.json \
  --config configs/runpod/native/qwen3.5-0.8b-sm89-final.json \
  --prepared results/native-runpod/prepared-0.8b-sm89-final-01/prepared.json \
  --config configs/runpod/native/qwen3.5-2b-sm89-final.json \
  --prepared results/native-runpod/prepared-2b-sm89-final-01/prepared.json \
  --config configs/runpod/native/qwen3.5-4b-sm89-final.json \
  --prepared results/native-runpod/prepared-4b-sm89-final-01/prepared.json \
  --tool target/release/switchyard \
  --output results/native-runpod/bake-models-sm89-driver-target
```

Artifact assembly happens after baking. `prepare-family.py` verifies the finished components and derives serving compatibility labels with the offline Rust `cache-bindings` command. This operation performs no cloud allocation, inference scheduling, or generation. Existing family components retain their original bytes and identities when a new component is added.

## Verification scope

The CPU HTTP fixtures establish authentication, hardware checks before program execution, exact driver-target rejection, fixed-operation gating, known-bundle access, and retained sanitized stderr on failure. Run them with `python -m unittest discover -s deploy/native/tests -p test_bake_http.py`. Three lifecycle fixtures also verify that confirmed cleanup and receipt copying precede publication, and that generation failure or unconfirmed cleanup prevents publication. Run those with `python -m unittest discover -s deploy/native/tests -p test_model_bakes.py`.

The discovery attempt established driver 580.173.02 and confirmed owned-pod cleanup. It published no additional cache component. The HTTP request headers and output-parent creation have been repaired, and diagnostics now persist before cleanup. A successful cloud bake remains required to establish component coverage. Further production allocations are held until the engine source with prefix caching and paged attention is frozen.
