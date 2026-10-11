# RunPod CPU snapshot deployment

The native vLLM API/worker CPU snapshot cannot run on the RunPod Secure Cloud container tested on October 11, 2026. CRIU 4.2.1 failed its normal check, unprivileged check and checkpoint of an owned `sleep` process, all with exit code 1. The error was `CRIU needs to have the CAP_SYS_ADMIN or the CAP_CHECKPOINT_RESTORE capability`.

This is a provider container-permissions result. It prevents preparation before model hydration or GPU compilation begins. It does not measure hydration performance or inference latency. The [sanitized receipt](receipts/runpod-cpu-tree-preflight.json) contains the complete diagnostic output, configuration, tool SHA-256, allocation observations and cleanup result.

## Allocation and environment

The preferred [RTX 5090 configuration](../configs/runpod/cpu-tree-preflight-5090.json) was refused for unavailable capacity; the catalog also reported `NONE`. The [RTX 4090 preflight configuration](../configs/runpod/cpu-tree-preflight-4090.json) allocated one Secure Cloud pod in EU-RO-1. GPU-specific serving artifacts were not used for this permissions test.

| Field | Observation |
| --- | --- |
| GPU | RTX 4090, 24,564 MiB |
| CPU | AMD Ryzen 9 7950X |
| Allocated host resources | 16 vCPUs, 61 GB RAM |
| Driver / provider CUDA version | 580.159.04 / 13.0 |
| Kernel | 6.8.0-124-generic |
| Container base | Ubuntu 24.04.5 LTS, pinned Switchyard runtime image |
| CRIU | 4.2.1, separately downloaded diagnostic tool |
| Effective / bounding capabilities | `00000000a80425fb` / `00000000a80425fb` |
| Seccomp | Filter mode, one filter |
| `/proc/sys` | Read-only |
| Allocated compute quote | $0.89/hour |

Both capability sets exclude `CAP_CHECKPOINT_RESTORE`, `CAP_SYS_ADMIN` and `CAP_SYS_PTRACE`. CRIU reports missing effective capabilities 40 and 21 before checkpoint work begins. The corresponding bits are also absent from the bounding set, so the binary's suggested `setcap` command cannot grant them inside this container. Seccomp and the read-only proc mount are additional observed constraints; this test did not establish that either caused the first failure.

The probe's cgroup-v2 CPU and memory files were absent. Those limits were not measured, and the receipt makes no CPU throttling or memory-pressure claim. The provider's CUDA version is its placement metadata; no CUDA workload ran.

## Test ownership and timing

The controller ran in the existing tmux session 0. It published a temporary CRIU diagnostic bundle to R2, passed only a short-lived signed GET URL to its uniquely named pod, streamed sanitized logs and permanently terminated that pod after receiving the completion marker. Long-lived credentials remained on the controller. Inventory confirmed the pod's absence, and the temporary R2 object was deleted.

| Controller milestone | Seconds after controller start |
| --- | ---: |
| Allocation requested | 2.650 |
| Allocation returned | 3.797 |
| Probe output completed | 19.177 |
| Termination requested | 19.924 |
| Inventory confirmed absence | 20.385 |

Allocation return and log delivery are control-plane observations, not precise container-start timestamps. Estimated compute rent was $0.00504, excluding container-disk charges. This estimate includes controller setup before allocation and is not a reconciled invoice. Full serving bake, engine restore and launch-to-first-token remain unmeasured on RunPod.

## Deployment requirements

A matched bake must preserve the selected runtime, CPU features, kernel, mapped base files, CUDA stack, GPU profile and serving arguments. Matching these on one allocated pod addresses snapshot compatibility; it does not change the container's checkpoint permissions.

The next complete CRIU rollout requires a host or container service that grants the required capabilities and supports the native process tree's kernel resources. A VM or bare-metal host with control over the container runtime can be evaluated for that purpose. Passing a basic CRIU permissions test would still be only the first step; the native API/worker bake and full serving restore must then be tested there.

RunPod's documented [pod creation schema](https://docs.runpod.io/api-reference-v2/pods/create-a-pod) exposes container commands and placement constraints, but no capability or privileged-container option. Its `args`, `entrypoint` and `cmd` fields set the container command, so putting Docker daemon flags in those fields does not request elevated permissions. No alternative RunPod tier or provider-specific exception was tested.

A userspace checkpoint backend would require a separate adapter implementation and measurements of the full native tree. Ordinary startup and userspace checkpointing were not substituted for this CRIU rollout. The simulator and scheduler contain no RunPod or CRIU workaround.

## Ordinary startup configuration

For hosts without checkpoint permissions, use the [no-CRIU bake and cold-start methods](cpu-process-snapshots.md#startup-without-criu). `prepare_serving_artifacts.py --without-criu` omits the CPU snapshot, while `run_prepared_cold.py --without-criu` can override an existing snapshot fixture. Both retain baked GPU caches, prepared tensors and fixed memory plans. The ordinary-import path has passed a fresh local serving rollout using the small Ubuntu image and normal container permissions. Its full RunPod serving rollout remains untested; GPU caches and memory profiles still need to match the allocated hardware.
