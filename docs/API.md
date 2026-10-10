# Fleet simulator HTTP API

The HTTP server simulates GPU provisioning, model loading, host probes, and inference delivery. Its OpenAI-compatible endpoints return synthetic text; startup time, queueing, token delivery, and usage follow the configured fleet simulation. Every request changes one persistent event engine. Polling a resource reads its current state without rerunning earlier work.

The stable client endpoint is `POST /v1/chat/completions`. A separate routing runtime chooses a compatible hot instance or provisions capacity for the requested model. The provider API exposes instances and their inference endpoints so an external scheduler can make those decisions itself. Each instance also has a pinned OpenAI endpoint whose queue represents requests waiting for that instance's inference engine. The pinned queue does not choose another GPU.

## Server and virtual time

Start a server with the supplied profiles and an automatically advancing clock:

```sh
cargo run -- serve --bind 127.0.0.1:8080 --clock-rate 1 --seed 7
```

`--clock-rate` is simulated seconds per real second. A rate of `1` tracks elapsed real time; `100` advances the fleet one hundred times faster. A rate of `0` enables manual advancement for deterministic experiments:

```sh
cargo run -- serve --bind 127.0.0.1:8080 --clock-rate 0 --seed 7
curl -s http://127.0.0.1:8080/control/time
curl -s http://127.0.0.1:8080/control/advance \
  -H 'Content-Type: application/json' -d '{"seconds":10}'
```

The controls return `{"time_s":10.0}`. An advance is relative to the current clock, accepts zero, and must be finite and between zero and 3,600 seconds. Longer experiments can make repeated calls. Manual mode requires a separate client to advance time while a JSON chat response or SSE stream waits for inference. A waiting chat handler does not hold the fleet lock.

Use `--profiles profiles.json` to supply another hardware catalog and `--environment environment.json` to supply a `CloudConfig`. Environment fields include instance quotas, pinned and routed queue limits, launch/build probabilities, warm instances, faults, host conditions, scale factors, and routing policy. Missing environment fields use their defaults. The default runtime policy is `planned_hybrid`; the default environment has no warm instances or launch failures.

Times in receipts are virtual seconds from server initialization. Throughput computed from those times is simulated throughput, independent of HTTP client execution speed. Identical profiles, environment, seed, and ordered mutations produce an identical manual replay. Separate random streams keep host probes and rejected admission attempts from changing sampled inference lengths.

## Stable model endpoint

`GET /v1/models` returns an OpenAI model list with one entry per model in the catalog. `POST /v1/chat/completions` accepts standard chat messages and routes the request through the runtime:

```sh
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"qwen3.5-0.8b",
    "messages":[{"role":"user","content":"Summarize this paragraph."}],
    "max_completion_tokens":256,
    "simulator":{"input_tokens":120,"planned_output":180}
  }'
```

The model name must match the catalog. The client submits model and workload inputs; it does not need a profile or instance ID. The runtime observes model-loading progress, delivered requests, and completed host probes when choosing provider actions. It does not receive sampled future output lengths or actual future readiness times.

Supported chat fields are `model`, `messages`, `max_completion_tokens` or `max_tokens`, `stream`, `stream_options.include_usage`, and `n` with a value of `1`. `max_completion_tokens` takes precedence when both limits are present; the default limit is `512`, and limits range from `1` to `65536`. Messages may contain string content, content-part arrays, or null content. Image content requires a configured vision offering; submitting images to an instance configured only for text returns `422`.

The simulator accepts ordinary sampling/metadata fields such as `temperature`, `top_p`, `seed`, `stop`, penalties, log-probability options, `response_format`, `metadata`, `store`, `reasoning_effort`, and `service_tier`. These fields do not alter the synthetic text or its sampled length. Tool execution and multiple completions are unsupported. The simulator evaluates capacity and timing rather than answer quality.

The optional `simulator` object supplies:

| Field | Meaning | Default |
| --- | --- | --- |
| `input_tokens` | Tokenized input length supplied by the client | Ceiling of serialized message JSON UTF-8 bytes divided by four, at least one |
| `planned_output` | Estimated generated length used for planning | Requested maximum output |
| `upload_bytes` | Request payload bytes used for simulated upload time | `0` |
| `cached_fraction` | Fraction of input eligible for cached prefill | `0` |
| `image_megapixels` | Image work used by vision profiles | `0` |

The byte estimate is a reproducible placeholder rather than a model tokenizer. For benchmark comparisons, provide measured token counts and upload/image sizes. The extension rejects unknown fields, including `actual_output`.

## JSON completions and streaming

A JSON chat response waits for delivered completion. It has `object: "chat.completion"`, an assistant message, `finish_reason`, and prompt/completion/total usage. Each simulated output token contributes the string `" tok"`. `finish_reason` is `length` when the sampled output reaches the requested bound and `stop` otherwise. `created` is the virtual submission second rather than a wall-clock Unix timestamp.

With `"stream":true`, the server returns `text/event-stream`. No data event is emitted until the receipt reports a delivered first token. The first event supplies the assistant role; subsequent events supply synthetic content in chunks of at most 128 tokens. A finish event follows delivered completion, followed by `[DONE]`. Set `"stream_options":{"include_usage":true}` to receive the standard final usage chunk with an empty `choices` array before `[DONE]`.

HTTP response headers can arrive before the first stream event. Measure inference TTFT from the first content delivery or the virtual receipt, rather than header arrival. Automatic clock advancement runs every 50 ms, and waiting handlers poll every 10 ms; wall-clock stream timings therefore include HTTP runtime overhead. A disconnected client does not automatically cancel admitted physical work.

Each chat response ID has the form `chatcmpl-sim-42`. The numeric suffix identifies `GET /v1/requests/42`, which supplies virtual timestamps and delivered token counts for measurements. A failed or cancelled JSON request returns HTTP `503` with a structured error. A stream that has already begun emits an error event and then `[DONE]`.

When the server receives Ctrl-C, pending JSON requests return a shutdown error and pending streams emit a shutdown event followed by `[DONE]`. This also applies in manual mode, where no clock driver may remain to complete the work.

## Provider lifecycle routes

The provider routes are available independently of the stable client endpoint. An external scheduler can provision, observe, probe, submit to, and release its own instances.

The stable client endpoint activates a bundled controller for the same fleet. Run external-controller experiments through the provider and pinned serving routes in a fresh session so that one controller owns provisioning and release decisions. The library also supports replacing the bundled controller with `CloudSimulator::set_router(RoutingRuntime::with_scheduler(controller))`.

| Method and path | Result |
| --- | --- |
| `GET /v1/offerings` | `{"offerings":[...]}` with advertised profiles and their provenance |
| `GET /v1/instances` | `{"instances":[...]}` containing observable instance state |
| `POST /v1/instances` | Provision `{"profile_id":"..."}`; return `201`, an ID, endpoint base path, and `Location` |
| `GET /v1/instances/{id}` | Instance state, loading progress, completed probes, and pinned endpoint base path |
| `DELETE /v1/instances/{id}` | Release an idle/loading instance; return `204` |
| `POST /v1/instances/{id}/probes` | Start an asynchronous host probe; return `202` |
| `POST /v1/instances/{id}/inference` | Admit a generic request to a ready instance; return `202` and a request ID |
| `GET /v1/requests/{id}` | Request state, observed timestamps, delivered tokens, and completed usage |

A provisioning receipt includes a relative `openai_base_path`, for example `/v1/instances/0/openai/v1`. Append `/models` or `/chat/completions` to use that instance. An OpenAI SDK can use the full server URL plus this path as its `base_url`. The pinned chat endpoint accepts a bounded queue during loading; the generic `/inference` route admits immediately and returns `409` while the instance is not ready or `429` when admission capacity is full.

Generic inference input is explicit:

```json
{
  "model": "qwen3.5-0.8b",
  "input_tokens": 120,
  "planned_output": 180,
  "max_output": 256,
  "upload_bytes": 0,
  "workload": "text",
  "image_megapixels": 0,
  "adapter": null,
  "class": "standard",
  "cached_fraction": 0
}
```

`model`, `input_tokens`, `planned_output`, and `max_output` are required. Other fields default to the values shown. `max_output` reserves memory even when `planned_output` is small. Both maximum concurrent requests and total input/output token reservations limit admission. `class` is a generic service-class name; the provider API contains no evaluation or judge-specific request type.

Instances expose `starting`, `ready`, or `released` status. The actual readiness timestamp is absent until the instance becomes ready. Host probe results become available only after the probe duration has elapsed. Releasing a loading instance cancels its pending pinned requests and prevents pending probes from publishing a result. Releasing an instance with active inference returns `409`.

Request status is `queued`, `active`, `completed`, or `failed`. `submitted_at_s` records receipt of the request, so first-token latency is `first_token_at_s - submitted_at_s`, including queue and loading delays. `generated_tokens` counts tokens delivered to the client after egress delay; it does not expose future GPU work. Completed `usage` contains prompt and completion token counts. Polling before completion does not disclose sampled final output length.

## Errors and resource bounds

Errors use an OpenAI-compatible envelope:

```json
{"error":{"code":"capacity_exceeded","message":"capacity: instance admission limit reached","type":"capacity_exceeded","param":null}}
```

`400` reports malformed JSON or path IDs; `404` reports unknown resources; `405` reports an unsupported method; `409` reports readiness, release, or probe conflicts; `413` reports a body over 64 KiB; `415` reports a missing JSON content type; `422` reports invalid inputs; and `429` reports a full queue, instance quota, or admission capacity. A client can retry `429` after advancing time or observing available capacity.

The default server listens on loopback and does not require authentication. Binding another address changes who can control the simulated fleet and its clock. Configuration and observed state are exposed through the API; sampled physical host state and future inference output are kept inside the engine.

## Rust HTTP client example

Run the server in manual mode, then run a 100-request client burst through the stable endpoint:

```sh
cargo run --example http_scheduler -- frontdoor http://127.0.0.1:8080
```

To exercise an external scheduler through the provider API:

```sh
cargo run --example http_scheduler -- direct http://127.0.0.1:8080
```

Both modes accept an optional model/profile ID and concurrency limit after the URL. The example advances the manual clock in a separate task and prints delivered output throughput, first-token latency, and usage receipts. Its `CloudClient` contains HTTP transport; its policy functions decide when to submit or retry. The direct policy chooses an advertised text offering, provisions one instance, waits for readiness, performs a host probe, submits bounded concurrent requests, and releases that instance after completion. It is an example policy rather than an optimized fleet controller.
