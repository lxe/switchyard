"""Require exact F32 bake output against an identical local artifact execution."""

import re

from control import write


def required(config):
    return (
        config["engine"]["dtype"] == "f32"
        and config["engine"].get("cache", {}).get("max_bytes", 0) > 0
    )


def validate_reference(reference, config, prepared):
    if (
        reference["format"] != "switchyard-native-generation-reference-v1"
        or reference["configuration"] != config["engine"]
        or reference["expected_cloud_build_commit"] != config["engine_source_commit"]
        or reference["model_artifact_id"] != prepared["model"]["id"]
        or reference["model_revision"] != config["model_revision"]
        or reference["weights_verified_against_artifact_manifest"] is not True
        or not re.fullmatch(r"[a-f0-9]{64}", reference["local_engine_sha256"])
        or len(reference["generated"]) != len(reference["requests"])
        or not reference["requests"]
    ):
        raise ValueError(
            "local generation reference differs from fixed model/build inputs"
        )
    for request, row in zip(reference["requests"], reference["generated"]):
        ids = row.get("token_ids")
        if (
            not isinstance(ids, list)
            or not ids
            or any(type(token) is not int or not 0 <= token < 2**32 for token in ids)
            or row.get("prompt_tokens") != len(request["prompt"])
            or (request.get("ignore_eos", False) and len(ids) != request["max_tokens"])
            or row.get("finish_reason") not in ["length", "stop"]
        ):
            raise ValueError(
                "local reference contains incomplete or invalid generation rows"
            )


def compare(reference, receipt, generated):
    requests = reference["requests"]
    if (
        receipt["config"] != reference["configuration"]
        or receipt["requests"] != requests
    ):
        raise ValueError(
            "cloud bake configuration or requests differ from local reference"
        )
    expected = reference["generated"]
    differences = []
    for index in range(max(len(expected), len(generated))):
        want = expected[index] if index < len(expected) else None
        got = generated[index] if index < len(generated) else None
        request = requests[index] if index < len(requests) else None
        if want is None or got is None:
            differences.append({"row": index, "expected": want, "actual": got})
            continue
        ids = got.get("token_ids")
        if (
            ids != want["token_ids"]
            or got.get("prompt_tokens") != len(request["prompt"])
            or got.get("finish_reason") != want["finish_reason"]
            or any(type(token) is not int or not 0 <= token < 2**32 for token in ids)
            or (request.get("ignore_eos", False) and len(ids) != request["max_tokens"])
        ):
            differences.append({"row": index, "expected": want, "actual": got})
    return {
        "format": "switchyard-native-cross-gpu-generation-parity-v1",
        "passed": not differences,
        "rows": len(expected),
        "exact_rows": len(expected)
        - sum(d["row"] < len(expected) for d in differences),
        "local_engine_sha256": reference["local_engine_sha256"],
        "cloud_engine_sha256": receipt["engine_sha256"],
        "model_artifact_id": reference["model_artifact_id"],
        "configuration": receipt["config"],
        "requests": requests,
        "expected_token_ids": [row["token_ids"] for row in expected],
        "actual_token_ids": [row.get("token_ids") for row in generated],
        "differences": differences,
        "scope": "complete token IDs from identical offline F32 requests; separate from teacher-forced mathematical gates and serving readiness",
    }


def verify(reference, receipt, generated, output):
    result = compare(reference, receipt, generated)
    write(output, result)
    if not result["passed"]:
        raise ValueError(
            "native F32 generation differs from local reference; retained parity receipt requires diagnosis"
        )
    return result
