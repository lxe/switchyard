//! Opt-in acceptance through the actual live router client, against an owned native endpoint.
use std::{path::PathBuf, sync::Arc, time::Instant};
use switchyard_live::{
    planner::{Message, Planner},
    serving::{Call, Endpoint},
};
use tokio::sync::mpsc;

#[tokio::test]
#[ignore = "requires an explicitly supplied running native endpoint and pinned tokenizer files"]
async fn native_endpoint_accepts_live_tokenization_and_single_burst_streams() {
    let url = std::env::var("SWITCHYARD_NATIVE_ENDPOINT").expect("native endpoint URL");
    let root = PathBuf::from(
        std::env::var("SWITCHYARD_NATIVE_MODEL_ROOT").expect("pinned tokenizer root"),
    );
    let model = std::env::var("SWITCHYARD_NATIVE_MODEL").expect("served model name");
    let planner = Arc::new(
        Planner::from_files(
            &root.join("tokenizer.json"),
            &root.join("chat_template.jinja"),
        )
        .unwrap(),
    );
    let endpoint = Endpoint::new(url, "local-test", 60).unwrap();
    let messages = vec![Message {
        role: "user".into(),
        content: "Name the capital of France.".into(),
    }];
    let input = planner.input_tokens(&messages).unwrap();
    endpoint
        .verify_tokenization(&model, &messages, input)
        .await
        .unwrap();
    let epoch = Instant::now();
    let (events, mut rx) = mpsc::channel(256);
    let drainer = tokio::spawn(async move { while rx.recv().await.is_some() {} });
    let mut receipts = vec![];
    for (count, mixed) in [(1, false), (8, false), (8, true)] {
        let mut calls = Vec::new();
        for i in 0..count {
            let mut call = Call {
                id: receipts.len() + i,
                model: model.clone(),
                messages: if mixed {
                    vec![Message {role:"user".into(), content: [
                        "Name the capital of France.",
                        "France is a country in western Europe. Name its capital city.",
                        "A geography exercise asks about France, a country in western Europe. The answer should name the capital city of that country. Give a short answer with the city name."
                    ][i % 3].into()}]
                } else {
                    messages.clone()
                },
                max_output: 32,
                input_tokens: 0,
                planner: planner.clone(),
                epoch,
            };
            call.input_tokens = planner.input_tokens(&call.messages).unwrap();
            endpoint
                .verify_tokenization(&model, &call.messages, call.input_tokens)
                .await
                .unwrap();
            calls.push(call);
        }
        // Finish CPU token-count checks before releasing the concurrent burst.
        let mut jobs = tokio::task::JoinSet::new();
        for call in calls {
            let endpoint = endpoint.clone();
            let events = events.clone();
            jobs.spawn(async move { endpoint.complete(call, events).await });
        }
        while let Some(result) = jobs.join_next().await {
            let result = result.unwrap().unwrap();
            assert!(result.input_tokens > 0);
            assert!(result.output_tokens > 0 && result.output_tokens <= 32);
            assert!(result.first_content_s >= result.started_s);
            receipts.push(result);
        }
    }
    drop(events);
    drainer.await.unwrap();
    receipts.sort_by_key(|r| r.request);
    if let Ok(path) = std::env::var("SWITCHYARD_NATIVE_RECEIPT") {
        std::fs::write(path, serde_json::to_vec_pretty(&receipts).unwrap()).unwrap();
    }
    println!(
        "native live-client acceptance: {} streams, input_tokens={input}",
        receipts.len()
    );
}

/// Counts every planned request using the actual local pinned model assets; no GPU required.
#[test]
#[ignore = "requires explicitly supplied checkout with downloaded pinned model tokenizers"]
fn native_gauntlet_fits_all_three_pinned_model_contexts() {
    let root = PathBuf::from(std::env::var("SWITCHYARD_NATIVE_CHECKOUT").expect("checkout path"));
    let models: serde_json::Value =
        serde_json::from_str(include_str!("../../../configs/engine/models.json")).unwrap();
    let plan: switchyard_live::workloads::Experiment =
        serde_json::from_str(include_str!("../../../configs/runpod/native-gauntlet.json")).unwrap();
    plan.validate().unwrap();
    let mut counts = vec![];
    for model in models["models"].as_array().unwrap() {
        let config: serde_json::Value = serde_json::from_slice(
            &std::fs::read(root.join(model["config"].as_str().unwrap())).unwrap(),
        )
        .unwrap();
        let assets = root.join(config["model_root"].as_str().unwrap());
        let planner = Planner::from_files(
            &assets.join("tokenizer.json"),
            &assets.join("chat_template.jinja"),
        )
        .unwrap();
        let mut maximum = 0;
        for case in &plan.cases {
            for request in case.requests(model["id"].as_str().unwrap()) {
                let total = planner.input_tokens(&request.messages).unwrap() + request.max_output;
                assert!(
                    total <= config["max_context"].as_u64().unwrap(),
                    "{} / {} needs {total} tokens",
                    model["id"],
                    case.name
                );
                maximum = maximum.max(total);
            }
        }
        counts.push(serde_json::json!({"model":model["id"],"revision":model["source_revision"],"maximum_input_plus_output_tokens":maximum,"calls":1470}));
    }
    if let Ok(path) = std::env::var("SWITCHYARD_NATIVE_PLAN_RECEIPT") {
        std::fs::write(path, serde_json::to_vec_pretty(&counts).unwrap()).unwrap();
    }
    println!("{}", serde_json::to_string(&counts).unwrap());
}
