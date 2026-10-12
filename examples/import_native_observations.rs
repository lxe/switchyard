//! Offline adapter from native cloud receipts to the generic simulation dataset.
//! Usage: cargo run --example import_native_observations -- OUT SERIAL_RUN BACKGROUND_RUN
use serde_json::{Value, json};
use std::{collections::BTreeMap, error::Error, fs, path::Path};
use switchyard::{Job, Policy, Profile, Request, Scenario, WarmNode, Workload};
use switchyard_core::stats::percentile;
use switchyard_gauntlet::empirical::{Case, Dataset, Metrics};
use switchyard_sim::observations::{
    CapacityLimit, Interval, Observations, ObservedEnvironment, StartupSample,
};

fn read(p: impl AsRef<Path>) -> Result<Value, Box<dyn Error>> {
    Ok(serde_json::from_slice(&fs::read(p)?)?)
}
fn records(p: impl AsRef<Path>) -> Result<Vec<Value>, Box<dyn Error>> {
    let p = p.as_ref();
    if !p.exists() {
        return Ok(vec![]);
    }
    fs::read_to_string(p)?
        .lines()
        .map(|l| Ok(serde_json::from_str(l)?))
        .collect()
}
fn n(v: &Value, k: &str) -> f64 {
    v[k].as_f64().unwrap_or(0.0)
}
fn count(v: &Value, k: &str) -> u64 {
    v[k].as_u64().expect("receipt integer")
}
fn strings(v: &Value) -> &[Value] {
    v.as_array().map(Vec::as_slice).unwrap_or(&[])
}
fn merge(mut intervals: Vec<Interval>) -> Vec<Interval> {
    intervals.sort_by(|a, b| a.start_s.total_cmp(&b.start_s));
    let mut out: Vec<Interval> = vec![];
    for p in intervals {
        if let Some(last) = out.last_mut()
            && p.start_s <= last.end_s
        {
            last.end_s = last.end_s.max(p.end_s);
        } else {
            out.push(p);
        }
    }
    out
}

struct ImportedModel {
    profiles: Vec<Profile>,
    cases: Vec<Case>,
}

fn import(root: &Path, serial: bool) -> Result<Vec<ImportedModel>, Box<dyn Error>> {
    let specs = read(root.join("workloads.json"))?;
    let mut output = vec![];
    let mut directories: Vec<_> = fs::read_dir(root)?
        .filter_map(Result::ok)
        .filter(|p| p.path().join("configuration.json").exists())
        .collect();
    directories.sort_by_key(|p| p.file_name());
    for dir in directories {
        let path = dir.path();
        let model = dir.file_name().to_string_lossy().into_owned();
        if read(path.join("outcome.json"))?["error"] != Value::Null {
            continue;
        }
        let mut profiles: Vec<Profile> =
            serde_json::from_value(read(path.join("profiles-used.json"))?)?;
        let config = read(path.join("configuration.json"))?;
        let owners = read(path.join("owned-pods.json"))?;
        let owners = strings(&owners);
        let obs = records(path.join("agent-observations.ndjson"))?;
        let rejected = read(path.join("allocation-rejections.json")).unwrap_or(json!([]));
        let mut samples: BTreeMap<usize, StartupSample> = BTreeMap::new();
        for (index, owner) in owners.iter().enumerate() {
            let pod = owner["pod"]["id"].as_str().ok_or("missing pod")?;
            let states: Vec<_> = obs
                .iter()
                .filter(|o| o["pod"] == pod && !o["state"].is_null())
                .collect();
            let Some(ready) = states.iter().find(|o| o["state"]["phase"] == "ready") else {
                continue;
            };
            let state = &ready["state"];
            let milestones: BTreeMap<_, _> = strings(&state["startup_milestones"])
                .iter()
                .map(|m| (m["event"].as_str().unwrap_or(""), n(m, "elapsed_s")))
                .collect();
            let duration = |role: &str| {
                milestones
                    .get(format!("{role}_hydration_finished").as_str())
                    .copied()
                    .unwrap_or(0.0)
                    - milestones
                        .get(format!("{role}_hydration_started").as_str())
                        .copied()
                        .unwrap_or(0.0)
            };
            let hyd = duration("model").max(duration("runtime"));
            let engine = milestones
                .get("engine_health_ready")
                .copied()
                .unwrap_or(0.0)
                - milestones.get("engine_spawned").copied().unwrap_or(0.0);
            let total = n(ready, "finish_s") - n(owner, "allocated_s");
            let mut started = None;
            let mut prefill = vec![];
            for row in records(
                path.join("provider-logs")
                    .join(pod)
                    .join("container-events.ndjson"),
            )? {
                let Ok(e) = serde_json::from_str::<Value>(row["line"].as_str().unwrap_or(""))
                else {
                    continue;
                };
                if e["event"] == "batch_started" || e["event"] == "mixed_batch_started" {
                    started = Some(n(&e, "elapsed_s"));
                }
                if e["event"] == "prefix_prefill"
                    && let Some(t) = started.take()
                {
                    prefill.push(n(&e, "elapsed_s") - t);
                }
            }
            let normal = percentile(prefill.iter().skip(1).copied().collect(), 0.5).unwrap_or(0.0);
            samples.insert(
                index,
                StartupSample {
                    boot_s: (total - hyd - engine).max(0.0),
                    hydration_s: hyd,
                    hydration_bytes: count(state, "bytes_total")
                        + count(state, "artifact_bytes_total"),
                    ready_s: total,
                    first_use_s: (prefill.first().copied().unwrap_or(0.0) - normal).max(0.0),
                },
            );
        }
        for (i, p) in profiles.iter_mut().enumerate() {
            let ss: Vec<_> = owners
                .iter()
                .enumerate()
                .filter(|(_, o)| o["profile"].as_u64() == Some(i as u64))
                .filter_map(|(j, _)| samples.get(&j))
                .collect();
            let all: Vec<_> = samples.values().collect();
            let ss = if ss.is_empty() { all } else { ss };
            let median = |f: fn(&StartupSample) -> f64| {
                percentile(ss.iter().map(|s| f(s)).collect(), 0.5).unwrap_or(0.0)
            };
            p.startup.allocation_s = 0.0;
            p.startup.container_s = median(|s| s.boot_s);
            p.startup.weights_s = median(|s| s.hydration_s);
            p.startup.engine_s =
                (median(|s| s.ready_s) - p.startup.container_s - p.startup.weights_s).max(0.0);
            p.startup.jitter = 0.0;
            p.startup.restore_samples_s.clear();
            p.startup.build_samples_s.clear();
            p.host.model_bytes = 0;
            p.host.container_image_bytes = 0;
            p.host.measurement_noise = 0.0;
            p.evidence="empirical startup/quoted rental; effective serving parameters fitted only on hot-single and two hot policy workloads; region-specific serving variation unmeasured".into();
            p.source="native-cloud-observations-2026-10-11; model/runtime/controller hashes retained in dataset provenance".into();
        }
        let mut cases = vec![];
        for spec in strings(&specs["cases"]) {
            let name = spec["name"].as_str().ok_or("missing case name")?;
            let r = read(path.join("workloads").join(format!("{name}.json")))?;
            let cs = strings(&r["completions"]);
            if cs.len() != count(&r, "requests") as usize {
                return Err("incomplete workload cannot calibrate hidden output lengths".into());
            }
            let gap = n(spec, "arrival_interval_s");
            let adjusted: Vec<_> = cs
                .iter()
                .map(|c| n(c, "first_content_s") - count(c, "request") as f64 * gap)
                .collect();
            let begin = percentile(adjusted, 0.5).unwrap() - n(&r, "ttft_p50_s");
            let ttft: Vec<_> = cs
                .iter()
                .map(|c| n(c, "first_content_s") - begin - count(c, "request") as f64 * gap)
                .collect();
            if (percentile(ttft.clone(), 0.95).unwrap() - n(&r, "ttft_p95_s")).abs() > 1e-6 {
                return Err("arrival epoch disagrees with recorded percentiles".into());
            }
            let end = begin + n(&r, "elapsed_s");
            let budget = n(spec, "completion_budget_s");
            let warm: Vec<_> = owners
                .iter()
                .enumerate()
                .filter(|(i, o)| {
                    n(o, "allocated_s") < begin
                        && samples
                            .get(i)
                            .is_some_and(|s| n(o, "allocated_s") + s.ready_s <= begin)
                })
                .map(|(_, o)| WarmNode {
                    profile: profiles[count(o, "profile") as usize].id.clone(),
                    copies: 1,
                })
                .collect();
            let requests = cs
                .iter()
                .map(|c| {
                    let id = count(c, "request") as usize;
                    Request {
                        id,
                        job: 0,
                        model: model.clone(),
                        workload: Workload::Text,
                        class: "standard".into(),
                        arrival_s: id as f64 * gap,
                        input_tokens: count(c, "input_tokens"),
                        planned_output: count(spec, "planned_output").min(
                            if strings(&spec["output_bounds"]).is_empty() {
                                count(spec, "max_output")
                            } else {
                                strings(&spec["output_bounds"])
                                    [id % strings(&spec["output_bounds"]).len()]
                                .as_u64()
                                .unwrap()
                            },
                        ),
                        max_output: if strings(&spec["output_bounds"]).is_empty() {
                            count(spec, "max_output")
                        } else {
                            strings(&spec["output_bounds"])
                                [id % strings(&spec["output_bounds"]).len()]
                            .as_u64()
                            .unwrap()
                        },
                        actual_output: count(c, "output_tokens"),
                        image_megapixels: 0.0,
                        upload_bytes: 0,
                        cached_fraction: 0.0,
                        adapter: None,
                        latency_target_s: budget,
                        depends_on: if spec["serial_dependencies"] == true && id > 0 {
                            vec![id - 1]
                        } else {
                            vec![]
                        },
                    }
                })
                .collect::<Vec<_>>();
            let mut requests = requests;
            requests.sort_by_key(|r| r.id);
            let scenario=Scenario{topology:Default::default(),name:name.into(),notes:"Measured token counts/output lengths; output remains private to physics. Each workload resets to its observed warm fleet.".into(),horizon_s:budget,tick_s:n(&config["limits"],"poll_ms")/1000.0,idle_timeout_s:30.0,max_nodes_per_model:count(&config["limits"],"max_instances") as usize,
                launch_fail_probability:0.0,reactive_delay_s:0.0,capacity_scale:1.0,startup_scale:1.0,build_probability:0.0,scheduling:serde_json::from_value(config["scheduling"].clone())?,host_reality:None,warm,
                jobs:vec![Job{id:0,name:name.into(),announced_s:0.0,arrival_s:0.0,completion_budget_s:budget,desired_output_tps:n(spec,"desired_output_tps")}],requests,faults:vec![]};
            let mut observations = Observations::default();
            if serial {
                observations.controller_pauses = merge(
                    obs.iter()
                        .filter_map(|o| {
                            let start = (n(o, "start_s") - begin).max(0.0);
                            let stop = (n(o, "finish_s") - begin).min(budget);
                            (stop > start).then_some(Interval {
                                start_s: start,
                                end_s: stop,
                            })
                        })
                        .collect(),
                );
            }
            for (index, p) in profiles.iter().enumerate() {
                let initial = scenario
                    .warm
                    .iter()
                    .filter(|w| w.profile == p.id)
                    .map(|w| w.copies)
                    .sum();
                let mut capacity = vec![CapacityLimit {
                    at_s: 0.0,
                    instances: initial,
                }];
                let mut startups = vec![];
                for (j, o) in owners.iter().enumerate().filter(|(_, o)| {
                    count(o, "profile") as usize == index
                        && n(o, "allocated_s") >= begin
                        && n(o, "allocated_s") <= end
                }) {
                    // Availability is interval-censored. Use the lower observed bound,
                    // immediately after the last rejection preceding this success.
                    let last = strings(&rejected)
                        .iter()
                        .filter(|x| {
                            count(x, "profile") as usize == index
                                && n(x, "at_s") >= begin
                                && n(x, "at_s") < n(o, "allocated_s")
                        })
                        .map(|x| n(x, "at_s") - begin)
                        .max_by(f64::total_cmp);
                    let at = last.map_or(0.0, |t| t + 0.001);
                    let slots = capacity.last().unwrap().instances + 1;
                    if at <= capacity.last().unwrap().at_s {
                        capacity.last_mut().unwrap().instances = slots;
                    } else {
                        capacity.push(CapacityLimit {
                            at_s: at,
                            instances: slots,
                        });
                    }
                    if let Some(sample) = samples.get(&j) {
                        startups.push(sample.clone());
                    }
                }
                if startups.is_empty() {
                    startups = samples.values().cloned().collect();
                }
                observations.profiles.insert(
                    p.id.clone(),
                    ObservedEnvironment {
                        capacity,
                        startups,
                        rejection_delay_s: 1.0,
                        ..Default::default()
                    },
                );
            }
            cases.push(Case {
                model: model.clone(),
                policy: serde_json::from_value::<Policy>(r["policy"].clone())?,
                training: matches!(name, "hot-single" | "hot-reactive" | "hot-planned_hybrid"),
                scenario,
                observations,
                measured: Metrics {
                    completed: count(&r, "completed") as usize,
                    elapsed_s: n(&r, "elapsed_s"),
                    output_tps: n(&r, "output_tokens_per_s"),
                    ttft_p50_s: n(&r, "ttft_p50_s"),
                    ttft_p95_s: n(&r, "ttft_p95_s"),
                    ttft_p99_s: percentile(ttft, 0.99).unwrap(),
                    rental_usd: n(&r, "estimated_rental_usd"),
                },
            });
        }
        output.push(ImportedModel { profiles, cases });
    }
    Ok(output)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 3 {
        return Err("usage: OUT SERIAL_RUN BACKGROUND_RUN".into());
    }
    let mut dataset = Dataset {
        profiles: vec![],
        cases: vec![],
        provenance: json!({"format":"native-observations-v1","scope":"three single cloud runs; 0.8B/2B serial observer, 4B background observer; first use penalty estimated from cold minus median warm prefill; availability lower bounds, not production distributions","training":["hot-single","hot-reactive","hot-planned_hybrid"],"validation":"all other cases; no validation loss is used in fitting","sources":[read(Path::new(&args[1]).join("observer-source.json"))?,read(Path::new(&args[2]).join("observer-source.json"))?]}),
    };
    for (root, serial) in [(&args[1], true), (&args[2], false)] {
        for model in import(Path::new(root), serial)? {
            dataset.profiles.extend(model.profiles);
            dataset.cases.extend(model.cases);
        }
    }
    // Profiles may share a model/GPU but never share region-specific launch realities.
    fs::create_dir_all(Path::new(&args[0]).parent().ok_or("output parent")?)?;
    fs::write(&args[0], serde_json::to_vec_pretty(&dataset)?)?;
    println!(
        "exported {} profiles and {} observed cases",
        dataset.profiles.len(),
        dataset.cases.len()
    );
    Ok(())
}
