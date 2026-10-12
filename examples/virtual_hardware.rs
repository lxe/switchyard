//! Compare hypothetical compute, memory, and network tiers on identical requests.
use std::{error::Error, fmt::Write, fs, path::PathBuf};
use switchyard::{Policy, Profile, Scenario, WarmNode, simulate};

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args_os().skip(1);
    let catalog = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| "configs/virtual-hardware/profiles.json".into());
    let out = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| "results/virtual-hardware".into());
    if args.next().is_some() {
        return Err("usage: virtual_hardware [PROFILES_JSON] [OUTPUT_DIRECTORY]".into());
    }
    let profiles: Vec<Profile> = serde_json::from_slice(&fs::read(&catalog)?)?;
    let template: Scenario =
        serde_json::from_str(include_str!("../configs/quickstart/burst.json"))?;
    let mut comparisons = vec![];
    fs::create_dir_all(&out)?;
    fs::write(
        out.join("profiles.json"),
        serde_json::to_vec_pretty(&profiles)?,
    )?;
    for (index, profile) in profiles.iter().enumerate() {
        for (name, count, warm, upload_bytes) in [
            ("cold-single", 1, false, 4096),
            ("cold-burst-1000", 1000, false, 4096),
            ("hot-burst-1000", 1000, true, 4096),
            ("hot-upload-heavy-1000", 1000, true, 64 * 1024 * 1024),
        ] {
            let mut scenario = template.clone();
            scenario.name = format!("{}:{name}", profile.id);
            scenario.notes = "Hypothetical single-instance sensitivity comparison; upload-heavy uses transport padding to isolate upload delay. No provider or kernel measurements.".into();
            scenario.horizon_s = 600.0;
            scenario.idle_timeout_s = 5.0;
            scenario.max_nodes_per_model = 1;
            scenario.reactive_delay_s = 0.0;
            scenario.scheduling.warm_batch_capacity_fraction = 1.0;
            scenario.scheduling.reserve_warm_slots = 0;
            scenario.warm = if warm {
                vec![WarmNode {
                    profile: profile.id.clone(),
                    copies: 1,
                }]
            } else {
                vec![]
            };
            let job = &mut scenario.jobs[0];
            job.name = name.into();
            job.announced_s = 0.0;
            job.arrival_s = 0.0;
            job.completion_budget_s = scenario.horizon_s;
            job.desired_output_tps = 0.0;
            let request = &template.requests[0];
            scenario.requests = (0..count)
                .map(|id| {
                    let mut r = request.clone();
                    r.id = id;
                    r.model = profile.model.clone();
                    r.workload = profile.workload;
                    r.arrival_s = 0.0;
                    r.upload_bytes = upload_bytes;
                    r.latency_target_s = scenario.horizon_s;
                    r
                })
                .collect();
            let policy = if warm {
                Policy::WarmOnly
            } else {
                Policy::Reactive
            };
            let run = simulate(std::slice::from_ref(profile), &scenario, policy, 7)?;
            if run.summary.completed != count || run.summary.invalid_decisions > 0 {
                return Err(format!("{} did not complete cleanly", scenario.name).into());
            }
            let case_out = out.join(format!("{index}-{name}"));
            fs::create_dir_all(&case_out)?;
            fs::write(
                case_out.join("scenario.json"),
                serde_json::to_vec_pretty(&scenario)?,
            )?;
            fs::write(case_out.join("run.json"), serde_json::to_vec_pretty(&run)?)?;
            comparisons.push(run);
        }
    }
    let mut markdown = String::from(
        "# Virtual hardware comparison\n\nHypothetical single-instance simulations; all hardware and network values are assumptions.\n\n| Scenario | Completed | Elapsed, s | Output tok/s | TTFT p95, s |\n| --- | ---: | ---: | ---: | ---: |\n",
    );
    for run in &comparisons {
        let summary = &run.summary;
        let job = &summary.jobs[0];
        writeln!(
            markdown,
            "| {} | {} | {:.3} | {:.1} | {:.3} |",
            summary.scenario,
            summary.completed,
            job.elapsed_s.ok_or("completed job lacks elapsed time")?,
            job.output_tps,
            summary
                .first_token_p95_s
                .ok_or("completed job lacks first token time")?,
        )?;
    }
    fs::write(out.join("comparison.md"), &markdown)?;
    fs::write(
        out.join("summaries.json"),
        serde_json::to_vec_pretty(&comparisons.iter().map(|r| &r.summary).collect::<Vec<_>>())?,
    )?;
    print!("{markdown}");
    Ok(())
}
