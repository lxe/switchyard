//! Portable machine-readable results plus a compact comparison table.
use crate::engine::Run;
use std::fmt::Write;

pub fn markdown(runs: &[Run]) -> String {
    let mut text = String::from(
        "# Fleet simulation results\n\nThese are simulated outcomes, not live cloud benchmarks. Read profile provenance before interpreting a hardware comparison. Cost includes baseline rent over the entire observation window, startup, minimum billing, and elastic idle timeout.\n\n| Scenario | Policy | Completed | Eval seconds | Eval output TPS | Interactive p95 seconds | Interactive SLO | USD | Idle USD | Elastic launches | Target jobs met |\n|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    );
    for run in runs {
        let s = &run.summary;
        let number = |v: Option<f64>| v.map_or("—".into(), |x| format!("{x:.2}"));
        let class_latencies = s
            .classes
            .iter()
            .map(|c| format!("{}: {}", c.name, number(c.latency_p95_s)))
            .collect::<Vec<_>>()
            .join("; ");
        let class_slos = s
            .classes
            .iter()
            .map(|c| format!("{}: {:.1}%", c.name, c.slo_fraction * 100.0))
            .collect::<Vec<_>>()
            .join("; ");
        writeln!(
            text,
            "| {} | {} | {} | {} | {:.1} | {} | {} | {:.4} | {:.4} | {} | {}/{} |",
            s.scenario,
            s.policy,
            s.completed,
            number(s.batch_elapsed_s),
            s.batch_output_tps,
            class_latencies,
            class_slos,
            s.cost_usd,
            s.idle_cost_usd,
            s.launched,
            s.target_jobs_met,
            s.target_jobs_total
        )
        .expect("writing into String cannot fail");
    }
    text.push_str("\nEval rate divides completed output tokens by the interval from the first eval arrival to the last completed eval response. Check `unfinished` and per-job completion before interpreting partial runs. The interactive SLO denominator includes unfinished arrived requests. JSON includes request timing, instance costs, failures, and a one-second fleet timeline.\n");
    text
}
