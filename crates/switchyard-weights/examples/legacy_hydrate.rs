//! Benchmark entry point invoking the unchanged legacy production hydrator.
use std::{path::Path, sync::Arc, time::Instant};
use switchyard_agent::{State, Ticket};
use tokio::sync::RwLock;
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().collect::<Vec<_>>();
    let ticket: Ticket = serde_json::from_slice(&std::fs::read(&args[1])?)?;
    let state = Arc::new(RwLock::new(State::new(&ticket)));
    let start = Instant::now();
    switchyard_agent::hydrate::hydrate(&ticket, Path::new(&args[2]), state).await?;
    println!(
        "{}",
        serde_json::json!({"event":"hydration_complete", "seconds":start.elapsed().as_secs_f64(), "wire_bytes":ticket.manifest.total_size})
    );
    Ok(())
}
