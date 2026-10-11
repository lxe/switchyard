use clap::Parser;
use switchyard_quantized::{
    gdn::Qwen35Requirements,
    gguf::{Index, IndexLimits},
};
#[derive(Parser)]
struct Args {
    file: std::path::PathBuf,
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let mut file = std::fs::File::open(args.file)?;
    let index = Index::read(&mut file, IndexLimits::default())?;
    let requirements = Qwen35Requirements::inspect(&index)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "scope":"GGUF format and recurrent layout inspection, not native serving admission",
            "file_bytes":index.file_bytes(),"header_bytes":index.header_bytes(),
            "tensors":index.tensors().collect::<Vec<_>>(),"requirements":requirements
        }))?
    );
    Ok(())
}
