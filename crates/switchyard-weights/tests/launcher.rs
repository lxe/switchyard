//! Spawned Python workers must import the launcher without consuming its gate.
use std::{
    fs,
    io::Write,
    process::{Command, Stdio},
};

#[test]
fn spawned_worker_does_not_repeat_metadata_gate() {
    let root = std::env::temp_dir().join(format!("switchyard-loader-spawn-{}", std::process::id()));
    fs::create_dir(&root).unwrap();
    switchyard_weights::write_vllm_bundle(&root.join("plugin")).unwrap();
    let package = root.join("plugin/vllm/entrypoints/cli");
    fs::create_dir_all(&package).unwrap();
    for path in ["vllm", "vllm/entrypoints", "vllm/entrypoints/cli"] {
        fs::write(root.join("plugin").join(path).join("__init__.py"), "").unwrap();
    }
    fs::write(
        root.join("plugin/worker.py"),
        "def child():\n    print('spawned worker ready', flush=True)\n",
    )
    .unwrap();
    fs::write(package.join("main.py"), "import multiprocessing\nfrom worker import child\np = multiprocessing.get_context('spawn').Process(target=child)\np.start()\np.join(10)\nassert p.exitcode == 0, p.exitcode\n").unwrap();
    let mut process = Command::new("python3")
        .arg(root.join("plugin/serve.py"))
        .env("PYTHONPATH", root.join("plugin"))
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    process
        .stdin
        .take()
        .unwrap()
        .write_all(b"switchyard-metadata-ready-v1\n")
        .unwrap();
    let output = process.wait_with_output().unwrap();
    fs::remove_dir_all(root).unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("spawned worker ready"));
}

#[test]
fn prepared_startup_cpu_acceptance() {
    let output = Command::new("python3")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/python/tests/test_startup.py"
        ))
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}
