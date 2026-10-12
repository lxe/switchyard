//! CPU-only CRIU preparation/restore adapter. No scheduler or GPU snapshot behavior.
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    io::{IoSlice, Read, Write},
    mem::MaybeUninit,
    os::{
        fd::{AsFd, AsRawFd},
        unix::{fs::MetadataExt, net::UnixStream},
    },
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    time::{Duration, Instant},
};
use switchyard_artifacts::{Manifest, Metadata, PackOptions};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
/// Names a file whose existence means the model and runtime dependencies are published.
pub const READY_FILE: &str = "SWITCHYARD_READY_FILE";
pub const PAYLOAD: &str = include_str!("../python/switchyard_cpu_snapshot.py");
fn payload_sha() -> String {
    sha(format!("{PAYLOAD}{}", crate::loader::CPU_TREE).as_bytes())
}
const LIMIT: usize = 65536;

/// Wait until the file named by [`READY_FILE`] exists. Standalone use, without
/// the variable, proceeds at once. The owner cancels by killing this process.
pub fn wait_ready() {
    let Some(path) = std::env::var_os(READY_FILE).filter(|path| !path.is_empty()) else {
        return;
    };
    let path = PathBuf::from(path);
    while !path.exists() {
        std::thread::sleep(Duration::from_millis(5));
    }
}

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn file_sha(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path)?;
    let mut digest = Sha256::new();
    let mut bytes = vec![0; 1024 * 1024];
    loop {
        let size = file.read(&mut bytes)?;
        if size == 0 {
            break;
        }
        digest.update(&bytes[..size]);
    }
    Ok(format!("{:x}", digest.finalize()))
}
fn write<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    fs::write(path, format!("{}\n", serde_json::to_string_pretty(value)?))?;
    Ok(())
}
fn read<T: for<'a> Deserialize<'a>>(path: &Path) -> Result<T> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}
fn monotonic() -> f64 {
    let clock = rustix::time::clock_gettime(rustix::time::ClockId::Monotonic);
    clock.tv_sec as f64 + clock.tv_nsec as f64 / 1e9
}
fn emit(event: &str, start: Instant, values: Value) {
    println!(
        "SWITCHYARD_CPU_SNAPSHOT {}",
        json!({"event":event,"monotonic_s":monotonic(),"elapsed_s":start.elapsed().as_secs_f64(),"values":values})
    );
}
fn receive(socket: &mut UnixStream) -> Result<Value> {
    let mut header = [0; 4];
    socket.read_exact(&mut header)?;
    let size = u32::from_be_bytes(header) as usize;
    if size == 0 || size > LIMIT {
        return Err("CPU snapshot control frame exceeds limit".into());
    }
    let mut bytes = vec![0; size];
    socket.read_exact(&mut bytes)?;
    Ok(serde_json::from_slice(&bytes)?)
}
fn send(socket: &mut UnixStream, value: &Value) -> Result<()> {
    let bytes = serde_json::to_vec(value)?;
    if bytes.is_empty() || bytes.len() > LIMIT {
        return Err("CPU snapshot fresh configuration exceeds limit".into());
    }
    socket.write_all(&(bytes.len() as u32).to_be_bytes())?;
    socket.write_all(&bytes)?;
    Ok(())
}
fn inheritable(fd: &impl AsFd, value: bool) -> Result<()> {
    rustix::io::fcntl_setfd(
        fd,
        if value {
            rustix::io::FdFlags::empty()
        } else {
            rustix::io::FdFlags::CLOEXEC
        },
    )?;
    Ok(())
}
fn criu(program: &Path) -> Command {
    let mut command = Command::new(program);
    command
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .arg("--no-default-config");
    command
}
fn checked(command: &mut Command) -> Result<()> {
    if !command.status()?.success() {
        return Err("CRIU operation failed; inspect dump/restore log".into());
    }
    Ok(())
}
fn fresh(parent: &Path, prefix: &str) -> Result<PathBuf> {
    fs::create_dir_all(parent)?;
    let mut random = [0; 16];
    getrandom::fill(&mut random)?;
    let name: String = random.iter().map(|b| format!("{b:02x}")).collect();
    let path = parent.canonicalize()?.join(format!(".{prefix}-{name}"));
    fs::create_dir(&path)?;
    Ok(path)
}
struct OwnedChild(Child);
impl Drop for OwnedChild {
    fn drop(&mut self) {
        if self.0.try_wait().ok().flatten().is_none() {
            if let Some(pid) = rustix::process::Pid::from_raw(self.0.id() as i32) {
                let _ = rustix::process::kill_process_group(pid, rustix::process::Signal::KILL);
            }
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
}

struct RestoredChild {
    pid: rustix::process::Pid,
    active: bool,
}
impl Drop for RestoredChild {
    fn drop(&mut self) {
        if self.active {
            let _ = rustix::process::kill_process_group(self.pid, rustix::process::Signal::KILL);
            let _ = rustix::process::kill_process(self.pid, rustix::process::Signal::KILL);
            let _ = rustix::process::waitpid(Some(self.pid), rustix::process::WaitOptions::empty());
        }
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct HostProfile {
    pub architecture: String,
    pub kernel_release: String,
    pub cpu_flags_sha256: String,
    pub cpu_vendor_model: String,
    pub criu_version: String,
    pub base_os_sha256: String,
}
impl HostProfile {
    pub fn inspect(program: &Path) -> Result<Self> {
        let version = Command::new(program).arg("--version").output()?;
        if !version.status.success() {
            return Err("CRIU executable is unavailable".into());
        }
        let cpu = fs::read_to_string("/proc/cpuinfo")?;
        let first = cpu.split("\n\n").next().ok_or("missing CPU profile")?;
        let field = |name: &str| {
            first
                .lines()
                .find_map(|line| {
                    line.split_once(':')
                        .filter(|(k, _)| k.trim() == name)
                        .map(|(_, v)| v.trim())
                })
                .unwrap_or("")
        };
        let mut flags: Vec<_> = field("flags").split_whitespace().collect();
        flags.sort();
        let release = Command::new("uname").arg("-r").output()?;
        Ok(Self {
            architecture: std::env::consts::ARCH.into(),
            kernel_release: String::from_utf8(release.stdout)?.trim().into(),
            cpu_flags_sha256: sha(flags.join(" ").as_bytes()),
            cpu_vendor_model: format!("{} {}", field("vendor_id"), field("model name")),
            criu_version: String::from_utf8(version.stdout)?.trim().into(),
            base_os_sha256: file_sha(Path::new("/etc/os-release"))?,
        })
    }
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Snapshot {
    pub schema_version: u32,
    pub runtime_id: String,
    pub runtime_root: PathBuf,
    pub payload_sha256: String,
    pub profile: HostProfile,
    pub control_socket_inode: u64,
    pub prepared: Value,
    pub base_files: BTreeMap<PathBuf, String>,
}
fn validate(snapshot: &Snapshot, root: &Path, runtime_id: &str, host: &HostProfile) -> Result<()> {
    if snapshot.schema_version != 1
        || snapshot.runtime_id != runtime_id
        || snapshot.runtime_root != root
        || snapshot.payload_sha256 != payload_sha()
        || &snapshot.profile != host
    {
        return Err(
            "CPU snapshot runtime/payload/CPU/kernel/base-image/CRIU profile differs".into(),
        );
    }
    if snapshot.prepared["cuda_initialized"] != false
        || snapshot.prepared["platform_resolved"] != true
        || snapshot.prepared["vllm_import_complete"] != true
        || snapshot.prepared["boundary"] != "Worker.init_device"
        || snapshot.prepared["gpu_device_fds"] != json!([])
    {
        return Err("CPU snapshot is not a complete GPU-free API/worker tree".into());
    }
    let workers = snapshot.prepared["workers"]
        .as_array()
        .ok_or("missing prepared worker tree")?;
    if workers.len() != 1
        || workers.iter().any(|worker| {
            worker["cuda_initialized"] != false || worker["gpu_device_fds"] != json!([])
        })
    {
        return Err("CPU snapshot requires one prepared worker without GPU state".into());
    }
    for (path, expected) in &snapshot.base_files {
        if file_sha(path)? != *expected {
            return Err(
                format!("CPU snapshot mapped base file differs: {}", path.display()).into(),
            );
        }
    }
    Ok(())
}

pub struct Prepare {
    pub runtime_root: PathBuf,
    pub runtime_manifest: PathBuf,
    pub serving_config: PathBuf,
    pub out: PathBuf,
    pub criu: PathBuf,
}
pub fn prepare(options: Prepare) -> Result<Value> {
    let start = Instant::now();
    if fs::symlink_metadata(&options.out).is_ok() {
        return Err("CPU snapshot output must be fresh".into());
    }
    let root = options.runtime_root.canonicalize()?;
    let manifest: Manifest = read(&options.runtime_manifest)?;
    emit(
        "runtime_verification_started",
        start,
        json!({"runtime_id":manifest.id}),
    );
    switchyard_artifacts::verify(&root, &manifest)?;
    let profile = HostProfile::inspect(&options.criu)?;
    checked(criu(&options.criu).arg("check"))?;
    let staging = fresh(
        options.out.parent().unwrap_or(Path::new(".")),
        "cpu-snapshot",
    )?;
    let source = staging.join("tree");
    fs::create_dir(&source)?;
    fs::create_dir(source.join("images"))?;
    fs::create_dir(source.join("no-plugins"))?;
    fs::write(staging.join("payload.py"), PAYLOAD)?;
    let serving: Value = read(&options.serving_config)?;
    write(
        &staging.join("prepare.json"),
        &json!({"schema_version":1,"runtime_id":manifest.id,"serving":serving}),
    )?;
    let libraries = manifest
        .entries
        .iter()
        .filter_map(|entry| {
            let path = entry.path();
            matches!(entry, switchyard_artifacts::Entry::Directory { .. }).then_some(path)
        })
        .filter(|p| {
            matches!(*p, "lib" | "python/lib")
                || p.ends_with("/lib") && (p.contains("/torch/") || p.contains("/nvidia/"))
        })
        .map(|p| root.join(p).to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join(":");
    let (mut control, external) = UnixStream::pair()?;
    control.set_read_timeout(Some(Duration::from_secs(120)))?;
    inheritable(&external, true)?;
    let inode = fs::metadata(format!("/proc/self/fd/{}", external.as_raw_fd()))?.ino();
    let mut command = Command::new("setsid");
    command
        .arg(root.join("python/bin/python3.12"))
        .arg(staging.join("payload.py"))
        .arg(external.as_raw_fd().to_string())
        .arg(staging.join("prepare.json"))
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .env("LANG", "C.UTF-8")
        .env("PYTHONHOME", root.join("python"))
        .env("PYTHONPATH", root.join("venv/lib/python3.12/site-packages"))
        .env("LD_LIBRARY_PATH", libraries)
        .env("TOKENIZERS_PARALLELISM", "false")
        .env("OMP_NUM_THREADS", "1")
        .env("HF_HUB_OFFLINE", "1")
        .env("TRANSFORMERS_OFFLINE", "1")
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .env("VLLM_NO_USAGE_STATS", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    {
        let environment: BTreeMap<String, String> =
            serde_json::from_value(serving["environment"].clone())?;
        command.envs(environment);
        // The bake has every prerequisite on disk, so its launcher proceeds at once.
        let ready = staging.join("ready");
        fs::write(&ready, b"")?;
        command.env(READY_FILE, ready);
        let log = fs::File::create(staging.join("preparation.log"))?;
        command.stdout(log.try_clone()?).stderr(log);
    }
    emit("cpu_prepare_started", start, json!({"gpu_snapshot":false}));
    let mut child = OwnedChild(command.spawn()?);
    drop(external);
    let prepared = receive(&mut control)?;
    if let Some(error) = prepared.get("error").and_then(Value::as_str) {
        return Err(format!(
            "CPU preparation failed: {error}; details: {}",
            staging.join("preparation.log").display()
        )
        .into());
    }
    let mut summary = prepared.clone();
    if let Some(value) = summary.as_object_mut() {
        value.remove("mapped_files");
    }
    emit("cpu_prepared", start, summary);
    let mut base_files = BTreeMap::new();
    for file in prepared["mapped_files"]
        .as_array()
        .ok_or("missing mapped file inventory")?
    {
        let path = PathBuf::from(file.as_str().ok_or("invalid mapped file")?);
        if !path.starts_with(&root) {
            base_files.insert(path.clone(), file_sha(&path)?);
        }
    }
    let snapshot = Snapshot {
        schema_version: 1,
        runtime_id: manifest.id.clone(),
        runtime_root: root,
        payload_sha256: payload_sha(),
        profile,
        control_socket_inode: inode,
        prepared,
        base_files,
    };
    validate(
        &snapshot,
        &snapshot.runtime_root,
        &snapshot.runtime_id,
        &snapshot.profile,
    )?;
    let checkpoint_start = Instant::now();
    emit("cpu_checkpoint_started", start, json!({"pid":child.0.id()}));
    checked(
        criu(&options.criu)
            .arg("dump")
            .args(["--tree", &child.0.id().to_string()])
            .arg("--images-dir")
            .arg(source.join("images"))
            .arg("--external")
            .arg(format!("unix[{inode}]"))
            .args(["--cpu-cap=all", "--shell-job"])
            .arg("--libdir")
            .arg(source.join("no-plugins"))
            .args(["--log-file", "dump.log", "-v4"]),
    )?;
    child.0.wait()?;
    fs::remove_file(staging.join("ready"))?;
    emit(
        "cpu_checkpoint_complete",
        start,
        json!({"duration_s":checkpoint_start.elapsed().as_secs_f64(),"image_bytes":fs::read_dir(source.join("images"))?.filter_map(std::result::Result::ok).filter_map(|e| e.metadata().ok()).map(|m|m.len()).sum::<u64>()}),
    );
    write(&source.join("cpu-snapshot.json"), &snapshot)?;
    let mut compatibility = manifest.metadata.compatibility.clone();
    compatibility.remove("cuda");
    compatibility.insert("runtime_id".into(), manifest.id.clone());
    compatibility.insert(
        "cpu_snapshot_profile".into(),
        sha(&serde_json::to_vec(&snapshot.profile)?),
    );
    let metadata = Metadata {
        kind: "cpu_process".into(),
        compatibility,
        attributes: BTreeMap::from([
            ("payload_sha256".into(), snapshot.payload_sha256.clone()),
            ("gpu_snapshot".into(), "false".into()),
        ]),
    };
    emit("cpu_snapshot_pack_started", start, json!({}));
    let packed = switchyard_artifacts::pack(
        &source,
        &staging.join("artifact"),
        metadata,
        PackOptions::default(),
    )?;
    write(&staging.join("artifact/manifest.json"), &packed)?;
    switchyard_artifacts::verify(&source, &packed)?;
    let receipt = json!({"snapshot_artifact_id":packed.id,"compressed_bytes":packed.chunks.iter().map(|c|c.size).sum::<u64>(),"runtime_id":manifest.id,"prepared":snapshot.prepared,"profile":snapshot.profile,"gpu_snapshot":false});
    write(&staging.join("receipt.json"), &receipt)?;
    rustix::fs::renameat_with(
        rustix::fs::CWD,
        &staging,
        rustix::fs::CWD,
        &options.out,
        rustix::fs::RenameFlags::NOREPLACE,
    )?;
    emit(
        "cpu_snapshot_published",
        start,
        json!({
            "snapshot_artifact_id": packed.id,
            "compressed_bytes": receipt["compressed_bytes"],
            "gpu_snapshot": false,
        }),
    );
    Ok(receipt)
}

pub struct Restore {
    pub snapshot: PathBuf,
    pub runtime_root: PathBuf,
    pub runtime_id: String,
    pub work: PathBuf,
    pub criu: PathBuf,
    pub probe: bool,
    pub args: Vec<String>,
}
fn fresh_environment() -> BTreeMap<String, String> {
    std::env::vars()
        .filter(|(key, _)| {
            matches!(
                key.as_str(),
                "PATH"
                    | "PYTHONHOME"
                    | "LD_LIBRARY_PATH"
                    | "PYTHONPATH"
                    | "LANG"
                    | "LC_ALL"
                    | "HOME"
                    | "OMP_NUM_THREADS"
                    | "CUDA_HOME"
                    | "CUDA_PATH"
                    | "CUDA_VISIBLE_DEVICES"
                    | "NVIDIA_VISIBLE_DEVICES"
                    | "NVIDIA_DRIVER_CAPABILITIES"
                    | "TOKENIZERS_PARALLELISM"
                    | "TRITON_CACHE_DIR"
                    | "TORCHINDUCTOR_CACHE_DIR"
                    | "CUDA_CACHE_PATH"
                    | "HF_HOME"
                    | "HF_HUB_OFFLINE"
                    | "TRANSFORMERS_OFFLINE"
                    | "PYTHONDONTWRITEBYTECODE"
                    | "NATIVE_STARTUP_PROFILE"
                    | "NATIVE_ENV_OVERRIDE_SOURCE"
                    | "NATIVE_CLI_MAIN_SOURCE"
                    | "NATIVE_MODULE_SOURCES"
            ) || key.starts_with("VLLM_")
                || key.starts_with("FLASHINFER_")
                || matches!(
                    key.as_str(),
                    "SWITCHYARD_RUNTIME_ROOT"
                        | "SWITCHYARD_RUNTIME_ARTIFACT_ID"
                        | "SWITCHYARD_CACHE_ROOT"
                        | "SWITCHYARD_COMPONENTS_ROOT"
                        | "SWITCHYARD_STARTUP_EPOCH"
                        | "SWITCHYARD_WEIGHTS_SOCKET"
                        | "SWITCHYARD_LOADER_LEASE"
                        | "SWITCHYARD_MODEL_CONTENT_ID"
                )
        })
        .collect()
}
pub fn restore(options: Restore) -> Result<Value> {
    let start = Instant::now();
    let root = options.runtime_root.canonicalize()?;
    let snapshot: Snapshot = read(&options.snapshot.join("cpu-snapshot.json"))?;
    validate(
        &snapshot,
        &root,
        &options.runtime_id,
        &HostProfile::inspect(&options.criu)?,
    )?;
    emit("cpu_restore_waiting_for_prerequisites", start, json!({}));
    wait_ready();
    let work = fresh(&options.work, "cpu-restore")?;
    checked(
        criu(&options.criu)
            .args(["cpuinfo", "check"])
            .arg("--images-dir")
            .arg(options.snapshot.join("images")),
    )?;
    let (mut control, external) = UnixStream::pair()?;
    control.set_read_timeout(Some(Duration::from_secs(120)))?;
    inheritable(&external, true)?;
    emit(
        "cpu_restore_started",
        start,
        json!({"runtime_id":options.runtime_id}),
    );
    let restore_start = Instant::now();
    checked(
        criu(&options.criu)
            .arg("restore")
            .arg("--images-dir")
            .arg(options.snapshot.join("images"))
            .arg("--work-dir")
            .arg(&work)
            .arg("--libdir")
            .arg(options.snapshot.join("no-plugins"))
            .arg("--inherit-fd")
            .arg(format!(
                "fd[{}]:socket:[{}]",
                external.as_raw_fd(),
                snapshot.control_socket_inode
            ))
            .args([
                // The restored process belongs to this launch's cgroup, not
                // the preparation container's resource accounting hierarchy.
                "--manage-cgroups=ignore",
                "--cpu-cap=all",
                "--shell-job",
                "--restore-detached",
                "--restore-sibling",
                "--pidfile",
            ])
            .arg(work.join("restored.pid"))
            .args(["--log-file", "restore.log", "-v4"]),
    )?;
    drop(external);
    let pid: u32 = fs::read_to_string(work.join("restored.pid"))?
        .trim()
        .parse()?;
    let mut restored = RestoredChild {
        pid: rustix::process::Pid::from_raw(pid as i32).ok_or("invalid restored PID")?,
        active: true,
    };
    let (input, output) = rustix::pipe::pipe_with(rustix::pipe::PipeFlags::CLOEXEC)?;
    let stdout = std::io::stdout();
    let stderr = std::io::stderr();
    let descriptors = [input.as_fd(), stdout.as_fd(), stderr.as_fd()];
    let mut space = [MaybeUninit::uninit(); rustix::cmsg_space!(ScmRights(3))];
    let mut ancillary = rustix::net::SendAncillaryBuffer::new(&mut space);
    if !ancillary.push(rustix::net::SendAncillaryMessage::ScmRights(&descriptors)) {
        return Err("cannot bind restored stdio".into());
    }
    rustix::net::sendmsg(
        &control,
        &[IoSlice::new(b"F")],
        &mut ancillary,
        rustix::net::SendFlags::empty(),
    )?;
    // The restored process is past its readiness wait; its fresh stdin is empty.
    drop(output);
    let mut environment = fresh_environment();
    environment.insert(
        "SWITCHYARD_RUNTIME_ROOT".into(),
        root.to_string_lossy().into_owned(),
    );
    environment.insert(
        "SWITCHYARD_RUNTIME_ARTIFACT_ID".into(),
        options.runtime_id.clone(),
    );
    environment.insert("OMP_NUM_THREADS".into(), "1".into());
    let clock_send = monotonic();
    send(
        &mut control,
        &json!({"schema_version":1,"environment":environment,"args":options.args,"probe":options.probe,"runtime_id":options.runtime_id}),
    )?;
    let mut reply = receive(&mut control)?;
    let clock_received = monotonic();
    if reply["objects"] != snapshot.prepared["objects"]
        || reply["objects_preserved"] != true
        || reply["cuda_initialized"] != false
    {
        return Err("CPU restore state validation failed".into());
    }
    let child_clock = reply["child_monotonic_s"]
        .as_f64()
        .filter(|clock| clock.is_finite())
        .ok_or("restored CPU clock sample is absent")?;
    let clock_offset = (clock_send + clock_received) / 2.0 - child_clock;
    let clock_uncertainty = (clock_received - clock_send) / 2.0;
    send(
        &mut control,
        &json!({"event":"clock_calibration","offset_s":clock_offset,"uncertainty_s":clock_uncertainty}),
    )?;
    reply["clock_offset_s"] = json!(clock_offset);
    reply["clock_uncertainty_s"] = json!(clock_uncertainty);
    write(&work.join("receipt.json"), &reply)?;
    let mut telemetry = reply.clone();
    telemetry["duration_s"] = json!(restore_start.elapsed().as_secs_f64());
    emit("cpu_restore_complete", start, telemetry);
    let (_, status) =
        rustix::process::waitpid(Some(restored.pid), rustix::process::WaitOptions::empty())?
            .ok_or("restored child wait returned no status")?;
    restored.active = false;
    if status.exit_status() != Some(0) {
        return Err(format!("restored native process failed: {status:?}").into());
    }
    Ok(reply)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot() -> Snapshot {
        Snapshot {
            schema_version: 1,
            runtime_id: "runtime".into(),
            runtime_root: "/runtime".into(),
            payload_sha256: payload_sha(),
            profile: HostProfile {
                architecture: "x86_64".into(),
                kernel_release: "kernel".into(),
                cpu_flags_sha256: "flags".into(),
                cpu_vendor_model: "cpu".into(),
                criu_version: "criu".into(),
                base_os_sha256: "base".into(),
            },
            control_socket_inode: 1,
            prepared: json!({"cuda_initialized":false,"platform_resolved":true,"vllm_import_complete":true,
                "boundary":"Worker.init_device", "gpu_device_fds":[],
                "workers":[{"cuda_initialized":false,"gpu_device_fds":[]}]}),
            base_files: BTreeMap::new(),
        }
    }

    #[test]
    fn restore_rejects_gpu_state_and_different_runtime_or_kernel() {
        let mut saved = snapshot();
        let root = Path::new("/runtime");
        assert!(validate(&saved, root, "runtime", &saved.profile).is_ok());
        assert!(validate(&saved, root, "different", &saved.profile).is_err());
        assert!(validate(&saved, Path::new("/other"), "runtime", &saved.profile).is_err());
        let mut host = HostProfile {
            ..snapshot().profile
        };
        host.kernel_release = "other".into();
        assert!(validate(&saved, root, "runtime", &host).is_err());
        saved.prepared["cuda_initialized"] = json!(true);
        assert!(validate(&saved, root, "runtime", &saved.profile).is_err());
        saved.prepared["cuda_initialized"] = json!(false);
        saved.prepared["gpu_device_fds"] = json!(["/dev/nvidia0"]);
        assert!(validate(&saved, root, "runtime", &saved.profile).is_err());
        saved = snapshot();
        saved.prepared["workers"][0]["cuda_initialized"] = json!(true);
        assert!(validate(&saved, root, "runtime", &saved.profile).is_err());
        saved = snapshot();
        saved.prepared["workers"][0]["gpu_device_fds"] = json!(["/dev/nvidiactl"]);
        assert!(validate(&saved, root, "runtime", &saved.profile).is_err());
        saved = snapshot();
        saved.prepared["workers"] = json!([]);
        assert!(validate(&saved, root, "runtime", &saved.profile).is_err());
        saved = snapshot();
        saved.prepared["vllm_import_complete"] = json!(false);
        assert!(validate(&saved, root, "runtime", &saved.profile).is_err());
    }

    #[test]
    fn external_control_rejects_oversized_and_truncated_frames() {
        let (mut receiver, mut writer) = UnixStream::pair().unwrap();
        writer
            .write_all(&((LIMIT + 1) as u32).to_be_bytes())
            .unwrap();
        assert!(receive(&mut receiver).is_err());
        let (mut receiver, mut writer) = UnixStream::pair().unwrap();
        writer.write_all(&8u32.to_be_bytes()).unwrap();
        writer.write_all(b"{").unwrap();
        drop(writer);
        assert!(receive(&mut receiver).is_err());
    }
}
