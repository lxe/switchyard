//! One owner controls the engine child and all processes in its Unix process group.
use crate::{Error, Result};
use nix::{
    errno::Errno,
    sys::signal::{Signal, killpg},
    unistd::Pid,
};
use std::{fs, sync::Arc, time::Duration};
use tokio::{
    process::{Child, Command},
    sync::{mpsc, oneshot},
    time::Instant,
};

pub(super) struct Stop {
    pub(super) timeout: Duration,
    pub(super) reply: oneshot::Sender<Result<()>>,
}
pub(super) type Control = mpsc::Sender<Stop>;

pub(super) struct Process {
    pub(super) child: Child,
    group: Pid,
    stopped: bool,
    retained: Vec<Arc<dyn Send + Sync>>,
}
impl Process {
    pub(super) fn spawn(command: &mut Command) -> Result<Self> {
        command.process_group(0).kill_on_drop(true);
        let child = command.spawn()?;
        let pid = child
            .id()
            .and_then(|id| i32::try_from(id).ok())
            .ok_or_else(|| Error::Transfer("engine process identity unavailable".into()))?;
        Ok(Self {
            child,
            group: Pid::from_raw(pid),
            stopped: false,
            retained: vec![],
        })
    }
    pub(super) fn retain<T: Send + Sync + 'static>(&mut self, owner: Arc<T>) {
        self.retained.push(owner);
    }
    pub(super) async fn stop(&mut self, timeout: Duration) -> Result<()> {
        self.signal(Signal::SIGTERM)?;
        let deadline = Instant::now() + timeout;
        loop {
            if self.child.try_wait()?.is_some() && !live_group(self.group)? {
                self.stopped = true;
                return Ok(());
            }
            if Instant::now() >= deadline {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        // Forced cleanup is not evidence of a clean cache flush: do not export after timeout.
        self.signal(Signal::SIGKILL)?;
        let _ = tokio::time::timeout(Duration::from_secs(2), self.child.wait()).await;
        Err(Error::Transfer(
            "engine process group did not stop before export deadline".into(),
        ))
    }
    fn signal(&self, signal: Signal) -> Result<()> {
        match killpg(self.group, signal) {
            Ok(()) | Err(Errno::ESRCH) => Ok(()),
            Err(_) => Err(Error::Transfer("engine process group signal failed".into())),
        }
    }
}
impl Drop for Process {
    fn drop(&mut self) {
        if !self.stopped {
            let _ = self.signal(Signal::SIGKILL);
        }
    }
}

/// Zombies cannot write caches. A surviving descendant still prevents publication even if
/// the direct child has exited. Failure to inspect a process fails closed.
pub(super) fn live_group(group: Pid) -> Result<bool> {
    for entry in fs::read_dir("/proc")? {
        let entry = entry?;
        if entry
            .file_name()
            .to_str()
            .and_then(|name| name.parse::<u32>().ok())
            .is_none()
        {
            continue;
        }
        let stat = match fs::read_to_string(entry.path().join("stat")) {
            Ok(stat) => stat,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => return Err(error.into()),
        };
        let fields = stat
            .rsplit_once(')')
            .map(|(_, rest)| rest.split_whitespace().collect::<Vec<_>>())
            .ok_or_else(|| Error::Transfer("engine process inspection failed".into()))?;
        let process_group = fields
            .get(2)
            .and_then(|value| value.parse::<i32>().ok())
            .ok_or_else(|| Error::Transfer("engine process inspection failed".into()))?;
        if process_group == group.as_raw()
            && fields
                .first()
                .is_some_and(|state| !matches!(*state, "Z" | "X"))
        {
            return Ok(true);
        }
    }
    Ok(false)
}

#[cfg(test)]
mod tests;
