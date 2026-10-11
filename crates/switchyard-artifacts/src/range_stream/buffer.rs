//! Private RAM/spool reservations and reader-held reclamation.
use super::*;
use aligned_vec::{AVec, ConstAlign};
use std::{
    fs::File,
    os::{fd::AsRawFd, unix::fs::OpenOptionsExt},
};
pub(super) struct Backing {
    pub(super) file: File,
    path: Option<PathBuf>,
    location: Location,
    bytes: u64,
    id: usize,
    shared: std::sync::Weak<Shared>,
}
impl Drop for Backing {
    fn drop(&mut self) {
        if let Some(path) = &self.path {
            let _ = fs::remove_file(path);
        }
        if let Some(s) = self.shared.upgrade() {
            {
                let mut st = s.state.lock().unwrap();
                match self.location {
                    Location::Ram => st.stats.ram_bytes -= self.bytes,
                    Location::Spool => st.stats.spool_bytes -= self.bytes,
                };
                st.stats.reclaimed_bytes += self.bytes;
            }
            s.wake.notify_waiters();
            s.emit(Some(self.id), Some(Lifecycle::Reclaimed));
        }
    }
}
/// The descriptor remains owned until explicit completion and all cloned leases drop.
/// A consumer must close its descriptor before sending the completion acknowledgement.
#[derive(Clone)]
pub struct ReadyRange {
    pub spec: RangeSpec,
    pub(super) backing: Arc<Backing>,
}
impl ReadyRange {
    pub fn location(&self) -> &Location {
        &self.backing.location
    }
    pub fn descriptor_path(&self) -> String {
        format!(
            "/proc/{}/fd/{}",
            std::process::id(),
            self.backing.file.as_raw_fd()
        )
    }
    pub fn copying(&self) {
        if let Some(s) = self.backing.shared.upgrade() {
            s.emit(Some(self.spec.id), Some(Lifecycle::Copying));
        }
    }
    pub fn consumed(self) {
        if let Some(s) = self.backing.shared.upgrade() {
            s.emit(Some(self.spec.id), Some(Lifecycle::Consumed));
        }
    }
    pub fn read_into(&self, buffer: &mut [u8]) -> Result<()> {
        if buffer.len() != self.spec.length as usize {
            return Err(Error::Invalid("range read bound".into()));
        }
        let file = self.backing.file.try_clone()?;
        // Spool readers use the same aligned O_DIRECT contract, never a buffered reopen.
        let mut aligned = AVec::<u8, ConstAlign<ALIGN>>::new(ALIGN);
        aligned.resize(self.spec.padded_length as usize, 0);
        use std::os::unix::fs::FileExt;
        file.read_exact_at(&mut aligned, 0)?;
        buffer.copy_from_slice(&aligned[..buffer.len()]);
        Ok(())
    }
}

impl RangeStream {
    pub(super) async fn reserve(&self, spec: RangeSpec) -> Result<ReadyRange> {
        let began = Instant::now();
        let location = loop {
            let wake = self.shared.wake.notified();
            tokio::pin!(wake);
            wake.as_mut().enable();
            self.shared.cancelled()?;
            let location = {
                let mut st = self.shared.state.lock().unwrap();
                let bytes = spec.padded_length;
                if st.stats.ram_bytes + bytes <= self.shared.limits.ram_bytes {
                    st.stats.ram_bytes += bytes;
                    st.stats.peak_ram_bytes = st.stats.peak_ram_bytes.max(st.stats.ram_bytes);
                    Some(Location::Ram)
                } else if st.stats.spool_bytes + bytes <= self.shared.limits.spool_bytes {
                    st.stats.spool_bytes += bytes;
                    st.stats.peak_spool_bytes = st.stats.peak_spool_bytes.max(st.stats.spool_bytes);
                    Some(Location::Spool)
                } else {
                    None
                }
            };
            if let Some(loc) = location {
                break loc;
            }
            wake.await;
        };
        self.shared
            .state
            .lock()
            .unwrap()
            .stats
            .exhausted_buffer_seconds += began.elapsed().as_secs_f64();
        let (file, path) = match location {
            Location::Ram => {
                let fd = nix::sys::memfd::memfd_create(
                    c"weight-chunk",
                    nix::sys::memfd::MFdFlags::MFD_CLOEXEC,
                )
                .map_err(std::io::Error::from)?;
                (File::from(fd), None)
            }
            Location::Spool => {
                let path = self.shared.directory.join(format!("{}.chunk", spec.id));
                let file = File::options()
                    .read(true)
                    .write(true)
                    .create_new(true)
                    .mode(0o600)
                    .custom_flags(nix::libc::O_DIRECT)
                    .open(&path)?;
                (file, Some(path))
            }
        };
        file.set_len(spec.padded_length)?;
        if matches!(location, Location::Spool) {
            nix::fcntl::fallocate(
                &file,
                nix::fcntl::FallocateFlags::empty(),
                0,
                spec.padded_length as i64,
            )
            .map_err(std::io::Error::from)?;
        }
        let range = ReadyRange {
            backing: Arc::new(Backing {
                file,
                path,
                location,
                bytes: spec.padded_length,
                id: spec.id,
                shared: Arc::downgrade(&self.shared),
            }),
            spec,
        };
        self.shared
            .emit(Some(range.spec.id), Some(Lifecycle::Reserved));
        Ok(range)
    }
}
