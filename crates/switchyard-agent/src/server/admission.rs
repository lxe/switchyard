//! Admission ownership lasts through the complete proxied response body.
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use tokio::sync::{OwnedRwLockReadGuard, OwnedRwLockWriteGuard, RwLock};

#[derive(Clone, Default)]
pub(super) struct Admission {
    closed: Arc<AtomicBool>,
    active: Arc<RwLock<()>>,
}
impl Admission {
    pub(super) async fn enter(&self) -> Option<OwnedRwLockReadGuard<()>> {
        if self.closed.load(Ordering::SeqCst) {
            return None;
        }
        let guard = self.active.clone().read_owned().await;
        if self.closed.load(Ordering::SeqCst) {
            return None;
        }
        Some(guard)
    }
    pub(super) fn close(&self) {
        self.closed.store(true, Ordering::SeqCst);
    }
    pub(super) async fn drain(&self) -> OwnedRwLockWriteGuard<()> {
        self.active.clone().write_owned().await
    }
}
