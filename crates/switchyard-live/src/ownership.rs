//! Reconcile allocations whose POST response may have been lost.
use crate::{Error, Result, runtime::PendingLaunch};
use std::{
    collections::BTreeSet,
    time::{Duration, Instant},
};
use switchyard_provider::{Provider, storage::Storage};

pub async fn reconcile<P: Provider>(
    provider: &P,
    storage: &Storage,
    launches: &mut [PendingLaunch],
    confirmed: &BTreeSet<String>,
) -> Result<()> {
    for launch in launches.iter_mut() {
        if confirmed.contains(&launch.name) {
            launch.resolved = true;
        }
    }
    let until = Instant::now() + Duration::from_secs(30);
    while launches.iter().any(|l| !l.resolved) {
        let pods = provider.list().await?;
        for launch in launches.iter_mut().filter(|l| !l.resolved) {
            let matching = pods
                .iter()
                .filter(|p| p.name == launch.name)
                .collect::<Vec<_>>();
            if matching.is_empty() {
                continue;
            }
            let mut confirmed = true;
            for pod in matching {
                provider.terminate(&pod.id).await?;
                if provider
                    .get(&pod.id)
                    .await?
                    .is_some_and(|p| p.status != "TERMINATED")
                {
                    confirmed = false;
                }
            }
            launch.resolved = confirmed;
        }
        if launches.iter().any(|l| !l.resolved) {
            if Instant::now() >= until {
                return Err(Error::Request("allocation outcome is unresolved; retain the journal and bootstrap ticket and run recovery again".into()));
            }
            tokio::time::sleep(Duration::from_secs(2)).await;
        }
    }
    for launch in launches {
        storage.delete(&launch.ticket_key).await?;
    }
    Ok(())
}
