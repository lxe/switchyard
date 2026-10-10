//! Validated admission and dispatch to an owned inference endpoint.
use super::{LiveRuntime, WorkCall};
use crate::{
    Result,
    serving::{Call, Event},
};
use std::collections::BTreeMap;
use switchyard_core::scheduler::{ObservedActive, ObservedRequest, PlannedRequest, RequestStatus};
use switchyard_provider::Provider;
use tokio::sync::mpsc;

pub(super) struct RouteContext<'a> {
    pub(super) now: f64,
    pub(super) calls: &'a [WorkCall],
    pub(super) requests: &'a BTreeMap<usize, PlannedRequest>,
    pub(super) states: &'a mut BTreeMap<usize, ObservedRequest>,
    pub(super) attempts: &'a mut [usize],
    pub(super) events: &'a mpsc::Sender<Event>,
}

impl<P: Provider> LiveRuntime<P> {
    pub(super) async fn route_request(
        &mut self,
        request: usize,
        node: usize,
        context: RouteContext<'_>,
    ) -> Result<()> {
        let RouteContext {
            now,
            calls,
            requests,
            states,
            attempts,
            events: tx,
        } = context;
        let (Some(r), Some(n)) = (requests.get(&request), self.nodes.get(node)) else {
            self.invalid += 1;
            return Ok(());
        };
        let profile = &self.profiles[n.observed.profile];
        let reserved = n
            .observed
            .active
            .iter()
            .map(|a| requests[&a.id].input_tokens + requests[&a.id].max_output)
            .sum::<u64>();
        if states[&request].state != RequestStatus::Queued
            || !n.observed.is_ready
            || n.owned.ended_s.is_some()
            || !profile.compatible(r)
            || n.observed.active.len() >= profile.max_in_flight
            || reserved + r.input_tokens + r.max_output > profile.kv_tokens
            || r.depends_on
                .iter()
                .any(|d| states[d].state != RequestStatus::Done)
        {
            self.invalid += 1;
            return Ok(());
        }
        let work = calls[request].clone();
        if !self.nodes[node].checked_tokenizer {
            self.nodes[node]
                .endpoint
                .verify_tokenization(&work.model, &work.messages, r.input_tokens)
                .await?;
            self.nodes[node].checked_tokenizer = true;
        }
        let input_tokens = r.input_tokens;
        let endpoint = self.nodes[node].endpoint.clone();
        let planner = self.planners[&work.model].clone();
        let tx = tx.clone();
        let epoch = self.epoch;
        self.nodes[node].observed.active.push(ObservedActive {
            id: request,
            prefill: profile.prefill_work(r),
            generated: 0.0,
            available_s: now,
        });
        self.nodes[node].observed.last_used_s = now;
        states.get_mut(&request).expect("known request").state = RequestStatus::Active;
        attempts[request] += 1;
        self.tasks.spawn(async move {
            let result = endpoint
                .complete(
                    Call {
                        id: request,
                        model: work.model,
                        messages: work.messages,
                        max_output: work.max_output,
                        input_tokens,
                        planner,
                        epoch,
                    },
                    tx.clone(),
                )
                .await;
            let _ = tx.send(Event::Completed { request, result }).await;
        });
        Ok(())
    }
}
