//! Continuous batching. Every step feeds each running sequence's next token
//! and fills the remaining token budget with prompt chunks, so new requests
//! start while others decode. Steps without prompt tokens replay a CUDA graph
//! captured for the padded batch size.

use crate::gpu::{Gpu, Graph};
use crate::model::{Act, BLOCK_SIZE, Config, Meta, Model, State};
use crate::{Error, Result};
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::mpsc;
use std::time::Instant;
use tokio::sync::mpsc::UnboundedSender;

#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub model: PathBuf,
    pub device: i32,
    /// Most sequences decoding at once.
    pub max_seqs: usize,
    /// Most tokens in one step (decode rows plus prompt chunks).
    pub max_batch_tokens: usize,
    /// Longest prompt plus output, in tokens.
    pub max_context: usize,
    /// Share of device memory the engine may hold.
    pub memory_fraction: f64,
    pub graphs: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Finish {
    Stop,
    Length,
}

#[derive(Debug)]
pub enum Event {
    Token(u32),
    Done {
        finish: Finish,
        prompt_tokens: usize,
        completion_tokens: usize,
    },
    Error(String),
}

pub struct Request {
    pub prompt: Vec<u32>,
    pub max_new: usize,
    pub temperature: f32,
    pub seed: u64,
    pub tx: UnboundedSender<Event>,
}

#[derive(Debug, Clone)]
pub struct Info {
    pub device: String,
    pub cfg: Config,
    pub max_seqs: usize,
    pub kv_blocks: usize,
    pub max_context: usize,
}

/// One teacher-forced step for accuracy checks: the model's pick and its top
/// log-probabilities before the forced token is fed.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Scored {
    pub argmax: u32,
    pub top: Vec<(u32, f32)>,
}

struct Seq {
    req: Request,
    slot: usize,
    blocks: Vec<u32>,
    /// Tokens of prompt ++ generated whose state is computed.
    fed: usize,
    generated: Vec<u32>,
    forced: Option<Vec<u32>>,
    scored: Vec<Scored>,
    admitted: Instant,
}

impl Seq {
    fn token(&self, i: usize) -> u32 {
        let p = self.req.prompt.len();
        if i < p {
            self.req.prompt[i]
        } else {
            self.generated[i - p]
        }
    }
    fn total(&self) -> usize {
        self.req.prompt.len() + self.generated.len()
    }
}

const BUCKETS: [usize; 16] = [
    1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 160, 192, 224, 256,
];

#[derive(Default, Debug, Clone, serde::Serialize)]
pub struct Stats {
    pub steps: u64,
    pub graph_steps: u64,
    pub decode_tokens: u64,
    pub prompt_tokens: u64,
    pub step_seconds: f64,
}

pub struct Engine {
    gpu: Gpu,
    model: Model,
    state: State,
    act: Act,
    cfg: EngineConfig,
    graphs: HashMap<usize, Graph>,
    free_slots: Vec<usize>,
    free_blocks: Vec<u32>,
    waiting: VecDeque<Seq>,
    running: Vec<Seq>,
    pub stats: Stats,
}

impl Engine {
    pub fn new(cfg: EngineConfig) -> Result<Self> {
        let gpu = Gpu::new(cfg.device)?;
        let model = Model::load(&gpu, &cfg.model)?;
        let mc = model.cfg.clone();
        let max_blocks = cfg.max_context.div_ceil(BLOCK_SIZE);
        let meta = Meta {
            max_tokens: cfg.max_batch_tokens.max(cfg.max_seqs),
            max_seqs: cfg.max_seqs,
            max_blocks,
        };
        let act = Act::new(&gpu, &mc, meta)?;
        // One extra slot absorbs the padding rows of graph replays.
        let slots = cfg.max_seqs + 1;
        let (free, total) = gpu.mem_info()?;
        let reserve = (total as f64 * (1.0 - cfg.memory_fraction)) as usize;
        let usable = free
            .saturating_sub(reserve)
            .saturating_sub(slots * mc.slot_bytes());
        let blocks = (usable / mc.block_bytes()).min(cfg.max_seqs * max_blocks);
        if blocks < max_blocks {
            return Err(Error::Config(format!(
                "not enough memory for one full context: {blocks} KV blocks, {max_blocks} needed"
            )));
        }
        let state = State::new(&gpu, &mc, slots, blocks)?;
        gpu.sync()?;
        let mut e = Self {
            gpu,
            model,
            state,
            act,
            graphs: HashMap::new(),
            free_slots: (0..cfg.max_seqs).rev().collect(),
            free_blocks: (0..blocks as u32).rev().collect(),
            waiting: VecDeque::new(),
            running: Vec::new(),
            stats: Stats::default(),
            cfg,
        };
        if e.cfg.graphs {
            e.capture()?;
        }
        Ok(e)
    }

    pub fn info(&self) -> Info {
        Info {
            device: self.gpu.name.clone(),
            cfg: self.model.cfg.clone(),
            max_seqs: self.cfg.max_seqs,
            kv_blocks: self.state.blocks,
            max_context: self.cfg.max_context,
        }
    }

    fn scratch_slot(&self) -> usize {
        self.cfg.max_seqs
    }

    /// Writes padding rows [from, to) into the staged metadata: each a one-token
    /// sequence on the scratch slot that writes no KV.
    fn stage_padding(&mut self, from: usize, to: usize) {
        let me = self.act.meta;
        let scratch = self.scratch_slot() as i32;
        let m = self.act.staging.as_mut_slice::<i32>();
        for i in from..to {
            m[me.ids() + i] = 0;
            m[me.pos() + i] = 0;
            m[me.slot_map() + i] = -1;
            m[me.tok_seq() + i] = i as i32;
            m[me.cu() + i + 1] = i as i32 + 1;
            m[me.slots() + i] = scratch;
            m[me.fresh() + i] = 1;
            m[me.last() + i] = i as i32;
            m[me.temps() + i] = 0;
            m[me.seeds() + i] = 0;
            m[me.block_table() + i * me.max_blocks..me.block_table() + (i + 1) * me.max_blocks]
                .fill(0);
        }
    }

    fn capture(&mut self) -> Result<()> {
        let t0 = Instant::now();
        let max = self.cfg.max_seqs;
        for &b in BUCKETS.iter().filter(|&&b| b <= max) {
            self.act.staging.as_mut_slice::<i32>()[self.act.meta.cu()] = 0;
            self.stage_padding(0, b);
            self.act.upload_meta(&self.gpu)?;
            // A plain run first, so cuBLAS settles its kernels outside capture.
            self.model
                .forward(&self.gpu, &self.state, &self.act, b, b)?;
            self.gpu.sync()?;
            self.gpu.begin_capture()?;
            let r = self.model.forward(&self.gpu, &self.state, &self.act, b, b);
            let g = self.gpu.end_capture();
            r?;
            self.graphs.insert(b, g?);
        }
        self.gpu.sync()?;
        eprintln!(
            "captured {} decode graphs in {:.2}s",
            self.graphs.len(),
            t0.elapsed().as_secs_f64()
        );
        Ok(())
    }

    pub fn submit(&mut self, req: Request) {
        self.enqueue(req, None);
    }

    fn enqueue(&mut self, mut req: Request, forced: Option<Vec<u32>>) {
        let limit = self.cfg.max_context;
        if req.prompt.is_empty() || req.prompt.len() >= limit {
            let _ = req.tx.send(Event::Error(format!(
                "prompt of {} tokens does not fit the context of {limit}",
                req.prompt.len()
            )));
            return;
        }
        req.max_new = req.max_new.clamp(1, limit - req.prompt.len());
        if let Some(f) = &forced {
            req.max_new = f.len();
        }
        self.waiting.push_back(Seq {
            req,
            slot: 0,
            blocks: Vec::new(),
            fed: 0,
            generated: Vec::new(),
            forced,
            scored: Vec::new(),
            admitted: Instant::now(),
        });
    }

    pub fn idle(&self) -> bool {
        self.running.is_empty() && self.waiting.is_empty()
    }

    fn admit(&mut self) {
        while self.running.len() < self.cfg.max_seqs {
            let Some(seq) = self.waiting.front() else {
                break;
            };
            // Reserve KV for the whole request up front: no preemption needed.
            let need = (seq.req.prompt.len() + seq.req.max_new).div_ceil(BLOCK_SIZE);
            if self.free_slots.is_empty() || self.free_blocks.len() < need {
                break;
            }
            let mut seq = self.waiting.pop_front().unwrap();
            seq.slot = self.free_slots.pop().unwrap();
            seq.blocks = self.free_blocks.split_off(self.free_blocks.len() - need);
            seq.admitted = Instant::now();
            self.running.push(seq);
        }
    }

    fn release(&mut self, seq: Seq) -> Seq {
        self.free_slots.push(seq.slot);
        self.free_blocks.extend_from_slice(&seq.blocks);
        seq
    }

    /// Runs one step. Returns sequences that finished during it.
    fn step(&mut self) -> Result<Vec<Seq>> {
        // Drop requests whose clients left.
        let mut finished = Vec::new();
        let mut i = 0;
        while i < self.running.len() {
            if self.running[i].forced.is_none() && self.running[i].req.tx.is_closed() {
                let s = self.running.swap_remove(i);
                self.release(s);
            } else {
                i += 1;
            }
        }
        self.waiting
            .retain(|s| s.forced.is_some() || !s.req.tx.is_closed());
        self.admit();
        if self.running.is_empty() {
            return Ok(finished);
        }
        // Decode rows first, then prompt chunks up to the token budget.
        self.running
            .sort_by_key(|s| (s.fed < s.req.prompt.len(), s.admitted));
        let me = self.act.meta;
        let mut budget = me.max_tokens;
        let mut plan: Vec<(usize, usize)> = Vec::new(); // (running index, tokens)
        for (i, s) in self.running.iter().enumerate() {
            let n = (s.total() - s.fed).min(budget);
            if n == 0 {
                break;
            }
            budget -= n;
            plan.push((i, n));
        }
        let s_count = plan.len();
        let t_count: usize = plan.iter().map(|p| p.1).sum();
        let decode_only = t_count == s_count;
        let bucket = if decode_only && self.cfg.graphs {
            BUCKETS
                .iter()
                .copied()
                .find(|&b| b >= s_count && self.graphs.contains_key(&b))
        } else {
            None
        };
        {
            let m = self.act.staging.as_mut_slice::<i32>();
            let mut t = 0;
            m[me.cu()] = 0;
            for (si, &(ri, n)) in plan.iter().enumerate() {
                let s = &self.running[ri];
                for k in 0..n {
                    let p = s.fed + k;
                    m[me.ids() + t] = s.token(p) as i32;
                    m[me.pos() + t] = p as i32;
                    m[me.slot_map() + t] =
                        (s.blocks[p / BLOCK_SIZE] as usize * BLOCK_SIZE + p % BLOCK_SIZE) as i32;
                    m[me.tok_seq() + t] = si as i32;
                    t += 1;
                }
                m[me.cu() + si + 1] = t as i32;
                m[me.slots() + si] = s.slot as i32;
                m[me.fresh() + si] = (s.fed == 0) as i32;
                m[me.last() + si] = t as i32 - 1;
                m[me.temps() + si] = s.req.temperature.max(0.0).to_bits() as i32;
                let seed =
                    s.req.seed ^ (s.generated.len() as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
                m[me.seeds() + si] = (seed ^ (seed >> 32)) as i32;
                let row = me.block_table() + si * me.max_blocks;
                m[row..row + s.blocks.len()].copy_from_slice(bytemuck_u32(&s.blocks));
            }
        }
        let t0 = Instant::now();
        match bucket {
            Some(b) => {
                self.stage_padding(s_count, b);
                self.act.upload_meta(&self.gpu)?;
                self.gpu.replay(&self.graphs[&b])?;
                self.stats.graph_steps += 1;
            }
            None => {
                self.act.upload_meta(&self.gpu)?;
                self.model
                    .forward(&self.gpu, &self.state, &self.act, t_count, s_count)?;
            }
        }
        self.act.download_ids(&self.gpu, s_count)?;
        self.gpu.sync()?;
        self.stats.steps += 1;
        self.stats.step_seconds += t0.elapsed().as_secs_f64();
        let picked: Vec<u32> = self.act.ids_host.as_slice::<i32>()[..s_count]
            .iter()
            .map(|&x| x as u32)
            .collect();
        let vocab = self.model.cfg.vocab;
        let eos = self.model.cfg.eos.clone();
        let mut done = Vec::new();
        for (si, &(ri, n)) in plan.iter().enumerate() {
            let prompt_len = self.running[ri].req.prompt.len();
            if self.running[ri].fed >= prompt_len {
                self.stats.decode_tokens += n as u64;
            } else {
                self.stats.prompt_tokens += n as u64;
            }
            let s = &mut self.running[ri];
            s.fed += n;
            if s.fed < s.total() {
                continue; // mid-prompt chunk: its pick is not a token yet
            }
            let tok = match &s.forced {
                Some(f) => {
                    let mut row = self.gpu.pinned(vocab * 4)?;
                    self.gpu.dtoh(
                        &mut row,
                        self.act.logits.ptr + (si * vocab * 4) as u64,
                        vocab * 4,
                    )?;
                    self.gpu.sync()?;
                    s.scored.push(score(picked[si], row.as_slice::<f32>()));
                    f[s.generated.len()]
                }
                None => picked[si],
            };
            let stop = s.forced.is_none() && eos.contains(&tok);
            if !stop {
                s.generated.push(tok);
                let _ = s.req.tx.send(Event::Token(tok));
            }
            if stop || s.generated.len() >= s.req.max_new {
                let finish = if stop { Finish::Stop } else { Finish::Length };
                let _ = s.req.tx.send(Event::Done {
                    finish,
                    prompt_tokens: prompt_len,
                    completion_tokens: s.generated.len(),
                });
                done.push(ri);
            }
        }
        done.sort_unstable_by(|a, b| b.cmp(a));
        for ri in done {
            let s = self.running.swap_remove(ri);
            finished.push(self.release(s));
        }
        Ok(finished)
    }

    /// Teacher-forced scoring of several (prompt, continuation) pairs, batched
    /// together as a server would run them.
    pub fn score(&mut self, pairs: Vec<(Vec<u32>, Vec<u32>)>) -> Result<Vec<Vec<Scored>>> {
        let n = pairs.len();
        let mut txs = Vec::new();
        for (i, (prompt, forced)) in pairs.into_iter().enumerate() {
            let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
            drop(rx);
            txs.push(tx.clone());
            self.enqueue(
                Request {
                    prompt,
                    max_new: forced.len(),
                    temperature: 0.0,
                    seed: i as u64,
                    tx,
                },
                Some(forced),
            );
        }
        let mut out: Vec<Option<Vec<Scored>>> = vec![None; n];
        // Sequences carry no id; match them back by their channel.
        while !self.idle() {
            for s in self.step()? {
                let i = txs
                    .iter()
                    .position(|t| t.same_channel(&s.req.tx))
                    .expect("known channel");
                out[i] = Some(s.scored);
            }
        }
        Ok(out.into_iter().map(|o| o.unwrap_or_default()).collect())
    }
}

fn bytemuck_u32(v: &[u32]) -> &[i32] {
    // u32 block ids are far below i32::MAX.
    // A safe reinterpretation via iterator would allocate; this is a plain cast.
    #[allow(unsafe_code)]
    // SAFETY: u32 and i32 share size and alignment.
    unsafe {
        std::slice::from_raw_parts(v.as_ptr().cast(), v.len())
    }
}

fn score(argmax: u32, logits: &[f32]) -> Scored {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let lse = max
        + logits
            .iter()
            .map(|&x| ((x - max) as f64).exp())
            .sum::<f64>()
            .ln() as f32;
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.select_nth_unstable_by(5, |&a, &b| logits[b].total_cmp(&logits[a]));
    let mut top: Vec<(u32, f32)> = idx[..5]
        .iter()
        .map(|&i| (i as u32, logits[i] - lse))
        .collect();
    top.sort_by(|a, b| b.1.total_cmp(&a.1));
    Scored { argmax, top }
}

/// Runs the engine on its own thread. Requests arrive on the returned channel.
pub fn spawn(cfg: EngineConfig) -> Result<(mpsc::Sender<Request>, Info)> {
    let (tx, rx) = mpsc::channel::<Request>();
    let (ready_tx, ready_rx) = mpsc::channel();
    std::thread::Builder::new()
        .name("engine".into())
        .spawn(move || {
            let mut engine = match Engine::new(cfg) {
                Ok(e) => {
                    let _ = ready_tx.send(Ok(e.info()));
                    e
                }
                Err(e) => {
                    let _ = ready_tx.send(Err(e));
                    return;
                }
            };
            let mut last_report = Instant::now();
            loop {
                if engine.idle() {
                    match rx.recv() {
                        Ok(r) => engine.submit(r),
                        Err(_) => return,
                    }
                }
                while let Ok(r) = rx.try_recv() {
                    engine.submit(r);
                }
                if let Err(e) = engine.step() {
                    eprintln!("engine step failed: {e}");
                    for s in engine.running.drain(..).chain(engine.waiting.drain(..)) {
                        let _ = s.req.tx.send(Event::Error(e.to_string()));
                    }
                    return;
                }
                if last_report.elapsed().as_secs() >= 10 && engine.stats.steps > 0 {
                    let st = &engine.stats;
                    eprintln!(
                        "steps {} (graph {}), decode {} tok, prompt {} tok, {:.1} ms/step, running {}, waiting {}",
                        st.steps,
                        st.graph_steps,
                        st.decode_tokens,
                        st.prompt_tokens,
                        1e3 * st.step_seconds / st.steps as f64,
                        engine.running.len(),
                        engine.waiting.len()
                    );
                    last_report = Instant::now();
                }
            }
        })?;
    let info = ready_rx
        .recv()
        .map_err(|_| Error::Config("engine thread exited".into()))??;
    Ok((tx, info))
}
