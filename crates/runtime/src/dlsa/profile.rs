// SPDX-License-Identifier: Apache-2.0
//
//! A stage profile of the D-LSA layer forward, for benchmarks (`docs/dlsa.md` §14.3).
//!
//! Off by default. `QW_DLSA_PROFILE=1` fences every full-attention D-LSA forward:
//! each stage ends in an `eval` and its wall time is added to one process-wide [`DlsaProfile`]
//! (read with [`snapshot`], cleared with [`reset`]). `QW_DLSA_PROFILE_FROM=<offset>` fences
//! only forwards whose first position is at or past `offset`, so one pass gives the unfenced
//! throughput of the early chunks and a fenced breakdown of the last ones. A benchmark can
//! override the environment with [`set_override`].
//!
//! With fences off, every hook is one atomic load and a thread-local read: no `eval`, no
//! timer, no extra MLX op, so the forward's graph and outputs are unchanged.

use std::cell::Cell;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use mlxcel_core::MlxArray;

/// A stage of the D-LSA forward of one full-attention layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DlsaStage {
    /// The input of the layer forward (whatever was still lazy before it).
    Input,
    /// The q/k/v/gate projections and q/k norms (dense code).
    Projection,
    /// Index-key projection (MLX).
    IndexKeys,
    /// Index-key readback to the host (`append_mlx`, minus Ward).
    KeyReadback,
    /// Ward clustering of the blocks the forward closed (host).
    Ward,
    /// Index-query projection (MLX).
    IndexQueries,
    /// Syncing the device centroid copy (upload of new rows, concatenate).
    CentroidSync,
    /// The centroid score matmul (MLX f32).
    ScoreMatmul,
    /// Readback of the `[L, V]` f32 scores.
    ScoresReadback,
    /// Host top-p pick and expansion.
    HostPick,
    /// K1 device select (top-p, expansion, window and sinks on the GPU).
    DeviceSelect,
    /// RoPE of q/k (dense code).
    Rotate,
    /// KV cache append and fetch (sparse forwards).
    KvFetch,
    /// Decode: sorting and uploading the read list.
    ListUpload,
    /// Decode: the `take` of the listed K/V rows.
    Gather,
    /// Prefill: building the additive list masks.
    MaskBuild,
    /// The sparse read (SDPA over the gathered keys or under the masks).
    Read,
    /// A dense-key-set forward's dense attention (KV append and fetch included).
    DenseRead,
    /// The gate and `o_proj`.
    Finish,
}

impl DlsaStage {
    pub const COUNT: usize = 19;
    pub const ALL: [DlsaStage; Self::COUNT] = [
        Self::Input,
        Self::Projection,
        Self::IndexKeys,
        Self::KeyReadback,
        Self::Ward,
        Self::IndexQueries,
        Self::CentroidSync,
        Self::ScoreMatmul,
        Self::ScoresReadback,
        Self::HostPick,
        Self::DeviceSelect,
        Self::Rotate,
        Self::KvFetch,
        Self::ListUpload,
        Self::Gather,
        Self::MaskBuild,
        Self::Read,
        Self::DenseRead,
        Self::Finish,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::Input => "input",
            Self::Projection => "qkv_projection",
            Self::IndexKeys => "index_key_projection",
            Self::KeyReadback => "key_readback",
            Self::Ward => "ward",
            Self::IndexQueries => "index_query_projection",
            Self::CentroidSync => "centroid_sync",
            Self::ScoreMatmul => "score_matmul",
            Self::ScoresReadback => "scores_readback",
            Self::HostPick => "host_pick",
            Self::DeviceSelect => "device_select",
            Self::Rotate => "rotate",
            Self::KvFetch => "kv_fetch",
            Self::ListUpload => "list_upload",
            Self::Gather => "gather",
            Self::MaskBuild => "mask_build",
            Self::Read => "read",
            Self::DenseRead => "dense_read",
            Self::Finish => "finish",
        }
    }
}

/// Total wall time and number of occurrences of one stage (or layer kind).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DlsaStageTime {
    pub time: Duration,
    pub count: u64,
}

impl DlsaStageTime {
    const ZERO: Self = Self {
        time: Duration::ZERO,
        count: 0,
    };

    fn add(&mut self, d: Duration) {
        self.time += d;
        self.count += 1;
    }
}

/// What the fenced forwards spent, by stage, plus layer totals a benchmark records.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DlsaProfile {
    /// By [`DlsaStage`] (index `stage as usize`).
    pub stages: [DlsaStageTime; DlsaStage::COUNT],
    /// Fenced D-LSA layer forwards (one `count` each; `time` is their stage sum).
    pub layer_forwards: DlsaStageTime,
    /// Linear-attention decoder layers, as a benchmark's fenced forward times them.
    pub linear_layers: DlsaStageTime,
    /// Full-attention decoder layers (dense or D-LSA), as a benchmark times them.
    pub full_attention_layers: DlsaStageTime,
    /// Whole fenced model forwards (embedding to logits), as a benchmark times them.
    pub forwards: DlsaStageTime,
}

impl Default for DlsaProfile {
    fn default() -> Self {
        Self::new()
    }
}

impl DlsaProfile {
    pub const fn new() -> Self {
        Self {
            stages: [DlsaStageTime::ZERO; DlsaStage::COUNT],
            layer_forwards: DlsaStageTime::ZERO,
            linear_layers: DlsaStageTime::ZERO,
            full_attention_layers: DlsaStageTime::ZERO,
            forwards: DlsaStageTime::ZERO,
        }
    }

    pub fn stage(&self, stage: DlsaStage) -> DlsaStageTime {
        self.stages[stage as usize]
    }

    /// Stages with at least one occurrence, in forward order.
    pub fn entries(&self) -> Vec<(&'static str, DlsaStageTime)> {
        DlsaStage::ALL
            .iter()
            .map(|&s| (s.name(), self.stage(s)))
            .filter(|(_, t)| t.count > 0)
            .collect()
    }
}

static PROFILE: Mutex<DlsaProfile> = Mutex::new(DlsaProfile::new());

const FOLLOW_ENV: u8 = 0;
const FORCE_ON: u8 = 1;
const FORCE_OFF: u8 = 2;
static OVERRIDE: AtomicU8 = AtomicU8::new(FOLLOW_ENV);

thread_local! {
    /// The end of the last stage of the forward being fenced on this thread, if any.
    static LAP: Cell<Option<Instant>> = const { Cell::new(None) };
}

/// `QW_DLSA_PROFILE` (unset, empty or `0` is off) and `QW_DLSA_PROFILE_FROM`, read once.
fn env_from() -> Option<usize> {
    static FROM: OnceLock<Option<usize>> = OnceLock::new();
    *FROM.get_or_init(|| {
        let on = std::env::var("QW_DLSA_PROFILE").is_ok_and(|v| !v.is_empty() && v != "0");
        on.then(|| {
            std::env::var("QW_DLSA_PROFILE_FROM")
                .ok()
                .and_then(|v| v.trim().parse().ok())
                .unwrap_or(0)
        })
    })
}

/// Force fences on (`Some(true)`) or off (`Some(false)`) for every forward, or follow the
/// environment again (`None`).
pub fn set_override(fences: Option<bool>) {
    let v = match fences {
        None => FOLLOW_ENV,
        Some(true) => FORCE_ON,
        Some(false) => FORCE_OFF,
    };
    OVERRIDE.store(v, Ordering::Relaxed);
}

/// Whether a forward whose first position is `offset` is fenced.
pub fn fences_at(offset: usize) -> bool {
    match OVERRIDE.load(Ordering::Relaxed) {
        FORCE_ON => true,
        FORCE_OFF => false,
        _ => env_from().is_some_and(|from| offset >= from),
    }
}

/// Whether a fenced forward is in progress on this thread.
pub fn active() -> bool {
    LAP.with(|l| l.get().is_some())
}

/// Start fencing a forward at `offset` if [`fences_at`] says so: evaluates `inputs` (timed as
/// [`DlsaStage::Input`]) and starts the lap timer. Returns whether it is fenced.
pub fn begin(offset: usize, inputs: &[&MlxArray]) -> bool {
    if !fences_at(offset) {
        return false;
    }
    let started = Instant::now();
    LAP.with(|l| l.set(Some(started)));
    lap(DlsaStage::Input, inputs);
    true
}

/// End the fenced forward on this thread (if any), counting it.
pub fn end() {
    if LAP.with(|l| l.take()).is_none() {
        return;
    }
    let mut p = PROFILE.lock().unwrap_or_else(|e| e.into_inner());
    p.layer_forwards.count += 1;
}

fn eval_all(arrays: &[&MlxArray]) {
    for a in arrays {
        mlxcel_core::eval(a);
    }
}

/// Close `stage` of the fenced forward: evaluate `outputs` and add the time since the last
/// lap. A no-op when no forward is fenced on this thread.
pub fn lap(stage: DlsaStage, outputs: &[&MlxArray]) {
    let Some(last) = LAP.with(|l| l.get()) else {
        return;
    };
    eval_all(outputs);
    let now = Instant::now();
    LAP.with(|l| l.set(Some(now)));
    let d = now - last;
    let mut p = PROFILE.lock().unwrap_or_else(|e| e.into_inner());
    p.stages[stage as usize].add(d);
    p.layer_forwards.time += d;
}

/// [`lap`] whose time splits into `part` of `part_stage` (measured inside it, e.g. Ward from
/// the cache's counters) and the rest, which goes to `stage`.
pub fn lap_split(stage: DlsaStage, part_stage: DlsaStage, part: Duration, outputs: &[&MlxArray]) {
    let Some(last) = LAP.with(|l| l.get()) else {
        return;
    };
    eval_all(outputs);
    let now = Instant::now();
    LAP.with(|l| l.set(Some(now)));
    let d = now - last;
    let part = part.min(d);
    let mut p = PROFILE.lock().unwrap_or_else(|e| e.into_inner());
    p.stages[stage as usize].add(d - part);
    p.stages[part_stage as usize].add(part);
    p.layer_forwards.time += d;
}

/// Record a decoder layer a benchmark timed (its output evaluated).
pub fn record_layer(full_attention: bool, time: Duration) {
    let mut p = PROFILE.lock().unwrap_or_else(|e| e.into_inner());
    if full_attention {
        p.full_attention_layers.add(time);
    } else {
        p.linear_layers.add(time);
    }
}

/// Record a whole model forward a benchmark timed.
pub fn record_forward(time: Duration) {
    PROFILE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .forwards
        .add(time);
}

/// The profile so far.
pub fn snapshot() -> DlsaProfile {
    PROFILE.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

/// Clear the profile.
pub fn reset() {
    *PROFILE.lock().unwrap_or_else(|e| e.into_inner()) = DlsaProfile::new();
}

#[cfg(test)]
mod tests {
    use super::*;

    // One test touches the process-wide state, so parallel tests cannot interleave.
    #[test]
    fn laps_only_inside_a_fenced_forward() {
        set_override(Some(false));
        assert!(!begin(0, &[]));
        assert!(!active());
        let before = snapshot();
        lap(DlsaStage::Read, &[]);
        end();
        assert_eq!(snapshot(), before, "an unfenced forward records nothing");

        set_override(Some(true));
        assert!(begin(7, &[]));
        assert!(active());
        lap(DlsaStage::IndexKeys, &[]);
        lap_split(
            DlsaStage::KeyReadback,
            DlsaStage::Ward,
            Duration::from_secs(3600),
            &[],
        );
        end();
        assert!(!active());
        set_override(None);
        let p = snapshot();
        assert!(p.layer_forwards.count > before.layer_forwards.count);
        assert!(p.stage(DlsaStage::IndexKeys).count >= 1);
        assert!(p.stage(DlsaStage::Ward).count >= 1);
        // The split never exceeds the lap
        assert!(p.stage(DlsaStage::Ward).time <= p.layer_forwards.time);
        assert!(p.entries().iter().all(|(_, t)| t.count > 0));
    }
}
