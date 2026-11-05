// SPDX-License-Identifier: Apache-2.0
//
// Adapted from https://github.com/Prefrontal-AI/latent-sparse-attention @5ebea73
// src/lsa/core/reference.py (read, _read_row, expand_buckets).
//
//! Centroid scoring, top-p bucket pick and expansion to one read list per query
//! (`docs/dlsa.md` §4.5).

use std::cmp::Ordering;

use anyhow::{Result, ensure};
use mlxcel_core::{MlxArray, UniquePtr, dtype};

use super::cache::DlsaBucketCache;
use super::config::DlsaIndexConfig;
use super::profile::{DlsaStage, lap};
use super::project::to_host_f32;

/// The read lists of `rows` queries, one list per query shared by all heads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DlsaSelection {
    pub rows: usize,
    /// `top_k + window_size + sink_size`.
    pub width: usize,
    pub top_p: usize,
    /// `[rows, width]` i32 key positions: the expansion (picked buckets best first, members
    /// ascending, cut at `top_k`), then the window ascending, then the sinks ascending; `-1` padded.
    pub lists: Vec<i32>,
    /// `[rows]` valid entries of each list.
    pub lengths: Vec<i32>,
    /// `[rows, top_p]` i32 picked bucket rows, best first, `-1` padded.
    pub picked: Vec<i32>,
    /// `[rows]` buckets each row could score.
    pub visible: Vec<i32>,
}

impl DlsaSelection {
    /// The valid entries of row `r`'s list.
    pub fn list(&self, r: usize) -> &[i32] {
        &self.lists[r * self.width..r * self.width + self.lengths[r] as usize]
    }

    /// The lists as an MLX `[rows, width]` i32 array (`-1` padded).
    pub fn lists_mlx(&self) -> UniquePtr<MlxArray> {
        mlxcel_core::from_slice_i32(&self.lists, &[self.rows as i32, self.width as i32])
    }

    /// The lengths as an MLX `[rows]` i32 array.
    pub fn lengths_mlx(&self) -> UniquePtr<MlxArray> {
        mlxcel_core::from_slice_i32(&self.lengths, &[self.rows as i32])
    }
}

/// Best bucket first: score descending, an equal score to the lower bucket (the reference's
/// stable descending argsort; the kernels' packed `(ordered_bits(score), 0x7FFFFFFF - bucket)`
/// key). Compares with `partial_cmp`, so `-0.0 == 0.0` as in torch.
fn rank(scores: &[f32], a: u32, b: u32) -> Ordering {
    scores[b as usize]
        .partial_cmp(&scores[a as usize])
        .expect("bucket scores are checked for NaN")
        .then(a.cmp(&b))
}

/// The `top_p` best of `scores` (the visible buckets' scores), best first. Errors on a NaN.
pub fn top_p_buckets(scores: &[f32], top_p: usize) -> Result<Vec<u32>> {
    ensure!(!scores.iter().any(|s| s.is_nan()), "a bucket score is NaN");
    let mut order: Vec<u32> = (0..scores.len() as u32).collect();
    if order.len() > top_p {
        if top_p == 0 {
            return Ok(Vec::new());
        }
        order.select_nth_unstable_by(top_p - 1, |&a, &b| rank(scores, a, b));
        order.truncate(top_p);
    }
    // A total order (ties broken by bucket), so an unstable sort is exact
    order.sort_unstable_by(|&a, &b| rank(scores, a, b));
    Ok(order)
}

/// Append the read list of a query at `position` to `out`: members of `picked` (best first,
/// each bucket's members ascending) with `sink_size <= m < window_start`, at most `top_k` of
/// them, then the window `[window_start, position]`, then the sinks
/// `[0, min(sink_size, window_start, position + 1))`. `members` is `[rows, bucket_cap]`.
pub fn expand_buckets(
    picked: &[u32],
    members: &[i32],
    position: usize,
    cfg: &DlsaIndexConfig,
    out: &mut Vec<i32>,
) {
    let cap = cfg.bucket_cap;
    let window_start = (position + 1).saturating_sub(cfg.window_size);
    let (lo, hi) = (cfg.sink_size as i64, window_start as i64);
    let mut taken = 0;
    'walk: for &b in picked {
        for &m in &members[b as usize * cap..(b as usize + 1) * cap] {
            let m64 = m as i64;
            if m64 >= lo && m64 < hi {
                if taken == cfg.top_k {
                    break 'walk;
                }
                out.push(m);
                taken += 1;
            }
        }
    }
    out.extend((window_start..=position).map(|p| p as i32));
    let sinks = cfg.sink_size.min(window_start).min(position + 1);
    out.extend((0..sinks).map(|p| p as i32));
}

/// Scores `[R, visible]` of `queries` (`[R, d]`) against the first `visible` centroids, as
/// sequential f32 dots (the host oracle).
pub fn score_buckets_host(
    queries: &[f32],
    centroids: &[f32],
    d: usize,
    visible: usize,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(queries.len() / d * visible);
    for q in queries.chunks_exact(d) {
        for c in centroids[..visible * d].chunks_exact(d) {
            let mut s = 0.0f32;
            for (x, y) in q.iter().zip(c) {
                s += x * y;
            }
            out.push(s);
        }
    }
    out
}

/// Row work at which the host pick fans out over threads.
const PARALLEL_WORK: usize = 1 << 16;

impl DlsaBucketCache {
    fn check_positions(&self, positions: &[usize]) -> Result<()> {
        for &p in positions {
            ensure!(
                p < self.len(),
                "a query at position {p} needs its own key in the cache, which holds {} keys",
                self.len()
            );
        }
        Ok(())
    }

    /// Most buckets any of `positions` may score.
    pub fn max_visible(&self, positions: &[usize]) -> usize {
        positions
            .iter()
            .map(|&p| self.visible_buckets(p))
            .max()
            .unwrap_or(0)
    }

    /// Score `index_queries` (`[R, d]` MLX, unit rows or any norm) against the centroids the
    /// rows may see, with one MLX f32 matmul. Returns `[R, stride]` f32 with
    /// `stride = max_visible(positions)`.
    pub fn score_mlx(
        &mut self,
        index_queries: &MlxArray,
        positions: &[usize],
    ) -> Result<(Vec<f32>, usize)> {
        let Some((scores, stride)) = self.score_device(index_queries, positions)? else {
            return Ok((Vec::new(), 0));
        };
        let host = to_host_f32(&scores);
        lap(DlsaStage::ScoresReadback, &[]);
        Ok((host, stride))
    }

    /// [`score_mlx`](Self::score_mlx) without the readback: the lazy `[R, stride]` f32 scores
    /// (the same op for every backend, so their scores are identical), `None` when no row can
    /// see a bucket.
    pub(super) fn score_device(
        &mut self,
        index_queries: &MlxArray,
        positions: &[usize],
    ) -> Result<Option<(UniquePtr<MlxArray>, usize)>> {
        let shape = mlxcel_core::array_shape(index_queries);
        let d = self.config().index_dim;
        ensure!(
            shape.len() == 2 && shape[0] as usize == positions.len() && shape[1] as usize == d,
            "index queries must be [{}, {d}], got {shape:?}",
            positions.len()
        );
        self.check_positions(positions)?;
        let stride = self.max_visible(positions);
        let Some(centroids) = self.device_centroids(stride) else {
            return Ok(None);
        };
        lap(DlsaStage::CentroidSync, &[&centroids]);
        let q = mlxcel_core::astype(index_queries, dtype::FLOAT32);
        let scores = mlxcel_core::matmul(&q, &mlxcel_core::transpose(&centroids));
        lap(DlsaStage::ScoreMatmul, &[&scores]);
        Ok(Some((scores, stride)))
    }

    /// Select the read lists of rows at `positions` from `index_queries` (`[R, d]` MLX):
    /// MLX scoring, then the host pick and expansion. Every position must be `< len()`: the
    /// cache must already hold the forward's keys (append first, then select).
    pub fn select(
        &mut self,
        index_queries: &MlxArray,
        positions: &[usize],
    ) -> Result<DlsaSelection> {
        let (scores, stride) = self.score_mlx(index_queries, positions)?;
        let sel = self.select_from_scores(&scores, stride, positions);
        lap(DlsaStage::HostPick, &[]);
        sel
    }

    /// [`select`](Self::select) on the host: `index_queries` `[R, d]` f32, scored by
    /// sequential f32 dots. The oracle of the MLX path.
    pub fn select_host(&self, index_queries: &[f32], positions: &[usize]) -> Result<DlsaSelection> {
        let d = self.config().index_dim;
        ensure!(
            index_queries.len() == positions.len() * d,
            "index queries must be [{}, {d}]",
            positions.len()
        );
        self.check_positions(positions)?;
        let stride = self.max_visible(positions);
        let scores = score_buckets_host(index_queries, self.centroids(), d, stride);
        self.select_from_scores(&scores, stride, positions)
    }

    /// Pick and expand from precomputed scores `[R, stride]` (row `r` uses its first
    /// `visible_buckets(positions[r])` columns).
    pub fn select_from_scores(
        &self,
        scores: &[f32],
        stride: usize,
        positions: &[usize],
    ) -> Result<DlsaSelection> {
        self.check_positions(positions)?;
        ensure!(
            scores.len() == positions.len() * stride && stride >= self.max_visible(positions),
            "scores must be [{}, >= {}]",
            positions.len(),
            self.max_visible(positions)
        );
        let cfg = *self.config();
        let (rows, width, top_p) = (positions.len(), cfg.list_width(), cfg.top_p);
        let mut sel = DlsaSelection {
            rows,
            width,
            top_p,
            lists: vec![-1; rows * width],
            lengths: vec![0; rows],
            picked: vec![-1; rows * top_p],
            visible: vec![0; rows],
        };
        // Only plain data crosses into the worker threads (the cache holds MLX arrays)
        let members = self.members();
        let total_rows = self.rows();
        let one_row = |r: usize,
                       list: &mut [i32],
                       length: &mut i32,
                       picked: &mut [i32],
                       visible: &mut i32,
                       buf: &mut Vec<i32>|
         -> Result<()> {
            let p = positions[r];
            let v = cfg.visible_buckets(p).min(total_rows);
            let order = top_p_buckets(&scores[r * stride..r * stride + v], top_p)?;
            for (slot, &b) in picked.iter_mut().zip(&order) {
                *slot = b as i32;
            }
            buf.clear();
            expand_buckets(&order, members, p, &cfg, buf);
            debug_assert!(buf.len() <= width);
            list[..buf.len()].copy_from_slice(buf);
            *length = buf.len() as i32;
            *visible = v as i32;
            Ok(())
        };
        let work = rows * stride.max(1);
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
        if work < PARALLEL_WORK || threads == 1 || rows == 1 {
            let mut buf = Vec::with_capacity(width);
            for r in 0..rows {
                one_row(
                    r,
                    &mut sel.lists[r * width..(r + 1) * width],
                    &mut sel.lengths[r],
                    &mut sel.picked[r * top_p..(r + 1) * top_p],
                    &mut sel.visible[r],
                    &mut buf,
                )?;
            }
        } else {
            let per = rows.div_ceil(threads);
            let results: Vec<Result<()>> = std::thread::scope(|s| {
                let handles: Vec<_> = sel
                    .lists
                    .chunks_mut(per * width)
                    .zip(sel.lengths.chunks_mut(per))
                    .zip(sel.picked.chunks_mut(per * top_p))
                    .zip(sel.visible.chunks_mut(per))
                    .enumerate()
                    .map(|(c, (((lists, lengths), picked), visible))| {
                        let one_row = &one_row;
                        s.spawn(move || -> Result<()> {
                            let mut buf = Vec::with_capacity(width);
                            for i in 0..lengths.len() {
                                one_row(
                                    c * per + i,
                                    &mut lists[i * width..(i + 1) * width],
                                    &mut lengths[i],
                                    &mut picked[i * top_p..(i + 1) * top_p],
                                    &mut visible[i],
                                    &mut buf,
                                )?;
                            }
                            Ok(())
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|h| h.join().expect("a selection thread panicked"))
                    .collect()
            });
            results.into_iter().collect::<Result<()>>()?;
        }
        Ok(sel)
    }
}
