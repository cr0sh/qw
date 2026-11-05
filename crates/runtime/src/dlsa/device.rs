// SPDX-License-Identifier: Apache-2.0
//
//! Device-side state of one bucket cache (`docs/dlsa.md` §14.1): the mirror of the bucket tables
//! the scoring matmul and the K1 select read, index keys appended on the device and not yet read
//! back (deferred decode keys), and a lazy count of the keys the Metal read visits.
//!
//! The host tables stay authoritative: the mirror is synced from them (rows appended since the
//! last sync are written with `slice_update` into preallocated arrays, which grow geometrically),
//! a trim only lowers the synced row count, and a reset or restore starts it over.

use mlxcel_core::{MlxArray, UniquePtr, dtype};

/// Rows the mirror allocates at least (Qwen: 32 rows a block, so 8 blocks).
const MIN_CAPACITY: usize = 256;

/// The device copy of a cache's bucket tables: f32 centroids `[capacity, d]` and i32 members
/// `[capacity, bucket_cap]`, of which the first `rows` mirror the host tables.
#[derive(Default)]
pub(crate) struct DeviceMirror {
    tables: Option<(UniquePtr<MlxArray>, UniquePtr<MlxArray>)>,
    capacity: usize,
    rows: usize,
}

impl DeviceMirror {
    /// Rows that mirror the host tables.
    #[cfg(test)]
    pub(crate) fn rows(&self) -> usize {
        self.rows
    }

    /// Rows allocated.
    #[cfg(test)]
    pub(crate) fn capacity(&self) -> usize {
        self.capacity
    }

    /// Forget every row past `rows` (after a trim: a reopened block closes again with other
    /// rows under the same row numbers, which the next sync writes over).
    pub(crate) fn truncate(&mut self, rows: usize) {
        self.rows = self.rows.min(rows);
    }

    /// Release the arrays.
    pub(crate) fn clear(&mut self) {
        *self = Self::default();
    }

    /// Bring the mirror up to the host tables (`centroids` `[total, d]`, `members`
    /// `[total, cap]`), uploading only the rows past the synced ones, then evaluate it.
    pub(crate) fn sync(&mut self, centroids: &[f32], members: &[i32], d: usize, cap: usize) {
        let total = members.len() / cap;
        debug_assert_eq!(centroids.len(), total * d);
        if total == self.rows && self.tables.is_some() {
            return;
        }
        if total > self.capacity || self.tables.is_none() {
            let capacity = total
                .max(2 * self.capacity)
                .max(MIN_CAPACITY)
                .next_power_of_two();
            let mut c = mlxcel_core::zeros(&[capacity as i32, d as i32], dtype::FLOAT32);
            let mut m = mlxcel_core::full_f32(&[capacity as i32, cap as i32], -1.0, dtype::INT32);
            if let Some((old_c, old_m)) = self.tables.take()
                && self.rows > 0
            {
                // Carry the synced rows over on the device
                let r = self.rows as i32;
                let kept_c = mlxcel_core::slice(&old_c, &[0, 0], &[r, d as i32]);
                let kept_m = mlxcel_core::slice(&old_m, &[0, 0], &[r, cap as i32]);
                c = mlxcel_core::slice_update(&c, &kept_c, &[0, 0], &[r, d as i32]);
                m = mlxcel_core::slice_update(&m, &kept_m, &[0, 0], &[r, cap as i32]);
            }
            self.tables = Some((c, m));
            self.capacity = capacity;
        }
        let (c, m) = self.tables.take().expect("allocated above");
        let (from, to) = (self.rows as i32, total as i32);
        let (c, m) = if from < to {
            let new_c = mlxcel_core::from_slice_f32(
                &centroids[self.rows * d..total * d],
                &[to - from, d as i32],
            );
            let new_m = mlxcel_core::from_slice_i32(
                &members[self.rows * cap..total * cap],
                &[to - from, cap as i32],
            );
            (
                mlxcel_core::slice_update(&c, &new_c, &[from, 0], &[to, d as i32]),
                mlxcel_core::slice_update(&m, &new_m, &[from, 0], &[to, cap as i32]),
            )
        } else {
            (c, m)
        };
        mlxcel_core::eval(&c);
        mlxcel_core::eval(&m);
        self.tables = Some((c, m));
        self.rows = total;
    }

    /// The first `rows` synced centroids `[rows, d]` f32 (a view of the mirror).
    pub(crate) fn centroids(&self, rows: usize) -> Option<UniquePtr<MlxArray>> {
        let (c, _) = self.tables.as_ref()?;
        assert!(
            rows <= self.rows,
            "the mirror holds {} rows, not {rows}",
            self.rows
        );
        let d = mlxcel_core::array_shape(c)[1];
        Some(mlxcel_core::slice(c, &[0, 0], &[rows as i32, d]))
    }

    /// The whole member table `[capacity, cap]` i32 (rows past [`rows`](Self::rows) are stale
    /// or `-1`; the K1 select reads only the first `total_rows`).
    pub(crate) fn members(&self) -> Option<&MlxArray> {
        self.tables.as_ref().map(|(_, m)| &**m)
    }

    /// Bytes allocated: (centroids, members).
    pub(crate) fn nbytes(&self, d: usize, cap: usize) -> (usize, usize) {
        if self.tables.is_none() {
            return (0, 0);
        }
        (self.capacity * d * 4, self.capacity * cap * 4)
    }
}

/// Index keys appended on the device and not yet read back: `[rows, d]` f32, the last `rows`
/// positions of the cache, all in its open block.
#[derive(Default)]
pub(crate) struct DeferredKeys {
    keys: Option<UniquePtr<MlxArray>>,
    rows: usize,
}

impl DeferredKeys {
    pub(crate) fn rows(&self) -> usize {
        self.rows
    }

    pub(crate) fn push(&mut self, keys: &MlxArray, rows: usize) {
        let f32_keys = mlxcel_core::astype(keys, dtype::FLOAT32);
        self.keys = Some(match self.keys.take() {
            Some(prev) => mlxcel_core::concatenate(&prev, &f32_keys, 0),
            None => f32_keys,
        });
        self.rows += rows;
    }

    /// The pending keys, emptied.
    pub(crate) fn take(&mut self) -> Option<(UniquePtr<MlxArray>, usize)> {
        let rows = std::mem::take(&mut self.rows);
        self.keys.take().map(|k| (k, rows))
    }

    /// The pending keys, kept.
    pub(crate) fn get(&self) -> Option<&MlxArray> {
        self.keys.as_deref()
    }

    pub(crate) fn clear(&mut self) {
        *self = Self::default();
    }
}

/// The lazy sum of the list lengths of the Metal reads since the last resolve (an i64 scalar
/// on the device), so counting the keys read adds no host sync.
#[derive(Default)]
pub(crate) struct DeviceCount {
    sum: Option<UniquePtr<MlxArray>>,
}

impl DeviceCount {
    /// Add the sum of `lengths` (`[R]` i32), lazily.
    pub(crate) fn add(&mut self, lengths: &MlxArray) {
        let part = mlxcel_core::sum_all(&mlxcel_core::astype(lengths, dtype::INT64));
        self.sum = Some(match self.sum.take() {
            Some(prev) => mlxcel_core::add(&prev, &part),
            None => part,
        });
    }

    /// The count so far (evaluating it: call where the caller already syncs), reset to 0.
    pub(crate) fn take(&mut self) -> u64 {
        self.sum
            .take()
            .map_or(0, |s| mlxcel_core::item_i64(&s) as u64)
    }

    pub(crate) fn clear(&mut self) {
        self.sum = None;
    }
}

#[cfg(test)]
#[path = "device_tests.rs"]
mod tests;
