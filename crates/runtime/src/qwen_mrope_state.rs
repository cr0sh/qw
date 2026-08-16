// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Single-sequence MRoPE position state retained by the dense text model.

use std::cell::{Cell, RefCell};

use mlxcel_core::{MlxArray, UniquePtr};

/// MRoPE state for a single sequence.
///
/// `position_ids` is `[3, 1, prefill_len]` and is populated only by the
/// vision-language prefill path (when image/video tokens are present).
/// `rope_deltas` is the scalar that decode steps add to `cache_offset`
/// to recover the absolute MRoPE position; it is non-zero when the row
/// went through a multimodal prefill, and zero (or absent) for text-only
/// rows.
pub(crate) struct MRopeEntry {
    pub position_ids: Option<UniquePtr<MlxArray>>,
    pub rope_deltas: Option<i32>,
}

impl MRopeEntry {
    pub(crate) fn empty() -> Self {
        Self {
            position_ids: None,
            rope_deltas: None,
        }
    }
}

/// Single-sequence fallback state. Dense text generation leaves this empty,
/// but retaining it preserves the position-state path used by Qwen3.5.
pub(crate) struct MRopeState {
    fallback: RefCell<MRopeEntry>,
    position: Cell<i32>,
}

impl MRopeState {
    pub(crate) fn new() -> Self {
        Self {
            fallback: RefCell::new(MRopeEntry::empty()),
            position: Cell::new(0),
        }
    }

    pub(crate) fn clear(&self) {
        let mut entry = self.fallback.borrow_mut();
        entry.position_ids = None;
        entry.rope_deltas = None;
        self.position.set(0);
    }

    pub(crate) fn set_position(&self, position: i32) {
        self.position.set(position);
    }

    pub(crate) fn position(&self) -> i32 {
        self.position.get()
    }

    pub(crate) fn rope_delta(&self) -> Option<i32> {
        self.fallback.borrow().rope_deltas
    }

    pub(crate) fn with_position_ids<R>(
        &self,
        f: impl FnOnce(Option<&MlxArray>) -> R,
    ) -> R {
        let entry = self.fallback.borrow();
        f(entry.position_ids.as_deref())
    }

    pub(crate) fn restore(
        &self,
        position: i32,
        position_ids: Option<&MlxArray>,
        rope_delta: Option<i32>,
    ) {
        let mut entry = self.fallback.borrow_mut();
        entry.position_ids = position_ids.map(mlxcel_core::copy);
        entry.rope_deltas = rope_delta;
        self.position.set(position);
    }
}

impl Default for MRopeState {
    fn default() -> Self {
        Self::new()
    }
}

