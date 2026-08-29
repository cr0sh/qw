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

//! Single-sequence rotary-position state.

use std::cell::{Cell, RefCell};

use mlxcel_core::{MlxArray, UniquePtr};

/// Optional explicit rotary positions for prefill and the scalar decode offset.
pub(crate) struct RopeEntry {
    pub position_ids: Option<UniquePtr<MlxArray>>,
    pub rope_deltas: Option<i32>,
}

impl RopeEntry {
    pub(crate) fn empty() -> Self {
        Self {
            position_ids: None,
            rope_deltas: None,
        }
    }
}

/// Position state for one active Qwen4 sequence.
pub(crate) struct RopeState {
    fallback: RefCell<RopeEntry>,
    pending: RefCell<Option<RopeEntry>>,
    position: Cell<i32>,
}

impl RopeState {
    pub(crate) fn new() -> Self {
        Self {
            fallback: RefCell::new(RopeEntry::empty()),
            pending: RefCell::new(None),
            position: Cell::new(0),
        }
    }

    pub(crate) fn clear(&self) {
        let mut entry = self.fallback.borrow_mut();
        entry.position_ids = None;
        entry.rope_deltas = None;
        self.position.set(0);
    }

    pub(crate) fn prepare(&self, position_ids: &MlxArray, rope_delta: i32) {
        *self.pending.borrow_mut() = Some(RopeEntry {
            position_ids: Some(mlxcel_core::copy(position_ids)),
            rope_deltas: Some(rope_delta),
        });
    }

    pub(crate) fn clear_prepared(&self) {
        *self.pending.borrow_mut() = None;
    }

    pub(crate) fn activate_prepared(&self) -> Result<(), String> {
        let entry = self
            .pending
            .borrow_mut()
            .take()
            .ok_or_else(|| "embedding prefill is missing prepared MRoPE state".to_string())?;
        *self.fallback.borrow_mut() = entry;
        self.position.set(0);
        Ok(())
    }

    pub(crate) fn finish_prefill(&self) {
        self.fallback.borrow_mut().position_ids = None;
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

    pub(crate) fn with_position_ids<R>(&self, f: impl FnOnce(Option<&MlxArray>) -> R) -> R {
        let entry = self.fallback.borrow();
        f(entry.position_ids.as_deref())
    }

    pub(crate) fn restore(
        &self,
        position: i32,
        position_ids: Option<&MlxArray>,
        rope_delta: Option<i32>,
    ) {
        *self.pending.borrow_mut() = None;
        let mut entry = self.fallback.borrow_mut();
        entry.position_ids = position_ids.map(mlxcel_core::copy);
        entry.rope_deltas = rope_delta;
        self.position.set(position);
    }
}

impl Default for RopeState {
    fn default() -> Self {
        Self::new()
    }
}
