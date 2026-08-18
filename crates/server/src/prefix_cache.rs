use qw_runtime::PromptSnapshot;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SnapshotRoute {
    Baseline,
    Mtp,
}

impl SnapshotRoute {
    fn matches(self, snapshot: &PromptSnapshot) -> bool {
        matches!(
            (self, snapshot),
            (Self::Baseline, PromptSnapshot::Baseline(_)) | (Self::Mtp, PromptSnapshot::Mtp(_))
        )
    }
}

pub struct PrefixMatch<'a> {
    pub token_count: usize,
    pub snapshot: &'a PromptSnapshot,
}

struct Entry {
    tokens: Vec<i32>,
    snapshots: Vec<PromptSnapshot>,
    route: SnapshotRoute,
    last_used: u64,
}

pub struct PrefixCache {
    max_tokens: usize,
    total_tokens: usize,
    clock: u64,
    entries: Vec<Entry>,
}

impl PrefixCache {
    pub fn new(max_tokens: usize) -> Self {
        assert!(max_tokens > 0, "prefix cache capacity must be nonzero");
        Self {
            max_tokens,
            total_tokens: 0,
            clock: 0,
            entries: Vec::new(),
        }
    }

    pub fn lookup(
        &mut self,
        prompt: &[i32],
        route: SnapshotRoute,
    ) -> Option<PrefixMatch<'_>> {
        let (index, snapshot_index) = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| entry.route == route)
            .filter_map(|(entry_index, entry)| {
                let common = entry
                    .tokens
                    .iter()
                    .zip(prompt)
                    .take_while(|(cached, requested)| cached == requested)
                    .count();
                entry
                    .snapshots
                    .iter()
                    .enumerate()
                    .filter(|(_, snapshot)| snapshot.token_len() <= common)
                    .max_by_key(|(_, snapshot)| snapshot.token_len())
                    .map(|(snapshot_index, snapshot)| {
                        (entry_index, snapshot_index, snapshot.token_len())
                    })
            })
            .max_by_key(|(_, _, token_count)| *token_count)
            .map(|(entry_index, snapshot_index, _)| (entry_index, snapshot_index))?;
        self.clock = self.clock.wrapping_add(1);
        self.entries[index].last_used = self.clock;
        let snapshot = &self.entries[index].snapshots[snapshot_index];
        Some(PrefixMatch {
            token_count: snapshot.token_len(),
            snapshot,
        })
    }

    pub fn insert(&mut self, tokens: Vec<i32>, mut snapshots: Vec<PromptSnapshot>) {
        let Some(first) = snapshots.first() else {
            return;
        };
        if tokens.len() > self.max_tokens {
            return;
        }
        let route = match first {
            PromptSnapshot::Baseline(_) => SnapshotRoute::Baseline,
            PromptSnapshot::Mtp(_) => SnapshotRoute::Mtp,
        };
        assert!(
            snapshots.iter().all(|snapshot| route.matches(snapshot)
                && snapshot.token_len() > 0
                && snapshot.token_len() <= tokens.len()),
            "prefix cache snapshots must match their route and prompt"
        );
        if let Some(index) = self
            .entries
            .iter()
            .position(|entry| entry.tokens == tokens && entry.route == route)
        {
            self.total_tokens -= self.entries[index].tokens.len();
            let mut previous = self.entries.swap_remove(index).snapshots;
            previous.retain(|old| {
                snapshots
                    .iter()
                    .all(|new| new.token_len() != old.token_len())
            });
            snapshots.extend(previous);
            snapshots.sort_by_key(PromptSnapshot::token_len);
        }
        self.clock = self.clock.wrapping_add(1);
        self.total_tokens += tokens.len();
        self.entries.push(Entry {
            tokens,
            snapshots,
            route,
            last_used: self.clock,
        });
        while self.total_tokens > self.max_tokens {
            let lru = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(index, _)| index)
                .expect("over-budget prefix cache must contain an entry");
            self.total_tokens -= self.entries[lru].tokens.len();
            self.entries.swap_remove(lru);
        }
    }

    #[cfg(test)]
    pub fn total_tokens(&self) -> usize {
        self.total_tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot(tokens: usize) -> PromptSnapshot {
        PromptSnapshot::Baseline(mlxcel_core::generate::ModelStateSnapshot::new(
            "test", tokens,
        ))
    }

    fn snapshots(lengths: impl IntoIterator<Item = usize>) -> Vec<PromptSnapshot> {
        lengths.into_iter().map(snapshot).collect()
    }

    fn lookup<'a>(cache: &'a mut PrefixCache, prompt: &[i32]) -> Option<PrefixMatch<'a>> {
        cache.lookup(prompt, SnapshotRoute::Baseline)
    }

    #[test]
    fn longest_restorable_common_prefix_uses_internal_snapshot() {
        let mut cache = PrefixCache::new(8);
        cache.insert(vec![1, 2, 3, 4, 5], snapshots(1..=5));

        let hit = lookup(&mut cache, &[1, 2, 3, 9]).expect("internal prefix hit");
        assert_eq!(hit.token_count, 3);
        assert_eq!(hit.snapshot.token_len(), 3);

        cache.insert(vec![1, 2, 3, 4, 5], snapshots([5]));
        let hit = lookup(&mut cache, &[1, 2, 3, 8]).expect("preserved internal prefix hit");
        assert_eq!(hit.token_count, 3);
    }

    #[test]
    fn longest_exact_prefix_wins_and_refreshes_lru() {
        let mut cache = PrefixCache::new(8);
        cache.insert(vec![1, 2], snapshots([2]));
        cache.insert(vec![1, 2, 3], snapshots([3]));
        let hit = lookup(&mut cache, &[1, 2, 3, 4]).expect("prefix hit");
        assert_eq!(hit.token_count, 3);
    }

    #[test]
    fn replacement_and_token_budget_evict_lru_entries() {
        let mut cache = PrefixCache::new(5);
        cache.insert(vec![1, 2], snapshots([2]));
        cache.insert(vec![3, 4], snapshots([2]));
        lookup(&mut cache, &[1, 2, 9]).expect("refresh first entry");
        cache.insert(vec![5, 6], snapshots([2]));
        assert!(lookup(&mut cache, &[3, 4, 9]).is_none());
        assert!(lookup(&mut cache, &[1, 2, 9]).is_some());
        assert_eq!(cache.total_tokens(), 4);

        cache.insert(vec![1, 2], snapshots([2]));
        assert_eq!(cache.total_tokens(), 4);
    }

    #[test]
    fn sequential_turns_preserve_full_cached_token_and_snapshot_lengths() {
        let mut cache = PrefixCache::new(32);
        for turn_len in [1_usize, 3, 5, 7, 9] {
            let tokens = (0..turn_len as i32).collect::<Vec<_>>();
            cache.insert(tokens.clone(), snapshots([turn_len]));

            let mut next_turn = tokens;
            next_turn.extend([100, 101]);
            let hit = lookup(&mut cache, &next_turn).expect("latest turn prefix");
            assert_eq!(hit.token_count, turn_len);
            let PromptSnapshot::Baseline(snapshot) = hit.snapshot else {
                panic!("test inserts baseline snapshots");
            };
            assert_eq!(snapshot.token_len(), turn_len);
        }
    }

    #[test]
    fn unrelated_histories_and_branches_coexist_and_select_longest_prefix() {
        let mut cache = PrefixCache::new(64);
        cache.insert(vec![1, 2], snapshots([2]));
        cache.insert(vec![9, 8, 7], snapshots([3]));
        cache.insert(vec![1, 2, 3, 4], snapshots([4]));
        cache.insert(vec![1, 2, 5], snapshots([3]));

        assert_eq!(lookup(&mut cache, &[9, 8, 7, 6]).expect("unrelated history").token_count, 3);
        assert_eq!(lookup(&mut cache, &[1, 2, 3, 4, 6]).expect("first branch").token_count, 4);
        assert_eq!(lookup(&mut cache, &[1, 2, 5, 6]).expect("second branch").token_count, 3);
        assert_eq!(lookup(&mut cache, &[1, 2, 6]).expect("common ancestor").token_count, 2);
    }

    #[test]
    fn global_token_budget_evicts_lru_across_unrelated_histories() {
        let mut cache = PrefixCache::new(9);
        cache.insert(vec![1, 2, 3], snapshots([3]));
        cache.insert(vec![4, 5, 6], snapshots([3]));
        cache.insert(vec![7, 8, 9], snapshots([3]));
        lookup(&mut cache, &[1, 2, 3, 0]).expect("refresh first history");
        cache.insert(vec![10, 11, 12], snapshots([3]));

        assert!(lookup(&mut cache, &[4, 5, 6, 0]).is_none());
        assert!(lookup(&mut cache, &[1, 2, 3, 0]).is_some());
        assert!(lookup(&mut cache, &[7, 8, 9, 0]).is_some());
        assert!(lookup(&mut cache, &[10, 11, 12, 0]).is_some());
        assert_eq!(cache.total_tokens(), 9);
    }

    #[test]
    fn over_budget_prompts_are_not_inserted() {
        let mut cache = PrefixCache::new(2);
        cache.insert(vec![1, 2, 3], snapshots([3]));
        assert!(lookup(&mut cache, &[1, 2, 3, 4]).is_none());
        assert_eq!(cache.total_tokens(), 0);
    }
}
