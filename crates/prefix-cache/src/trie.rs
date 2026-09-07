use qw_runtime::PromptSnapshot;

use crate::{EntryKey, ResponseResumeMetadata, SnapshotRoute};

pub(crate) struct Terminal {
    pub route: SnapshotRoute,
    pub observations: u64,
    pub reuse_count: u64,
    pub last_access_unix_ms: u64,
    pub expires_at_unix_ms: u64,
    pub serialized_bytes: u64,
    pub snapshot: Option<PromptSnapshot>,
    // Materialized identity survives hot/persistent eviction; structural-only nodes have none.
    pub entry_id: Option<EntryKey>,
    pub persistent_key: Option<EntryKey>,
    pub response_resume: Option<ResponseResumeMetadata>,
    pub page_refs: Vec<(u64, usize)>,
    pub local_bytes: usize,
    pub blob_refs: Vec<(String, u64)>,
}
impl Terminal {
    pub fn structural(route: SnapshotRoute, observations: u64, now: u64, expires_at: u64) -> Self {
        Self {
            route,
            observations,
            reuse_count: 0,
            last_access_unix_ms: now,
            expires_at_unix_ms: expires_at,
            serialized_bytes: 0,
            snapshot: None,
            entry_id: None,
            persistent_key: None,
            response_resume: None,
            page_refs: Vec::new(),
            local_bytes: 0,
            blob_refs: Vec::new(),
        }
    }
}

struct Node {
    fragment: Vec<i32>,
    parent: Option<usize>,
    children: Vec<usize>,
    terminals: Vec<Terminal>,
}

pub(crate) struct RadixTrie {
    nodes: Vec<Option<Node>>,
}

impl RadixTrie {
    pub fn new() -> Self {
        Self {
            nodes: vec![Some(Node {
                fragment: Vec::new(),
                parent: None,
                children: Vec::new(),
                terminals: Vec::new(),
            })],
        }
    }

    pub fn observe(
        &mut self,
        tokens: &[i32],
        route: SnapshotRoute,
        now: u64,
        expires_at: u64,
    ) -> usize {
        assert!(!tokens.is_empty());
        let mut node_id = 0;
        let mut consumed = 0;
        loop {
            if let Some(terminal) = self.terminal_mut(node_id, route) {
                terminal.observations = terminal.observations.saturating_add(1);
                // Matching an ancestor is an observation, not use of its materialized state.
                if terminal.snapshot.is_none() && terminal.persistent_key.is_none() {
                    terminal.last_access_unix_ms = now;
                    terminal.expires_at_unix_ms = expires_at;
                }
            }
            if consumed == tokens.len() {
                if self.terminal(node_id, route).is_none() {
                    self.node_mut(node_id)
                        .terminals
                        .push(Terminal::structural(route, 1, now, expires_at));
                }
                return node_id;
            }
            let next = tokens[consumed];
            let child = self
                .node(node_id)
                .children
                .iter()
                .copied()
                .find(|child| self.node(*child).fragment[0] == next);
            let Some(child_id) = child else {
                let id = self.push_node(Node {
                    fragment: tokens[consumed..].to_vec(),
                    parent: Some(node_id),
                    children: Vec::new(),
                    terminals: vec![Terminal::structural(route, 1, now, expires_at)],
                });
                self.node_mut(node_id).children.push(id);
                return id;
            };
            let fragment = &self.node(child_id).fragment;
            let common = fragment
                .iter()
                .zip(&tokens[consumed..])
                .take_while(|(a, b)| a == b)
                .count();
            if common == fragment.len() {
                consumed += common;
                node_id = child_id;
                continue;
            }
            let internal_id = self.split_child(node_id, child_id, common);
            consumed += common;
            self.node_mut(internal_id)
                .terminals
                .push(Terminal::structural(route, 2, now, expires_at));
            if consumed == tokens.len() {
                return internal_id;
            }
            let leaf = self.push_node(Node {
                fragment: tokens[consumed..].to_vec(),
                parent: Some(internal_id),
                children: Vec::new(),
                terminals: vec![Terminal::structural(route, 1, now, expires_at)],
            });
            self.node_mut(internal_id).children.push(leaf);
            return leaf;
        }
    }

    pub fn ensure(&mut self, tokens: &[i32], route: SnapshotRoute, terminal: Terminal) -> usize {
        let id = self.ensure_node(tokens);
        if let Some(existing) = self.terminal_mut(id, route) {
            *existing = terminal;
        } else {
            self.node_mut(id).terminals.push(terminal);
        }
        id
    }

    pub fn ensure_node(&mut self, tokens: &[i32]) -> usize {
        assert!(!tokens.is_empty());
        let mut node_id = 0;
        let mut consumed = 0;
        loop {
            if consumed == tokens.len() {
                return node_id;
            }
            let next = tokens[consumed];
            let child = self
                .node(node_id)
                .children
                .iter()
                .copied()
                .find(|child| self.node(*child).fragment[0] == next);
            let Some(child_id) = child else {
                let id = self.push_node(Node {
                    fragment: tokens[consumed..].to_vec(),
                    parent: Some(node_id),
                    children: Vec::new(),
                    terminals: Vec::new(),
                });
                self.node_mut(node_id).children.push(id);
                return id;
            };
            let fragment = &self.node(child_id).fragment;
            let common = fragment
                .iter()
                .zip(&tokens[consumed..])
                .take_while(|(a, b)| a == b)
                .count();
            if common == fragment.len() {
                consumed += common;
                node_id = child_id;
            } else {
                let internal = self.split_child(node_id, child_id, common);
                consumed += common;
                if consumed == tokens.len() {
                    return internal;
                }
                let leaf = self.push_node(Node {
                    fragment: tokens[consumed..].to_vec(),
                    parent: Some(internal),
                    children: Vec::new(),
                    terminals: Vec::new(),
                });
                self.node_mut(internal).children.push(leaf);
                return leaf;
            }
        }
    }

    pub fn path(&self, prompt: &[i32], route: SnapshotRoute) -> Vec<(usize, usize)> {
        let mut output = Vec::new();
        let mut node_id = 0;
        let mut consumed = 0;
        loop {
            if self.terminal(node_id, route).is_some() {
                output.push((node_id, consumed));
            }
            if consumed == prompt.len() {
                break;
            }
            let Some(child) = self
                .node(node_id)
                .children
                .iter()
                .copied()
                .find(|child| self.node(*child).fragment.first() == prompt.get(consumed))
            else {
                break;
            };
            let fragment = &self.node(child).fragment;
            if prompt.get(consumed..consumed + fragment.len()) != Some(fragment.as_slice()) {
                break;
            }
            consumed += fragment.len();
            node_id = child;
        }
        output
    }

    pub fn terminal(&self, node: usize, route: SnapshotRoute) -> Option<&Terminal> {
        self.node(node)
            .terminals
            .iter()
            .find(|terminal| terminal.route == route)
    }

    pub fn terminal_mut(&mut self, node: usize, route: SnapshotRoute) -> Option<&mut Terminal> {
        self.node_mut(node)
            .terminals
            .iter_mut()
            .find(|terminal| terminal.route == route)
    }

    pub fn remove_terminal(&mut self, node_id: usize, route: SnapshotRoute) -> Option<Terminal> {
        let index = self
            .node(node_id)
            .terminals
            .iter()
            .position(|terminal| terminal.route == route)?;
        let removed = self.node_mut(node_id).terminals.swap_remove(index);
        self.prune(node_id);
        Some(removed)
    }

    pub fn terminal_ids(&self) -> Vec<(usize, SnapshotRoute)> {
        self.nodes
            .iter()
            .enumerate()
            .filter_map(|(id, node)| node.as_ref().map(|node| (id, node)))
            .flat_map(|(id, node)| {
                node.terminals
                    .iter()
                    .map(move |terminal| (id, terminal.route))
            })
            .collect()
    }

    fn split_child(&mut self, parent_id: usize, child_id: usize, common: usize) -> usize {
        assert!(common > 0 && common < self.node(child_id).fragment.len());
        let prefix = self.node(child_id).fragment[..common].to_vec();
        self.node_mut(child_id).fragment.drain(..common);
        let internal = self.push_node(Node {
            fragment: prefix,
            parent: Some(parent_id),
            children: vec![child_id],
            terminals: Vec::new(),
        });
        self.node_mut(child_id).parent = Some(internal);
        let position = self
            .node(parent_id)
            .children
            .iter()
            .position(|id| *id == child_id)
            .expect("child belongs to parent");
        self.node_mut(parent_id).children[position] = internal;
        internal
    }

    fn prune(&mut self, mut node_id: usize) {
        while node_id != 0 {
            let (parent, empty, only_child) = {
                let node = self.node(node_id);
                (
                    node.parent.expect("non-root node has parent"),
                    node.terminals.is_empty() && node.children.is_empty(),
                    (node.terminals.is_empty() && node.children.len() == 1)
                        .then(|| node.children[0]),
                )
            };
            if empty {
                self.node_mut(parent)
                    .children
                    .retain(|child| *child != node_id);
                self.nodes[node_id] = None;
                node_id = parent;
            } else if let Some(child) = only_child {
                let fragment = self.node(node_id).fragment.clone();
                let child_fragment = self.node(child).fragment.clone();
                self.node_mut(child).fragment = [fragment, child_fragment].concat();
                self.node_mut(child).parent = Some(parent);
                let position = self
                    .node(parent)
                    .children
                    .iter()
                    .position(|id| *id == node_id)
                    .expect("node belongs to parent");
                self.node_mut(parent).children[position] = child;
                self.nodes[node_id] = None;
                node_id = parent;
            } else {
                break;
            }
        }
    }

    fn push_node(&mut self, node: Node) -> usize {
        let id = self.nodes.len();
        self.nodes.push(Some(node));
        id
    }
    fn node(&self, id: usize) -> &Node {
        self.nodes[id].as_ref().expect("live trie node")
    }
    fn node_mut(&mut self, id: usize) -> &mut Node {
        self.nodes[id].as_mut().expect("live trie node")
    }
}
