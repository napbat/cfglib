//! Where the arms of a branch rejoin when another arm diverges.
//!
//! The post-dominator tree joins every exit, a `throw` as much as a
//! `return`, so a branch with one arm that throws has no merge point even
//! when its other arms meet again. Those arms rejoin where they would if the
//! diverging arm were not there: at the immediate post-dominator of the
//! branch in the code that runs toward the end of the walk, with every block
//! that cannot reach that end left out.
//!
//! A walk with a stop ends there. A walk without one, such as the body of a
//! function outside any `try`, ends wherever its code leaves, and nothing in
//! the graph tells the exit after the join from the `throw` beside it. What
//! tells them apart is which arms reach them: the end of such a walk is the
//! set of exits that two or more arms of the branch reach, and an exit that
//! one arm alone reaches diverges from the others.

extern crate alloc;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use super::bound::Bound;
use super::{Structurer, flow_successors};
use crate::block::BlockId;
use crate::graph::dominator::DominatorTree;
use crate::graph::view::GraphView;

/// The immediate post-dominators of the code that runs up to one stop:
/// [`Structurer::rejoin`]'s cache.
///
/// The arms of nested branches stop at the same block (an `else if` chain
/// that rejoins before a `throw` in its last arm stops each link at the one
/// join), so the tree for a stop answers every branch inside its code. The
/// end of a walk without a stop depends on the arms of the branch, so that
/// tree answers its own branch only and is never kept.
#[derive(Default)]
pub(super) struct Rejoins {
    /// The stop the code runs up to.
    stop: Option<BlockId>,
    /// For each block of that code that reaches the end, its immediate
    /// post-dominator, or `None` when that is the end itself.
    ipdom: BTreeMap<BlockId, Option<BlockId>>,
}

/// The code from a branch toward the end of its walk, reversed: an edge for
/// each step of flow that runs toward the end, from its target to its
/// source. Node 0 is the end: the stop, or the exits that two or more arms
/// reach.
struct TowardStop {
    /// The reversed edges leaving each node: its predecessors in the flow.
    sources: Vec<Vec<usize>>,
    /// The reversed edges entering each node: its successors in the flow.
    targets: Vec<Vec<usize>>,
}

impl GraphView for TowardStop {
    type NodeId = usize;

    fn node_bound(&self) -> usize {
        self.sources.len()
    }

    fn node_ids(&self) -> impl Iterator<Item = usize> + '_ {
        0..self.sources.len()
    }

    fn successors(&self, node: usize) -> impl Iterator<Item = usize> + '_ {
        self.sources[node].iter().copied()
    }

    fn predecessors(&self, node: usize) -> impl Iterator<Item = usize> + '_ {
        self.targets[node].iter().copied()
    }
}

/// Which arms of the branch reach a node of its code.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Reach {
    Unreached,
    /// One arm only, named by the node where it enters the code.
    One(usize),
    Several,
}

impl Reach {
    /// What a node reached as `self` and from a node reached as `from` is
    /// reached by.
    fn join(self, from: Self) -> Self {
        match (self, from) {
            (reach, Self::Unreached) | (Self::Unreached, reach) => reach,
            (Self::One(arm), Self::One(other)) if arm == other => self,
            _ => Self::Several,
        }
    }
}

impl<I, E> Structurer<'_, I, E> {
    /// Where the arms of the branch or dispatch at `block` rejoin, in a walk
    /// within `allowed` that ends at `stop`, or at its exits when it has no
    /// stop, when the branch has no merge point because another arm
    /// diverges.
    ///
    /// The arms that reach the end rejoin at the immediate post-dominator of
    /// `block` in the code between them, and those that never reach it (a
    /// `throw`, a `return`, a loop without exit) are left out. The end of a
    /// walk without a stop is every exit of the code that two or more arms
    /// reach: the exit after the join, where an exit that one arm alone
    /// reaches is that arm diverging. There is no such merge when fewer than
    /// two arms reach the end, since a lone arm that completes simply runs
    /// on to it, and none when the arms only meet at the end itself.
    ///
    /// The code holds the blocks the walk may still emit: in the bound, not
    /// yet emitted, and not a loop's transfer target, along sequential flow
    /// that is no loop's back edge. A walk would make a transfer of any other
    /// step, never a silent join, so a block whose every step is such a
    /// transfer is an exit of the code, unless it closes a loop whose header
    /// is in the code: that loop is a construct of the code, and only its
    /// own exits leave it.
    pub(super) fn rejoin(
        &mut self,
        block: BlockId,
        allowed: &Bound,
        stop: Option<BlockId>,
    ) -> Option<BlockId> {
        let toward_exits;
        let rejoins = if stop.is_some() {
            if self.rejoins.stop != stop || !self.rejoins.ipdom.contains_key(&block) {
                self.rejoins = self.rejoins_toward(block, allowed, stop);
            }
            &self.rejoins
        } else {
            toward_exits = self.rejoins_toward(block, allowed, None);
            &toward_exits
        };
        let mut completing: Vec<BlockId> = flow_successors(self.cfg, block)
            .into_iter()
            .filter(|&succ| {
                !self.back_edges.contains(&(block, succ))
                    && (Some(succ) == stop || rejoins.ipdom.contains_key(&succ))
            })
            .collect();
        completing.sort_unstable();
        completing.dedup();
        if completing.len() < 2 {
            return None;
        }
        rejoins
            .ipdom
            .get(&block)
            .copied()
            .flatten()
            .filter(|&merge| allowed.contains(merge))
    }

    /// The immediate post-dominators toward `stop`, or toward the exits that
    /// two or more arms of `block` reach, of the code that `block` reaches
    /// within `allowed`: see [`rejoin`](Self::rejoin).
    fn rejoins_toward(&self, block: BlockId, allowed: &Bound, stop: Option<BlockId>) -> Rejoins {
        let cfg = self.cfg;
        // Node 0 is the end, node 1 the branch; the rest follow as the search
        // from the branch finds them. Node `n` above 0 is `blocks[n - 1]`.
        let mut node_of: BTreeMap<BlockId, usize> = BTreeMap::from([(block, 1)]);
        if let Some(stop) = stop {
            node_of.insert(stop, 0);
        }
        let mut blocks = alloc::vec![block];
        let mut steps: Vec<(usize, usize)> = Vec::new();
        let mut back_steps: Vec<(usize, BlockId)> = Vec::new();
        let mut pending = alloc::vec![block];
        while let Some(source) = pending.pop() {
            let from = node_of[&source];
            for target in flow_successors(cfg, source) {
                if self.back_edges.contains(&(source, target)) {
                    back_steps.push((from, target));
                    continue;
                }
                let to = if let Some(&to) = node_of.get(&target) {
                    to
                } else if allowed.contains(target)
                    && !self.is_visited(target)
                    && !self.loop_targets.contains_key(&target)
                {
                    blocks.push(target);
                    let to = blocks.len();
                    node_of.insert(target, to);
                    pending.push(target);
                    to
                } else {
                    continue;
                };
                steps.push((from, to));
            }
        }
        let mut graph = TowardStop {
            sources: alloc::vec![Vec::new(); blocks.len() + 1],
            targets: alloc::vec![Vec::new(); blocks.len() + 1],
        };
        for (from, to) in steps {
            graph.sources[to].push(from);
            graph.targets[from].push(to);
        }
        // Without a shared exit, no arm reaches the end of a walk without a
        // stop, so nothing rejoins: a guard such as `if (a) { return }`
        // needs no tree.
        if stop.is_none() {
            let mut closes_loop = alloc::vec![false; graph.targets.len()];
            for (from, header) in back_steps {
                if node_of.contains_key(&header) {
                    closes_loop[from] = true;
                }
            }
            if !graph.join_shared_exits(&closes_loop) {
                return Rejoins::default();
            }
        }
        let tree = DominatorTree::compute_from(&graph, 0);
        let ipdom = (1..=blocks.len())
            .filter_map(|node| {
                let idom = tree.idom(node)?;
                Some((blocks[node - 1], idom.checked_sub(1).map(|at| blocks[at])))
            })
            .collect();
        Rejoins { stop, ipdom }
    }
}

impl TowardStop {
    /// Leads to the end (node 0) each exit of the code from the branch at
    /// node 1 that two or more of its arms reach, and returns whether there
    /// is one. A node that `closes_loop` marks steps back to a loop header in
    /// the code: it is that loop's latch, not an exit.
    fn join_shared_exits(&mut self, closes_loop: &[bool]) -> bool {
        let mut reach = alloc::vec![Reach::Unreached; self.targets.len()];
        let mut pending: Vec<usize> = Vec::new();
        for &arm in &self.targets[1] {
            if arm != 1 && reach[arm] == Reach::Unreached {
                reach[arm] = Reach::One(arm);
                pending.push(arm);
            }
        }
        // A node's reach only grows, from one arm to several, so each node
        // is pending at most twice.
        while let Some(node) = pending.pop() {
            for &next in &self.targets[node] {
                if next == 1 {
                    continue;
                }
                let joined = reach[next].join(reach[node]);
                if joined != reach[next] {
                    reach[next] = joined;
                    pending.push(next);
                }
            }
        }
        let mut shared = false;
        for node in 2..self.targets.len() {
            if self.targets[node].is_empty() && !closes_loop[node] && reach[node] == Reach::Several
            {
                self.sources[0].push(node);
                self.targets[node].push(0);
                shared = true;
            }
        }
        shared
    }
}
