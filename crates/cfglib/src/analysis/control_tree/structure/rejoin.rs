//! Where the arms of a branch rejoin when another arm diverges.
//!
//! The post-dominator tree joins every exit, a `throw` as much as a
//! `return`, so a branch with one arm that throws has no merge point even
//! when its other arms meet again before the walk's stop. Those arms rejoin
//! where they would if the diverging arm were not there: at the immediate
//! post-dominator of the branch in the code that runs up to the stop, with
//! every block that cannot reach the stop left out.

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
/// join), so the tree for a stop answers every branch inside its code.
#[derive(Default)]
pub(super) struct Rejoins {
    /// The stop the code runs up to.
    stop: Option<BlockId>,
    /// The immediate post-dominator of each block of that code that reaches
    /// the stop.
    ipdom: BTreeMap<BlockId, BlockId>,
}

/// The code from a branch up to a stop, reversed: an edge for each step of
/// flow that runs toward the stop, from its target to its source. Node 0 is
/// the stop.
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

impl<I, E> Structurer<'_, I, E> {
    /// Where the arms of the branch or dispatch at `block` rejoin, in a walk
    /// within `allowed` that ends at `stop`, when the branch has no merge
    /// point because another arm diverges.
    ///
    /// The arms that reach `stop` rejoin at the immediate post-dominator of
    /// `block` in the code between them, and those that never reach it (a
    /// `throw`, a `return`, a loop without exit) are left out. There is no
    /// such merge when fewer than two arms reach `stop`, since a lone arm
    /// that completes simply runs on to `stop`, and none when the arms only
    /// meet at `stop` itself.
    ///
    /// The code holds the blocks the walk may still emit: in the bound, not
    /// yet emitted, and not a loop's transfer target, along sequential flow
    /// that is no loop's back edge. A walk would make a transfer of any other
    /// step, never a silent join.
    pub(super) fn rejoin(
        &mut self,
        block: BlockId,
        allowed: &Bound,
        stop: BlockId,
    ) -> Option<BlockId> {
        if self.rejoins.stop != Some(stop) || !self.rejoins.ipdom.contains_key(&block) {
            self.rejoins = self.rejoins_toward(block, allowed, stop);
        }
        let mut completing: Vec<BlockId> = flow_successors(self.cfg, block)
            .into_iter()
            .filter(|&succ| {
                !self.back_edges.contains(&(block, succ))
                    && (succ == stop || self.rejoins.ipdom.contains_key(&succ))
            })
            .collect();
        completing.sort_unstable();
        completing.dedup();
        if completing.len() < 2 {
            return None;
        }
        self.rejoins
            .ipdom
            .get(&block)
            .copied()
            .filter(|&merge| merge != stop && allowed.contains(merge))
    }

    /// The immediate post-dominators toward `stop` of the code that `block`
    /// reaches within `allowed`: see [`rejoin`](Self::rejoin).
    fn rejoins_toward(&self, block: BlockId, allowed: &Bound, stop: BlockId) -> Rejoins {
        let cfg = self.cfg;
        // Node 0 is the stop, node 1 the branch; the rest follow as the
        // search from the branch finds them.
        let mut node_of: BTreeMap<BlockId, usize> = BTreeMap::from([(stop, 0), (block, 1)]);
        let mut blocks = alloc::vec![stop, block];
        let mut steps: Vec<(usize, usize)> = Vec::new();
        let mut pending = alloc::vec![block];
        while let Some(source) = pending.pop() {
            let from = node_of[&source];
            for target in flow_successors(cfg, source) {
                if self.back_edges.contains(&(source, target)) {
                    continue;
                }
                let to = if let Some(&to) = node_of.get(&target) {
                    to
                } else if allowed.contains(target)
                    && !self.is_visited(target)
                    && !self.loop_targets.contains_key(&target)
                {
                    let to = blocks.len();
                    node_of.insert(target, to);
                    blocks.push(target);
                    pending.push(target);
                    to
                } else {
                    continue;
                };
                steps.push((from, to));
            }
        }
        let mut graph = TowardStop {
            sources: alloc::vec![Vec::new(); blocks.len()],
            targets: alloc::vec![Vec::new(); blocks.len()],
        };
        for (from, to) in steps {
            graph.sources[to].push(from);
            graph.targets[from].push(to);
        }
        let tree = DominatorTree::compute_from(&graph, 0);
        let ipdom = (1..blocks.len())
            .filter_map(|node| Some((blocks[node], blocks[tree.idom(node)?])))
            .collect();
        Rejoins {
            stop: Some(stop),
            ipdom,
        }
    }
}
