//! Dominance extended from a tree to a forest, so unreachable code still has
//! definitions that flow.

extern crate alloc;

use alloc::vec::Vec;

use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::graph::dominator::{DominatorScratch, DominatorTree};
use crate::graph::view::{DenseId, GraphView, RootedView};
use crate::kosaraju_scc;

/// Whole-CFG dominance view with a private virtual root connected to the
/// ordinary entry and every source SCC of otherwise unreachable code.
struct SsaDominatorView<'a, I, E> {
    cfg: &'a Cfg<I, E>,
    roots: &'a [BlockId],
}

impl<I, E> SsaDominatorView<'_, I, E> {
    fn virtual_root(&self) -> usize {
        self.cfg.block_bound()
    }
}

impl<I, E> GraphView for SsaDominatorView<'_, I, E> {
    type NodeId = usize;

    fn node_bound(&self) -> usize {
        self.cfg.block_bound() + 1
    }

    fn node_ids(&self) -> impl Iterator<Item = Self::NodeId> + '_ {
        self.cfg
            .block_ids()
            .map(DenseId::index)
            .chain(core::iter::once(self.virtual_root()))
    }

    fn successors(&self, node: Self::NodeId) -> impl Iterator<Item = Self::NodeId> + '_ {
        let original_bound = self.cfg.block_bound();
        let original = (node < original_bound).then(|| BlockId::from_index(node));
        (node == original_bound)
            .then_some(self.roots)
            .into_iter()
            .flatten()
            .copied()
            .map(BlockId::index)
            .chain(
                original
                    .into_iter()
                    .flat_map(|block| self.cfg.successors(block).map(BlockId::index)),
            )
    }

    fn predecessors(&self, node: Self::NodeId) -> impl Iterator<Item = Self::NodeId> + '_ {
        let original_bound = self.cfg.block_bound();
        let original = (node < original_bound).then(|| BlockId::from_index(node));
        let virtual_predecessor = original
            .filter(|block| self.roots.binary_search(block).is_ok())
            .map(|_| original_bound);
        virtual_predecessor.into_iter().chain(
            original
                .into_iter()
                .flat_map(|block| self.cfg.predecessors(block).map(BlockId::index)),
        )
    }
}

impl<I, E> RootedView for SsaDominatorView<'_, I, E> {
    fn root(&self) -> Self::NodeId {
        self.virtual_root()
    }
}

/// Extends ordinary entry-rooted dominance into a forest for disconnected
/// code while retaining independent version-zero live-ins at every source
/// component.
///
/// Returns `None` when the entry already reaches every block, which is the
/// common case and the one that must stay free: the check is a scan, and the
/// whole forest construction below it never runs.
pub(super) fn complete_dominator_forest<I, E>(
    scratch: &mut DominatorScratch,
    roots: &mut Vec<BlockId>,
    cfg: &Cfg<I, E>,
    dominators: &DominatorTree,
) -> Option<DominatorTree> {
    if cfg.block_ids().all(|block| dominators.is_reachable(block)) {
        return None;
    }

    let components = kosaraju_scc(cfg);
    roots.clear();
    roots.push(cfg.entry());
    for (component_index, component) in components.components.iter().enumerate() {
        let Some(&first) = component.nodes.iter().next() else {
            continue;
        };
        if dominators.is_reachable(first) {
            continue;
        }
        let has_external_predecessor = component.nodes.iter().copied().any(|block| {
            cfg.predecessors(block)
                .any(|predecessor| components.component_index(predecessor) != component_index)
        });
        if !has_external_predecessor {
            roots.push(first);
        }
    }
    roots.sort_unstable();
    roots.dedup();

    let view = SsaDominatorView { cfg, roots };
    let complete = DominatorTree::<usize>::compute_in(scratch, &view);
    let original_bound = cfg.block_bound();
    let idom = (0..original_bound)
        .map(|node| {
            complete
                .idom(node)
                .filter(|&parent| parent < original_bound)
                .map(BlockId::from_index)
        })
        .collect();
    let reachable = (0..original_bound)
        .map(|node| complete.is_reachable(node))
        .collect();
    Some(DominatorTree::from_forest_parts(idom, reachable))
}
