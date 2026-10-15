//! The exits of a graph and the cut that lets every block reach the virtual
//! exit.
//!
//! A block the entry reaches but that cannot reach an exit lies in a region
//! the program never leaves. Such a region would have no path to the virtual
//! exit, and its blocks would have no post-dominance facts. The cut gives it
//! one: it joins to the virtual exit the blocks that close each of the
//! region's loops.
//!
//! Let `U` be the reachable blocks that cannot reach an exit. A successor of a
//! block in `U` cannot reach an exit either, so `U` is closed under
//! successors. Every sink component `S` of `U`, a strongly connected component
//! with no edge to another one, is cut:
//!
//! 1. Its entry blocks are the graph's entry, when `S` holds it, and every
//!    block of `S` with a reachable predecessor outside `S`.
//! 2. Every block of `S` with an edge to an entry block of `S` is joined to
//!    the virtual exit.
//!
//! Every path inside `S` that reaches an entry block ends with such an edge,
//! so every block of `S` reaches the virtual exit. A component of `U` that is
//! not a sink reaches a sink, and needs no cut of its own. The result depends
//! only on the edge set among the reachable blocks, not on the order of
//! successor lists, on dense indices, or on the order of a depth-first search.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;
use core::mem;

use crate::graph::dominator::{slot, word};
use crate::graph::scc::tarjan_scc;
use crate::graph::view::{DenseId, GraphView};

/// What the analysis finds and the finished post-dominator tree keeps.
pub(super) struct ExitStructure<N> {
    /// Whether each block reaches an exit, `false` for a block the entry
    /// misses.
    pub(super) reaches_exit: Vec<bool>,
    /// The reachable blocks without successors, in ascending dense index.
    pub(super) exits: Vec<N>,
    /// The blocks the cut joins to the virtual exit, in ascending dense index.
    pub(super) cut_roots: Vec<N>,
}

/// Every buffer the analysis fills, owned by the caller.
///
/// The two tables the augmented graph reads stay here after the analysis, so
/// the construction that follows needs no copy of them.
#[derive(Debug, Clone, Default)]
pub(super) struct CutBuffers {
    /// Whether the entry reaches each block.
    pub(super) reachable: Vec<bool>,
    /// Whether each block has an edge to the virtual exit, which holds for an
    /// exit and for a cut root.
    pub(super) exit_edge: Vec<bool>,
    /// Whether each block is an entry block of the component being cut.
    entry: Vec<bool>,
    /// Blocks a search has yet to expand.
    stack: Vec<u32>,
}

impl CutBuffers {
    /// Find the exits, the blocks that reach them, and the cut of the part of
    /// `graph` that `root` reaches.
    pub(super) fn analyze<G: GraphView>(
        &mut self,
        graph: &G,
        root: G::NodeId,
    ) -> ExitStructure<G::NodeId> {
        self.mark_reachable(graph, root);
        let exits: Vec<G::NodeId> = (0..self.reachable.len())
            .filter(|&block| self.reachable[block])
            .map(G::NodeId::from_index)
            .filter(|&block| graph.successors(block).next().is_none())
            .collect();
        let reaches_exit = self.mark_reaching(graph, &exits);
        let cut_roots = self.cut_roots(graph, root, &reaches_exit);

        self.exit_edge.clear();
        self.exit_edge.resize(self.reachable.len(), false);
        for block in exits.iter().chain(&cut_roots) {
            self.exit_edge[block.index()] = true;
        }
        ExitStructure {
            reaches_exit,
            exits,
            cut_roots,
        }
    }

    /// Mark the blocks `root` reaches.
    fn mark_reachable<G: GraphView>(&mut self, graph: &G, root: G::NodeId) {
        self.reachable.clear();
        self.reachable.resize(graph.node_bound(), false);
        self.reachable[root.index()] = true;
        self.stack.clear();
        self.stack.push(word(root.index()));
        while let Some(block) = self.stack.pop() {
            for successor in graph.successors(G::NodeId::from_index(slot(block))) {
                if !mem::replace(&mut self.reachable[successor.index()], true) {
                    self.stack.push(word(successor.index()));
                }
            }
        }
    }

    /// Mark, among the reachable blocks, those with a path to an exit.
    fn mark_reaching<G: GraphView>(&mut self, graph: &G, exits: &[G::NodeId]) -> Vec<bool> {
        let mut reaching = vec![false; self.reachable.len()];
        self.stack.clear();
        for exit in exits {
            reaching[exit.index()] = true;
            self.stack.push(word(exit.index()));
        }
        while let Some(block) = self.stack.pop() {
            for predecessor in graph.predecessors(G::NodeId::from_index(slot(block))) {
                let predecessor = predecessor.index();
                if self.reachable[predecessor] && !mem::replace(&mut reaching[predecessor], true) {
                    self.stack.push(word(predecessor));
                }
            }
        }
        reaching
    }

    /// The blocks the cut joins to the virtual exit, in ascending dense index.
    fn cut_roots<G: GraphView>(
        &mut self,
        graph: &G,
        root: G::NodeId,
        reaches_exit: &[bool],
    ) -> Vec<G::NodeId> {
        let reachable = &self.reachable;
        let trapped = Trapped {
            graph,
            reachable,
            reaches_exit,
        };
        let mut roots = Vec::new();
        if trapped.node_ids().next().is_none() {
            return roots;
        }

        let components = tarjan_scc(&trapped);
        let entry = &mut self.entry;
        entry.clear();
        entry.resize(reachable.len(), false);
        for component in &components.components {
            let is_sink = component
                .nodes
                .iter()
                .all(|&block| graph.successors(block).all(|next| component.contains(next)));
            if !is_sink {
                continue;
            }
            for &block in &component.nodes {
                entry[block.index()] = block == root
                    || graph
                        .predecessors(block)
                        .any(|before| reachable[before.index()] && !component.contains(before));
            }
            roots.extend(
                component
                    .nodes
                    .iter()
                    .copied()
                    .filter(|&block| graph.successors(block).any(|next| entry[next.index()])),
            );
        }
        roots.sort_unstable_by_key(|block| block.index());
        roots
    }
}

/// The reachable blocks that cannot reach an exit, as a graph of their own.
///
/// The set is closed under successors, so every edge a member has ends at a
/// member and the view is an induced subgraph. A block outside the set has no
/// edges in the view.
struct Trapped<'a, G: GraphView> {
    graph: &'a G,
    reachable: &'a [bool],
    reaches_exit: &'a [bool],
}

impl<G: GraphView> Trapped<'_, G> {
    /// Whether the block at `index` is a member.
    fn holds(&self, index: usize) -> bool {
        self.reachable[index] && !self.reaches_exit[index]
    }

    /// `node` when it is a member.
    fn member(&self, node: G::NodeId) -> Option<G::NodeId> {
        self.holds(node.index()).then_some(node)
    }
}

impl<G: GraphView> GraphView for Trapped<'_, G> {
    type NodeId = G::NodeId;

    fn node_bound(&self) -> usize {
        self.graph.node_bound()
    }

    fn node_ids(&self) -> impl Iterator<Item = G::NodeId> + '_ {
        (0..self.reachable.len())
            .filter(|&index| self.holds(index))
            .map(G::NodeId::from_index)
    }

    fn successors(&self, node: G::NodeId) -> impl Iterator<Item = G::NodeId> + '_ {
        self.member(node)
            .map(|node| self.graph.successors(node))
            .into_iter()
            .flatten()
    }

    fn predecessors(&self, node: G::NodeId) -> impl Iterator<Item = G::NodeId> + '_ {
        self.member(node)
            .map(|node| {
                self.graph
                    .predecessors(node)
                    .filter(move |&before| self.holds(before.index()))
            })
            .into_iter()
            .flatten()
    }
}
