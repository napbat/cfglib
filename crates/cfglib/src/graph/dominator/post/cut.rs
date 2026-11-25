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
use alloc::vec::Vec;

use super::super::{ABSENT, word};
use crate::graph::scc::tarjan_scc;
use crate::graph::traverse::{TraversalDirection, reachable_in};
use crate::graph::view::{DenseId, GraphView};

/// What the analysis finds and the finished post-dominator tree keeps.
pub(crate) struct ExitStructure<N> {
    /// Whether each block reaches an exit, `false` for a block the entry
    /// misses.
    pub(crate) reaches_exit: Vec<bool>,
    /// The reachable blocks without successors, in ascending dense index.
    pub(crate) exits: Vec<N>,
    /// The blocks the cut joins to the virtual exit, in ascending dense index.
    pub(crate) cut_roots: Vec<N>,
}

/// The sink components of the reachable blocks that reach no exit, as
/// classes.
pub(super) struct SinkClasses<N> {
    /// The class of each block, [`ABSENT`] for a block in none. Empty when
    /// there is no class.
    pub(super) class_of: Vec<u32>,
    /// The blocks of every class, class after class, each class in ascending
    /// dense index and the classes in ascending order of their first block.
    pub(super) members: Vec<N>,
    /// Class `c` is `members[starts[c]..starts[c + 1]]`. Empty when there is
    /// no class.
    pub(super) starts: Vec<u32>,
}

/// Every buffer the analysis fills, owned by the caller.
///
/// The two tables the augmented graph reads stay here after the analysis, so
/// the construction that follows needs no copy of them.
#[derive(Debug, Clone, Default)]
pub(crate) struct CutBuffers {
    /// Whether the entry reaches each block.
    pub(crate) reachable: Vec<bool>,
    /// Whether each block has an edge to the virtual exit, which holds for an
    /// exit and for a cut root.
    pub(super) exit_edge: Vec<bool>,
    /// Whether each block is an entry block of the component being cut.
    entry: Vec<bool>,
    /// Blocks a search has yet to expand.
    stack: Vec<usize>,
}

impl CutBuffers {
    /// Find the exits, the blocks that reach them, and the cut of the part of
    /// `graph` that `root` reaches.
    pub(crate) fn analyze<G: GraphView>(
        &mut self,
        graph: &G,
        root: G::NodeId,
    ) -> ExitStructure<G::NodeId> {
        let (exits, reaches_exit) = self.find_exits(graph, root);
        let cut_roots = self.cut_roots(graph, root, &reaches_exit);
        self.join(&exits, &cut_roots);
        ExitStructure {
            reaches_exit,
            exits,
            cut_roots,
        }
    }

    /// Find the exits and the blocks that reach them, as
    /// [`analyze`](Self::analyze) does, and in place of the cut the sink
    /// components of the blocks that reach none. Each class's lowest block
    /// stands for it: it is joined to the virtual exit and listed among the
    /// returned structure's cut roots, which the augmented graph reads.
    pub(super) fn analyze_sinks<G: GraphView>(
        &mut self,
        graph: &G,
        root: G::NodeId,
    ) -> (ExitStructure<G::NodeId>, SinkClasses<G::NodeId>) {
        let (exits, reaches_exit) = self.find_exits(graph, root);
        let mut classes: Vec<Vec<G::NodeId>> = Vec::new();
        for_each_sink(graph, &self.reachable, &reaches_exit, |component| {
            let mut class = component.to_vec();
            class.sort_unstable_by_key(|block| block.index());
            classes.push(class);
        });
        classes.sort_unstable_by_key(|class| class[0].index());

        let mut sinks = SinkClasses {
            class_of: Vec::new(),
            members: Vec::new(),
            starts: Vec::new(),
        };
        if !classes.is_empty() {
            sinks.class_of.resize(self.reachable.len(), ABSENT);
            sinks.starts.push(0);
        }
        for (class, blocks) in classes.iter().enumerate() {
            for &block in blocks {
                sinks.class_of[block.index()] = word(class);
            }
            sinks.members.extend_from_slice(blocks);
            sinks.starts.push(word(sinks.members.len()));
        }
        let representatives: Vec<G::NodeId> = classes.iter().map(|class| class[0]).collect();
        self.join(&exits, &representatives);
        let structure = ExitStructure {
            reaches_exit,
            exits,
            cut_roots: representatives,
        };
        (structure, sinks)
    }

    /// Mark the blocks `root` reaches, and return the reachable blocks
    /// without successors and whether each block reaches one of them.
    fn find_exits<G: GraphView>(
        &mut self,
        graph: &G,
        root: G::NodeId,
    ) -> (Vec<G::NodeId>, Vec<bool>) {
        self.mark_reachable(graph, root);
        let exits: Vec<G::NodeId> = (0..self.reachable.len())
            .filter(|&block| self.reachable[block])
            .map(G::NodeId::from_index)
            .filter(|&block| graph.successors(block).next().is_none())
            .collect();
        let reaches_exit = self.mark_reaching(graph, &exits);
        (exits, reaches_exit)
    }

    /// Give each of `exits` and `joined` an edge to the virtual exit.
    fn join<N: DenseId>(&mut self, exits: &[N], joined: &[N]) {
        self.exit_edge.clear();
        self.exit_edge.resize(self.reachable.len(), false);
        for block in exits.iter().chain(joined) {
            self.exit_edge[block.index()] = true;
        }
    }

    /// Mark the blocks `root` reaches.
    fn mark_reachable<G: GraphView>(&mut self, graph: &G, root: G::NodeId) {
        reachable_in(
            graph,
            [root],
            TraversalDirection::Outgoing,
            None,
            &mut self.reachable,
            &mut self.stack,
        );
    }

    /// Mark, among the reachable blocks, those with a path to an exit.
    ///
    /// This is the blocks that reach `exits` walking the edges backwards,
    /// confined to the blocks the entry reaches. The confinement is part of
    /// the answer: a dead block with an edge into live code reaches an exit
    /// too, and has no facts, so it must not be marked. It costs the walk
    /// nothing it needs, because a successor of a live block is live, so a
    /// path from a live block to an exit never leaves the live blocks.
    fn mark_reaching<G: GraphView>(&mut self, graph: &G, exits: &[G::NodeId]) -> Vec<bool> {
        let mut reaching = Vec::new();
        reachable_in(
            graph,
            exits.iter().copied(),
            TraversalDirection::Incoming,
            Some(&self.reachable),
            &mut reaching,
            &mut self.stack,
        );
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
        let entry = &mut self.entry;
        entry.clear();
        entry.resize(reachable.len(), false);
        let mut roots = Vec::new();
        for_each_sink(graph, reachable, reaches_exit, |component| {
            let contains = |block: G::NodeId| component.binary_search(&block).is_ok();
            for &block in component {
                entry[block.index()] = block == root
                    || graph
                        .predecessors(block)
                        .any(|before| reachable[before.index()] && !contains(before));
            }
            roots.extend(
                component
                    .iter()
                    .copied()
                    .filter(|&block| graph.successors(block).any(|next| entry[next.index()])),
            );
        });
        roots.sort_unstable_by_key(|block| block.index());
        roots
    }
}

/// Visit every sink component of the reachable blocks that reach no exit, a
/// strongly connected component with no edge to another, as a slice
/// ascending by the blocks' order.
fn for_each_sink<G: GraphView>(
    graph: &G,
    reachable: &[bool],
    reaches_exit: &[bool],
    mut visit: impl FnMut(&[G::NodeId]),
) {
    let trapped = Trapped {
        graph,
        reachable,
        reaches_exit,
    };
    if trapped.node_ids().next().is_none() {
        return;
    }
    let components = tarjan_scc(&trapped);
    for component in components.components() {
        let contains = |block: G::NodeId| component.binary_search(&block).is_ok();
        if component
            .iter()
            .all(|&block| graph.successors(block).all(contains))
        {
            visit(component);
        }
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
