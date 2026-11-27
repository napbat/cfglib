//! Post-dominance with each exit-free sink as one class: the
//! nontermination-insensitive reading of a graph without a unique exit.
//!
//! Under [`ExitPolicy::SinkComponents`] every sink component of the blocks
//! that reach no exit, a strongly connected component no edge leaves,
//! becomes one node joined to the virtual exit, and the tree is the
//! post-dominator tree of that contracted graph. This is the SINK
//! post-dominance of Hecker, Bischof and Snelting ("On Time-sensitive
//! Control Dependencies", TOPLAS 44(1), 2021): a path that enters a sink
//! never leaves it and visits all of it, so a block outside every sink
//! post-dominates `b` exactly when every path from `b` into the sinks passes
//! it, and a block of a sink exactly when every path from `b` ends in that
//! sink. The paper does not state the contraction; the tests check it
//! against the paper's fixpoint definition on every small graph.
//!
//! A class is one node of the tree, named by its lowest block. The other
//! blocks of the class share its place in the numbering, so
//! [`post_dominates`](PostDominatorTree::post_dominates) stays one interval
//! test and holds both ways between two blocks of a class. The queries that
//! walk the tree read a block of a class as the class, and the ones that list
//! dependents expand a class into its blocks.
//!
//! A tree without classes, which is every tree of the cut, holds an empty
//! [`SinkClasses`], and every reading here is then the identity: the
//! contracted graph is the graph and a block stands for itself. The queries
//! therefore have one form for both policies.

extern crate alloc;
use alloc::vec::Vec;

use super::PostDominatorTree;
use crate::graph::dominator::DominatorTree;
use crate::graph::view::{DenseId, GraphView};
use crate::identity::{NO_INDEX, define_dense_id, slot, word};

/// How a [`PostDominatorTree`] gives facts to the blocks that reach no exit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum ExitPolicy {
    /// Join to the virtual exit the blocks that close each exit-free loop,
    /// the cut [`PostDominatorTree::compute`] makes. Facts there are relative
    /// to the cut, and a loop's latch controls its header.
    #[default]
    CutLoops,
    /// Join each sink component of the exit-free blocks to the virtual exit as
    /// one class, whose blocks post-dominate each other. Control dependence
    /// over this tree is nontermination-insensitive: a block that runs on
    /// every path into a sink depends on nothing inside it.
    SinkComponents,
}

define_dense_id! {
    /// A sink class of a tree computed under [`ExitPolicy::SinkComponents`],
    /// numbered from zero in ascending order of the classes' lowest blocks.
    pub struct SinkId(u32);
}

/// The sink components of the reachable blocks that reach no exit, as
/// classes; empty for a tree of the cut.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SinkClasses<N> {
    /// The class of each block, [`NO_INDEX`] for a block in none. Empty when
    /// there is no class.
    class_of: Vec<u32>,
    /// The blocks of every class, class after class, each class in ascending
    /// dense index and the classes in ascending order of their first block.
    members: Vec<N>,
    /// Class `c` is `members[starts[c]..starts[c + 1]]`. Empty when there is
    /// no class.
    starts: Vec<u32>,
}

impl<N> Default for SinkClasses<N> {
    /// No class at all, which allocates nothing.
    fn default() -> Self {
        Self {
            class_of: Vec::new(),
            members: Vec::new(),
            starts: Vec::new(),
        }
    }
}

impl<N: DenseId> SinkClasses<N> {
    /// The classes `classes` over `bound` blocks, each class ascending and
    /// the classes in ascending order of their first block.
    pub(super) fn new(bound: usize, classes: &[Vec<N>]) -> Self {
        let mut sinks = Self::default();
        if classes.is_empty() {
            return sinks;
        }
        sinks.class_of.resize(bound, NO_INDEX);
        sinks.starts.push(0);
        for (class, blocks) in classes.iter().enumerate() {
            for &block in blocks {
                sinks.class_of[block.index()] = word(class);
            }
            sinks.members.extend_from_slice(blocks);
            sinks.starts.push(word(sinks.members.len()));
        }
        sinks
    }

    /// Whether there is no class.
    pub(super) fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    /// How many blocks the classes hold beyond the one that names each:
    /// the blocks the contracted graph leaves out.
    pub(super) fn hidden(&self) -> usize {
        self.members.len() - self.starts.len().saturating_sub(1)
    }

    /// The class of `block`, [`NO_INDEX`] for none.
    #[inline]
    fn class(&self, block: N) -> u32 {
        self.class_of
            .get(block.index())
            .copied()
            .unwrap_or(NO_INDEX)
    }

    /// The blocks of class `class`, ascending.
    fn blocks(&self, class: u32) -> &[N] {
        &self.members[slot(self.starts[slot(class)])..slot(self.starts[slot(class) + 1])]
    }

    /// The block `block` is read as: the lowest block of its class, or
    /// itself.
    #[inline]
    pub(super) fn canonical(&self, block: N) -> N {
        match self.class(block) {
            NO_INDEX => block,
            class => self.blocks(class)[0],
        }
    }

    /// The blocks of the class `node` names, or none when it names no class.
    #[inline]
    pub(super) fn named(&self, node: N) -> &[N] {
        match self.class(node) {
            NO_INDEX => &[],
            class if self.blocks(class)[0] == node => self.blocks(class),
            _ => &[],
        }
    }

    /// Whether two blocks lie in one class.
    #[inline]
    pub(super) fn share(&self, first: N, second: N) -> bool {
        let class = self.class(first);
        class != NO_INDEX && class == self.class(second)
    }

    /// Give the other blocks of each class the place of its lowest block in
    /// the numbering of `tree`, which only the lowest block reached.
    pub(super) fn share_numbering(&self, tree: &mut DominatorTree<N>) {
        for class in self.starts.windows(2) {
            let blocks = &self.members[slot(class[0])..slot(class[1])];
            let first = blocks[0].index();
            for block in &blocks[1..] {
                let block = block.index();
                tree.pre[block] = tree.pre[first];
                tree.last[block] = tree.last[first];
                tree.level[block] = tree.level[first];
            }
        }
    }

    /// Give the other blocks of each class the `(runner, frontier block)`
    /// pairs of the block that names it, so a frontier table reads every
    /// block of a class as the class.
    pub(super) fn share_runs(&self, pairs: &mut Vec<(N, N)>) {
        if self.is_empty() {
            return;
        }
        for index in 0..pairs.len() {
            let (runner, block) = pairs[index];
            if let [_, others @ ..] = self.named(runner) {
                pairs.extend(others.iter().map(|&other| (other, block)));
            }
        }
    }
}

impl<N: DenseId> PostDominatorTree<N> {
    /// Return the sink class of `block` under
    /// [`ExitPolicy::SinkComponents`], or `None` when it is in none, which
    /// holds for every block of a tree computed under another policy.
    ///
    /// # Panics
    ///
    /// Panics when `block` has an index beyond the node bound the tree was
    /// computed with and the tree has sink classes.
    #[must_use]
    pub fn sink_class(&self, block: N) -> Option<SinkId> {
        let class = *self.classes.class_of.get(block.index())?;
        (class != NO_INDEX).then_some(SinkId(class))
    }

    /// Return the blocks of a sink class, in ascending dense index; the first
    /// names the class in [`ipdom`](Self::ipdom), [`preorder`](Self::preorder),
    /// [`subtree`](Self::subtree) and [`children`](Self::children).
    ///
    /// # Panics
    ///
    /// Panics when `class` is not a class of this tree.
    #[must_use]
    pub fn sink_members(&self, class: SinkId) -> &[N] {
        self.classes.blocks(class.0)
    }

    /// `graph` with each class contracted as the tree reads it, which is
    /// `graph` itself for a tree without classes.
    pub(super) const fn contracted<'a, G>(&'a self, graph: &'a G) -> Contracted<'a, G>
    where
        G: GraphView<NodeId = N>,
    {
        Contracted::new(graph, &self.classes)
    }
}

/// A graph with every sink class contracted into its lowest block.
///
/// An edge inside a class disappears, an edge into a class enters its lowest
/// block, and the other blocks of a class have no edges. No edge leaves a
/// sink, so nothing else changes, and without classes the view is the graph.
pub(super) struct Contracted<'a, G: GraphView> {
    graph: &'a G,
    classes: &'a SinkClasses<G::NodeId>,
}

impl<'a, G: GraphView> Contracted<'a, G> {
    /// `graph` with the classes of `classes` contracted.
    pub(super) const fn new(graph: &'a G, classes: &'a SinkClasses<G::NodeId>) -> Self {
        Self { graph, classes }
    }
}

impl<G: GraphView> GraphView for Contracted<'_, G> {
    type NodeId = G::NodeId;

    fn node_bound(&self) -> usize {
        self.graph.node_bound()
    }

    fn node_ids(&self) -> impl Iterator<Item = G::NodeId> + '_ {
        self.graph
            .node_ids()
            .filter(|&node| self.classes.canonical(node) == node)
    }

    #[inline]
    fn successors(&self, node: G::NodeId) -> impl Iterator<Item = G::NodeId> + '_ {
        let outside = (self.classes.class(node) == NO_INDEX).then(|| {
            self.graph
                .successors(node)
                .map(|next| self.classes.canonical(next))
        });
        outside.into_iter().flatten()
    }

    #[inline]
    fn predecessors(&self, node: G::NodeId) -> impl Iterator<Item = G::NodeId> + '_ {
        let class = self.classes.class(node);
        let outside = (class == NO_INDEX).then(|| self.graph.predecessors(node));
        let gathered = (class != NO_INDEX && self.classes.blocks(class)[0] == node).then(|| {
            self.classes.blocks(class).iter().flat_map(move |&block| {
                self.graph
                    .predecessors(block)
                    .filter(move |&before| self.classes.class(before) != class)
            })
        });
        outside
            .into_iter()
            .flatten()
            .chain(gathered.into_iter().flatten())
    }
}
