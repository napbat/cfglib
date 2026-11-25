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

extern crate alloc;
use alloc::vec::Vec;

use super::augmented::AugmentedReverse;
use super::{PostDominatorScratch, PostDominatorTree};
use crate::graph::dominator::{ABSENT, DominatorTree, slot, word};
use crate::graph::view::{DenseId, GraphView, RootedView};

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

/// A sink class of a tree computed under [`ExitPolicy::SinkComponents`],
/// numbered from zero in ascending order of the classes' lowest blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SinkId(u32);

impl SinkId {
    /// The number of the class, below the number of classes.
    #[must_use]
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

impl<N: DenseId> PostDominatorTree<N> {
    /// Compute the post-dominator tree under `policy` over caller-owned
    /// working storage.
    ///
    /// [`ExitPolicy::CutLoops`] builds exactly the tree of
    /// [`compute_in`](Self::compute_in). [`ExitPolicy::SinkComponents`]
    /// makes each sink component of the blocks that reach no exit one class
    /// joined to the virtual exit (see [`sink_class`](Self::sink_class)), and
    /// has no cut roots. On a graph whose reachable blocks all reach an exit
    /// the two trees are `==`. Both cost the same O(m log n) construction plus
    /// O(n + m) to find the exits and the sinks.
    ///
    /// A low-high certificate (see
    /// [`compute_certified_in`](Self::compute_certified_in)) covers the cut
    /// only, so [`verify_certificate`](Self::verify_certificate) rejects a tree
    /// with sink classes.
    ///
    /// # Panics
    ///
    /// Panics under the conditions of [`compute`](Self::compute).
    ///
    /// # Examples
    ///
    /// A server loop reached from the entry never exits. Under the cut its
    /// latch controls the loop; with the loop as one class nothing inside it
    /// controls anything, because every block of it runs again on every
    /// execution:
    ///
    /// ```
    /// use cfglib::{Cfg, EdgeKind, ExitPolicy, PostDominatorScratch, PostDominatorTree};
    ///
    /// let mut cfg = Cfg::<u32>::new();
    /// let entry = cfg.entry();
    /// let head = cfg.new_block();
    /// let ok = cfg.new_block();
    /// let latch = cfg.new_block();
    /// cfg.add_edge(entry, head, EdgeKind::Unconditional);
    /// cfg.add_edge(head, ok, EdgeKind::ConditionalTrue);
    /// cfg.add_edge(head, latch, EdgeKind::ConditionalFalse);
    /// cfg.add_edge(ok, latch, EdgeKind::Unconditional);
    /// cfg.add_edge(latch, head, EdgeKind::Back);
    ///
    /// let mut scratch = PostDominatorScratch::new();
    /// let post =
    ///     PostDominatorTree::compute_with_in(&mut scratch, &cfg, ExitPolicy::SinkComponents);
    /// let class = post.sink_class(ok).expect("the loop is a sink");
    /// assert_eq!(post.sink_members(class), [head, ok, latch]);
    /// assert!(post.post_dominates(ok, latch) && post.post_dominates(latch, ok));
    /// assert_eq!(post.control_dependents(head, ok).count(), 0);
    /// assert!(post.cut_roots().is_empty());
    /// ```
    #[must_use]
    pub fn compute_with_in<G>(
        scratch: &mut PostDominatorScratch,
        graph: &G,
        policy: ExitPolicy,
    ) -> Self
    where
        G: RootedView<NodeId = N>,
    {
        match policy {
            ExitPolicy::CutLoops => Self::compute_in(scratch, graph),
            ExitPolicy::SinkComponents => Self::compute_sinks(scratch, graph),
        }
    }

    /// The tree under [`ExitPolicy::SinkComponents`].
    fn compute_sinks<G>(scratch: &mut PostDominatorScratch, graph: &G) -> Self
    where
        G: RootedView<NodeId = N>,
    {
        let bound = graph.node_bound();
        assert!(
            bound < slot(ABSENT),
            "node bound plus the virtual exit exceeds the dense u32 range of a post-dominator tree"
        );
        let root = graph.root();
        assert!(
            root.index() < bound,
            "post-dominator root is not a node of the graph"
        );

        let PostDominatorScratch { dominators, cut } = scratch;
        let (structure, classes) = cut.analyze_sinks(graph, root);
        let contracted = Contracted {
            graph,
            class_of: &classes.class_of,
            members: &classes.members,
            starts: &classes.starts,
        };
        let augmented = AugmentedReverse::new(&contracted, cut, &structure);
        let mut parent = dominators
            .construction
            .immediate_dominators(&augmented, bound, false);
        // The virtual exit is reached and is not a block.
        let reached = dominators.construction.reached() - 1;
        debug_assert_eq!(
            reached + classes.members.len() - classes.starts.len().saturating_sub(1),
            cut.reachable.iter().filter(|&&reachable| reachable).count(),
            "a reachable block outside every sink has no path to the virtual exit"
        );

        parent.truncate(bound);
        let virtual_exit = word(bound);
        let mut roots = Vec::new();
        for (block, up) in parent.iter_mut().enumerate() {
            if *up == virtual_exit {
                *up = ABSENT;
                roots.push(word(block));
            }
        }
        let numbered = dominators.numbering.number(&parent, &roots, reached);
        let mut tree = DominatorTree::from_numbered(parent, numbered);
        // The other blocks of a class take its place in the numbering.
        for class in classes.starts.windows(2) {
            let blocks = &classes.members[slot(class[0])..slot(class[1])];
            let first = blocks[0].index();
            for block in &blocks[1..] {
                let block = block.index();
                tree.pre[block] = tree.pre[first];
                tree.last[block] = tree.last[first];
                tree.level[block] = tree.level[first];
            }
        }

        Self {
            tree,
            reaches_exit: structure.reaches_exit,
            exits: structure.exits,
            cut_roots: Vec::new(),
            class_of: classes.class_of,
            class_members: classes.members,
            class_starts: classes.starts,
        }
    }

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
        let class = *self.class_of.get(block.index())?;
        (class != ABSENT).then_some(SinkId(class))
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
        let index = class.index();
        &self.class_members[slot(self.class_starts[index])..slot(self.class_starts[index + 1])]
    }

    /// The block that names the class of `block`, or `block` itself when it
    /// is in no class.
    pub(super) fn canonical(&self, block: N) -> N {
        self.sink_class(block)
            .map_or(block, |class| self.sink_members(class)[0])
    }

    /// The blocks of the class `node` names, or none when it names no class.
    pub(super) fn named_class(&self, node: N) -> &[N] {
        match self.sink_class(node) {
            Some(class) if self.sink_members(class)[0] == node => self.sink_members(class),
            _ => &[],
        }
    }

    /// Whether two blocks lie in one sink class.
    pub(super) fn share_class(&self, first: N, second: N) -> bool {
        self.sink_class(first)
            .is_some_and(|class| self.sink_class(second) == Some(class))
    }

    /// `graph` with each class contracted as the tree reads it, for the
    /// frontier queries; `None` for a tree without classes, which reads
    /// `graph` as it is.
    pub(super) fn contracted<'a, G>(&'a self, graph: &'a G) -> Option<Contracted<'a, G>>
    where
        G: GraphView<NodeId = N>,
    {
        (!self.class_members.is_empty()).then_some(Contracted {
            graph,
            class_of: &self.class_of,
            members: &self.class_members,
            starts: &self.class_starts,
        })
    }
}

/// A graph with every sink class contracted into its lowest block.
///
/// An edge inside a class disappears, an edge into a class enters its lowest
/// block, and the other blocks of a class have no edges. No edge leaves a
/// sink, so nothing else changes.
pub(super) struct Contracted<'a, G: GraphView> {
    graph: &'a G,
    class_of: &'a [u32],
    members: &'a [G::NodeId],
    starts: &'a [u32],
}

impl<G: GraphView> Contracted<'_, G> {
    /// The class of `node`, [`ABSENT`] for none.
    fn class(&self, node: G::NodeId) -> u32 {
        self.class_of.get(node.index()).copied().unwrap_or(ABSENT)
    }

    /// The blocks of class `class`.
    fn blocks(&self, class: u32) -> &[G::NodeId] {
        &self.members[slot(self.starts[slot(class)])..slot(self.starts[slot(class) + 1])]
    }

    /// The node `node` is read as: the lowest block of its class, or itself.
    fn canonical(&self, node: G::NodeId) -> G::NodeId {
        match self.class(node) {
            ABSENT => node,
            class => self.blocks(class)[0],
        }
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
            .filter(|&node| self.canonical(node) == node)
    }

    fn successors(&self, node: G::NodeId) -> impl Iterator<Item = G::NodeId> + '_ {
        let outside = (self.class(node) == ABSENT)
            .then(|| self.graph.successors(node).map(|next| self.canonical(next)));
        outside.into_iter().flatten()
    }

    fn predecessors(&self, node: G::NodeId) -> impl Iterator<Item = G::NodeId> + '_ {
        let class = self.class(node);
        let outside = (class == ABSENT).then(|| self.graph.predecessors(node));
        let gathered = (class != ABSENT && self.blocks(class)[0] == node).then(|| {
            self.blocks(class).iter().flat_map(move |&block| {
                self.graph
                    .predecessors(block)
                    .filter(move |&before| self.class(before) != class)
            })
        });
        outside
            .into_iter()
            .flatten()
            .chain(gathered.into_iter().flatten())
    }
}
