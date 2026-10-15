//! Post-dominator trees over a virtual exit.
//!
//! A block `a` post-dominates `b` when every path from `b` to the exit passes
//! through `a`. A graph can have several exits, and a block the entry reaches
//! can have none to reach, so the tree is rooted at a virtual exit `X*` that
//! every exit joins. [`PostDominatorTree::compute`] builds it in the time of one
//! dominator computation plus one linear pass over the exits and one over the
//! exit-free regions.
//!
//! The work is split by responsibility. `cut` finds the exits and gives every
//! exit-free region a path to the virtual exit, `augmented` presents the
//! resulting graph to the simple Lengauer-Tarjan construction, and this module
//! holds the tree and its queries.

extern crate alloc;
use alloc::vec::Vec;

use super::frontier::FrontierScratch;
use super::{ABSENT, DominatorScratch, DominatorTree, slot, word};
use crate::block::BlockId;
use crate::graph::view::{DenseId, GraphView, Reversed, RootedView};
use augmented::AugmentedReverse;
use cut::{CutBuffers, ExitStructure};

mod augmented;
mod cut;

/// A node of a post-dominator tree: a block of the graph, or the virtual exit
/// that every exit joins.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PostNode<N> {
    /// A block of the graph.
    Node(N),
    /// The virtual exit `X*`, the root of the tree.
    Exit,
}

/// Every buffer [`PostDominatorTree::compute_in`] fills, owned by the caller.
///
/// This is [`DominatorScratch`] for the post-dominator tree: a pass that
/// computes one tree per procedure hands one scratch to every call, and the
/// buffers grow to the largest graph the pass meets instead of being
/// allocated again for each procedure. It holds the dominator computation's
/// buffers and the ones that find the exits and the cut.
///
/// Nodes are held as dense indices rather than as a node-id type, so the
/// scratch carries no type parameter: one instance serves a [`Cfg`](crate::Cfg),
/// a [`Graph`](crate::Graph), and a consumer-defined [`GraphView`], in any
/// order, and it is [`Send`]. Every buffer is cleared and resized on entry,
/// so a scratch used on a large graph and then on a small one is correct and
/// allocation-free.
///
/// # Examples
///
/// ```
/// use cfglib::{Cfg, EdgeKind, PostDominatorScratch, PostDominatorTree};
///
/// let mut cfg = Cfg::<u32>::new();
/// let entry = cfg.entry();
/// let then_block = cfg.new_block();
/// cfg.add_edge(entry, then_block, EdgeKind::ConditionalTrue);
///
/// let mut scratch = PostDominatorScratch::new();
/// let tree = PostDominatorTree::compute_in(&mut scratch, &cfg);
/// assert_eq!(tree, PostDominatorTree::compute(&cfg));
/// ```
#[derive(Debug, Clone, Default)]
pub struct PostDominatorScratch {
    /// The dominator computation's buffers.
    dominators: DominatorScratch,
    /// The buffers of the exit and cut analysis.
    cut: CutBuffers,
}

impl PostDominatorScratch {
    /// Scratch holding nothing yet.
    ///
    /// Every buffer is sized by the first graph it sees, so there is no node
    /// count to state here.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

/// A post-dominator tree computed from a rooted directed graph.
///
/// A block `a` post-dominates `b` when every path from `b` to the virtual exit
/// `X*` passes through `a`, and the immediate post-dominators form a tree
/// rooted at `X*`. Only the blocks the entry reaches are in the tree; the
/// other blocks have no facts.
///
/// # The virtual exit
///
/// Every reachable block without successors, such as a `return`, a tail call,
/// or an `unreachable` terminator, has an edge to `X*`. These blocks are the
/// [`exits`](Self::exits). The tree is a forest over the blocks, whose
/// roots are the children of `X*`: [`ipdom`](Self::ipdom) names `X*` as
/// [`PostNode::Exit`], and `children(PostNode::Exit)` lists the roots.
///
/// # Exit-free regions
///
/// A reachable block can be unable to reach any exit: a server's main loop,
/// or a loop that a tamper check spins in forever. For each such region that
/// no path leaves, the tree joins to `X*` the blocks that close its loops.
/// Every strongly connected component of the exit-free blocks that has no
/// edge to another component gets this cut, as follows.
///
/// - Its entry blocks are the graph's entry, if the component holds it, and
///   every block of the component with a reachable predecessor outside it.
/// - Every block of the component with an edge to one of its entry blocks gets
///   an edge to `X*`. These are the [`cut_roots`](Self::cut_roots).
///
/// For a loop with one header, that connects every latch of the header to
/// `X*`. The cut depends only on the edges among the reachable blocks, so it is
/// the same for every order of successor lists and every number of parallel
/// edges. Components that are not sinks reach a sink and need no cut of their
/// own.
///
/// What this means for a consumer:
///
/// - On a block with [`reaches_exit`](Self::reaches_exit), every fact is also
///   a fact of the plain graph with only the real exits. The extra edges
///   add paths, so they can only remove post-dominance.
/// - On a block without it, facts are relative to the cut. A consumer that
///   moves code must refuse a source or a target there.
///
/// # Queries
///
/// Every table is dense, indexed by [`DenseId::index`], and sized by the
/// graph's node bound when the tree was computed, so
/// [`post_dominates`](Self::post_dominates) is an O(1) interval test.
/// Querying a block beyond that bound panics.
///
/// # Examples
///
/// ```
/// use cfglib::{Cfg, EdgeKind, PostDominatorTree, PostNode};
///
/// let mut cfg = Cfg::<u32>::new();
/// let branch = cfg.entry();
/// let left = cfg.new_block();
/// let right = cfg.new_block();
/// let merge = cfg.new_block();
/// cfg.add_edge(branch, left, EdgeKind::ConditionalTrue);
/// cfg.add_edge(branch, right, EdgeKind::ConditionalFalse);
/// cfg.add_edge(left, merge, EdgeKind::Fallthrough);
/// cfg.add_edge(right, merge, EdgeKind::Fallthrough);
///
/// let post = PostDominatorTree::compute(&cfg);
/// assert_eq!(post.exits(), [merge]);
/// assert_eq!(post.ipdom(branch), Some(PostNode::Node(merge)));
/// assert_eq!(post.ipdom(merge), Some(PostNode::Exit));
/// assert!(post.post_dominates(merge, left));
/// assert!(!post.post_dominates(left, branch));
/// assert_eq!(
///     post.nearest_common_post_dominator(left, right),
///     Some(PostNode::Node(merge)),
/// );
/// ```
///
/// A branch whose one arm spins forever gets a cut, so the block after the
/// branch does not post-dominate it:
///
/// ```
/// use cfglib::{Cfg, EdgeKind, PostDominatorTree, PostNode};
///
/// let mut cfg = Cfg::<u32>::new();
/// let entry = cfg.entry();
/// let spin = cfg.new_block();
/// let out = cfg.new_block();
/// cfg.add_edge(entry, spin, EdgeKind::ConditionalTrue);
/// cfg.add_edge(entry, out, EdgeKind::ConditionalFalse);
/// cfg.add_edge(spin, spin, EdgeKind::Unconditional);
///
/// let post = PostDominatorTree::compute(&cfg);
/// assert_eq!(post.exits(), [out]);
/// assert_eq!(post.cut_roots(), [spin]);
/// assert!(post.reaches_exit(entry));
/// assert!(!post.reaches_exit(spin));
/// assert_eq!(post.ipdom(entry), Some(PostNode::Exit));
/// assert!(!post.post_dominates(out, entry));
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PostDominatorTree<N = BlockId> {
    /// The forest over the blocks the entry reaches, whose roots are the
    /// children of the virtual exit.
    tree: DominatorTree<N>,
    /// Whether each block has a path to an exit, `false` for a block the
    /// entry misses.
    reaches_exit: Vec<bool>,
    /// The reachable blocks without successors, in ascending dense index.
    exits: Vec<N>,
    /// The blocks the cut joins to the virtual exit, in ascending dense index.
    cut_roots: Vec<N>,
}

impl<N: DenseId> PostDominatorTree<N> {
    /// Compute the post-dominator tree of the blocks a rooted graph view
    /// reaches from its root.
    ///
    /// The running time is O(m log n) for a graph of n blocks and m edges, and
    /// the bound holds on irreducible graphs and on graphs without an exit.
    ///
    /// # Panics
    ///
    /// Panics when the root is not a node of `graph`, or when
    /// [`node_bound`](GraphView::node_bound) plus one for the virtual exit
    /// exceeds the range of a dense `u32` table.
    #[must_use]
    pub fn compute<G>(graph: &G) -> Self
    where
        G: RootedView<NodeId = N>,
    {
        Self::compute_in(&mut PostDominatorScratch::new(), graph)
    }

    /// Compute the post-dominator tree over caller-owned working storage.
    ///
    /// This is [`compute`](Self::compute) with the call's buffers taken out of
    /// it. The tree it returns is the same one, built by the same code: the
    /// allocating entry point is this function over a fresh
    /// [`PostDominatorScratch`].
    ///
    /// # Panics
    ///
    /// Panics under the conditions of [`compute`](Self::compute).
    #[must_use]
    pub fn compute_in<G>(scratch: &mut PostDominatorScratch, graph: &G) -> Self
    where
        G: RootedView<NodeId = N>,
    {
        let bound = graph.node_bound();
        // The virtual exit takes the index `bound`, which must stay below the
        // sentinel.
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
        let structure = cut.analyze(graph, root);
        let augmented = AugmentedReverse::new(graph, cut, &structure);
        let mut parent = dominators
            .construction
            .immediate_dominators(&augmented, bound);
        // The virtual exit is reached and is not a block.
        let reached = dominators.construction.reached() - 1;

        // The virtual exit is the last index and has no parent, so dropping it
        // leaves a table over the blocks. Its children become the roots.
        parent.truncate(bound);
        let virtual_exit = word(bound);
        let mut roots = Vec::new();
        for (block, up) in parent.iter_mut().enumerate() {
            if *up == virtual_exit {
                *up = ABSENT;
                roots.push(word(block));
            }
        }
        debug_assert_eq!(
            reached,
            cut.reachable.iter().filter(|&&reachable| reachable).count(),
            "the cut leaves a reachable block without a path to the virtual exit"
        );

        let numbered = dominators.numbering.number(&parent, &roots, reached);
        let ExitStructure {
            reaches_exit,
            exits,
            cut_roots,
        } = structure;
        let tree = Self {
            tree: DominatorTree::from_numbered(parent, numbered),
            reaches_exit,
            exits,
            cut_roots,
        };
        debug_assert!(
            tree.satisfies_parent_property(graph),
            "an edge ends at a block that its tail's immediate post-dominator does not post-dominate"
        );
        tree
    }

    /// Whether the parent property holds on `graph`: every edge out of a
    /// block in the tree ends at a block that the tail's immediate
    /// post-dominator post-dominates, unless that is the virtual exit.
    ///
    /// The check costs one interval test per edge and backs the debug
    /// assertion that follows every construction.
    fn satisfies_parent_property<G: GraphView<NodeId = N>>(&self, graph: &G) -> bool {
        self.preorder().iter().all(|&tail| {
            let Some(PostNode::Node(parent)) = self.ipdom(tail) else {
                return true;
            };
            graph
                .successors(tail)
                .all(|head| self.post_dominates(parent, head))
        })
    }

    /// Return whether `block` is in the tree, which means the entry reaches
    /// it. A block outside the tree has no post-dominance facts.
    #[must_use]
    pub fn contains(&self, block: N) -> bool {
        self.tree.is_reachable(block)
    }

    /// Return the immediate post-dominator of `block`: the nearest block that
    /// every path from it to the exit passes through, or [`PostNode::Exit`]
    /// when there is none.
    ///
    /// The answer is `None` exactly when [`contains`](Self::contains) is
    /// `false`.
    #[must_use]
    pub fn ipdom(&self, block: N) -> Option<PostNode<N>> {
        self.contains(block).then(|| lift(self.tree.idom(block)))
    }

    /// Return whether `post_dominator` post-dominates `block`, in constant
    /// time.
    ///
    /// A block post-dominates itself, whether the tree contains it or not.
    /// Any other pair with a block outside the tree answers `false`, and so
    /// do two blocks in different trees of the forest, because the blocks
    /// the virtual exit joins do not post-dominate each other.
    #[must_use]
    pub fn post_dominates(&self, post_dominator: N, block: N) -> bool {
        self.tree.dominates(post_dominator, block)
    }

    /// Return whether `post_dominator` post-dominates `block` and is a
    /// different block.
    #[must_use]
    pub fn strictly_post_dominates(&self, post_dominator: N, block: N) -> bool {
        self.tree.strictly_dominates(post_dominator, block)
    }

    /// Return the deepest node that post-dominates both `first` and `second`.
    ///
    /// The walk climbs from the deeper block until the two meet, so it costs
    /// O(depth). The answer is [`PostNode::Exit`] when the two sit in
    /// different trees of the forest, and `None` when either is outside the
    /// tree.
    #[must_use]
    pub fn nearest_common_post_dominator(&self, first: N, second: N) -> Option<PostNode<N>> {
        (self.contains(first) && self.contains(second))
            .then(|| lift(self.tree.nearest_common_dominator(first, second)))
    }

    /// Return the deepest node that post-dominates every block of `blocks`
    /// that the tree contains.
    ///
    /// Blocks outside the tree are skipped. The answer is `None` when no
    /// block is in the tree, and [`PostNode::Exit`] when the blocks share no
    /// tree of the forest.
    #[must_use]
    pub fn nearest_common_post_dominator_of(
        &self,
        blocks: impl IntoIterator<Item = N>,
    ) -> Option<PostNode<N>> {
        let mut meet = None;
        for block in blocks.into_iter().filter(|&block| self.contains(block)) {
            meet = Some(match meet {
                None => PostNode::Node(block),
                Some(PostNode::Node(previous)) => {
                    lift(self.tree.nearest_common_dominator(previous, block))
                }
                Some(PostNode::Exit) => return Some(PostNode::Exit),
            });
        }
        meet
    }

    /// Return whether some path leads from `block` to an exit.
    ///
    /// This is `false` for a block outside the tree, and for a block of an
    /// exit-free region, whose facts are relative to the cut.
    #[must_use]
    pub fn reaches_exit(&self, block: N) -> bool {
        self.reaches_exit[block.index()]
    }

    /// Return the reachable blocks without successors, in ascending dense
    /// index.
    #[must_use]
    pub fn exits(&self) -> &[N] {
        &self.exits
    }

    /// Return the blocks the cut joins to the virtual exit, in ascending dense
    /// index.
    ///
    /// A graph whose reachable blocks all reach an exit has none.
    #[must_use]
    pub fn cut_roots(&self) -> &[N] {
        &self.cut_roots
    }

    /// Return every block of the tree in preorder.
    ///
    /// Roots come in ascending dense index, and the children of a block come in
    /// ascending dense index after it, so a block always precedes the blocks
    /// it post-dominates.
    #[must_use]
    pub fn preorder(&self) -> &[N] {
        self.tree.preorder()
    }

    /// Return the blocks `block` post-dominates, itself first, in preorder. A
    /// block outside the tree has none.
    #[must_use]
    pub fn subtree(&self, block: N) -> &[N] {
        self.tree.subtree(block)
    }

    /// Iterate over the blocks whose immediate post-dominator is `node`, in
    /// ascending dense index.
    ///
    /// The children of [`PostNode::Exit`] are the roots of the forest: the
    /// exits, the cut roots, and the blocks whose paths split between them.
    /// The walk skips each child's subtree, so it costs one step per child.
    pub fn children(&self, node: PostNode<N>) -> impl Iterator<Item = N> + '_ {
        let tree = &self.tree;
        let (first, end) = match node {
            PostNode::Node(block) => {
                let span = tree.span(block);
                (span.start + 1, span.end)
            }
            PostNode::Exit => (0, tree.order.len()),
        };
        core::iter::successors((first < end).then_some(first), move |&position| {
            let next = slot(tree.last[tree.order[position].index()]) + 1;
            (next < end).then_some(next)
        })
        .map(move |position| tree.order[position])
    }

    /// Collect the control-dependence sources of `block` into `out`.
    ///
    /// A block `x` is a source when `block` post-dominates a successor of `x`
    /// without strictly post-dominating `x`. This is the dominance frontier of
    /// `block` on the reverse graph, which is [`DominatorTree::frontier`] over
    /// the predecessors of `graph`. `graph` is the graph this tree was
    /// computed from.
    ///
    /// `out` is cleared first and receives each source once, in preorder of
    /// this tree. A block outside the tree has no sources, and is no one's.
    /// A block that branches back to itself is its own source, although
    /// [`control_dependence_graph`](crate::control_dependence_graph) leaves
    /// that single self-edge out.
    ///
    /// # Panics
    ///
    /// Panics when `block`, or a predecessor of a block of its subtree, has
    /// an index beyond the node bound the tree was computed with.
    ///
    /// # Examples
    ///
    /// ```
    /// use cfglib::{Cfg, EdgeKind, PostDominatorTree};
    ///
    /// let mut cfg = Cfg::<u32>::new();
    /// let branch = cfg.entry();
    /// let left = cfg.new_block();
    /// let right = cfg.new_block();
    /// let merge = cfg.new_block();
    /// cfg.add_edge(branch, left, EdgeKind::ConditionalTrue);
    /// cfg.add_edge(branch, right, EdgeKind::ConditionalFalse);
    /// cfg.add_edge(left, merge, EdgeKind::Fallthrough);
    /// cfg.add_edge(right, merge, EdgeKind::Fallthrough);
    /// let post = PostDominatorTree::compute(&cfg);
    ///
    /// let mut sources = Vec::new();
    /// // Taking the left arm is decided at the branch; the merge is not.
    /// post.frontier(&cfg, left, &mut sources);
    /// assert_eq!(sources, [branch]);
    /// post.frontier(&cfg, merge, &mut sources);
    /// assert!(sources.is_empty());
    /// ```
    pub fn frontier<G>(&self, graph: &G, block: N, out: &mut Vec<N>)
    where
        G: GraphView<NodeId = N>,
    {
        self.tree.frontier(&Reversed::new(graph), block, out);
    }

    /// Collect the iterated control-dependence frontier of `blocks` into
    /// `out`.
    ///
    /// This is [`DominatorTree::iterated_frontier`] on the reverse graph: the
    /// blocks whose branches decide whether a block of `blocks` executes,
    /// directly or through other such branches. It is where a backward
    /// analysis places its merges, and the blocks a dead-code pass keeps
    /// alive for the blocks it keeps.
    ///
    /// With `filter`, the result is pruned the way liveness prunes phi
    /// placement: a block is added, and its own frontier explored, only when
    /// `filter` accepts it. A rejected block never enters the answer, and a
    /// block of `blocks` is a root of the search whether `filter` accepts it
    /// or not.
    ///
    /// `out` is cleared first and receives each block once, in preorder of
    /// this tree. A block of `blocks` outside the tree is ignored, and so is
    /// one named twice.
    ///
    /// # Panics
    ///
    /// Panics when a block of `blocks`, or a predecessor of a block the search
    /// walks, has an index beyond the node bound the tree was computed with.
    ///
    /// # Examples
    ///
    /// ```
    /// use cfglib::{Cfg, EdgeKind, FrontierScratch, PostDominatorTree};
    ///
    /// let mut cfg = Cfg::<u32>::new();
    /// let branch = cfg.entry();
    /// let left = cfg.new_block();
    /// let right = cfg.new_block();
    /// let merge = cfg.new_block();
    /// cfg.add_edge(branch, left, EdgeKind::ConditionalTrue);
    /// cfg.add_edge(branch, right, EdgeKind::ConditionalFalse);
    /// cfg.add_edge(left, merge, EdgeKind::Fallthrough);
    /// cfg.add_edge(right, merge, EdgeKind::Fallthrough);
    /// let post = PostDominatorTree::compute(&cfg);
    ///
    /// let mut scratch = FrontierScratch::new();
    /// let mut kept = Vec::new();
    /// post.iterated_frontier(&cfg, &mut scratch, [left, right], None, &mut kept);
    /// assert_eq!(kept, [branch]);
    ///
    /// let nothing_live = |_| false;
    /// post.iterated_frontier(&cfg, &mut scratch, [left], Some(&nothing_live), &mut kept);
    /// assert!(kept.is_empty());
    /// ```
    pub fn iterated_frontier<G>(
        &self,
        graph: &G,
        scratch: &mut FrontierScratch,
        blocks: impl IntoIterator<Item = N>,
        filter: Option<&dyn Fn(N) -> bool>,
        out: &mut Vec<N>,
    ) where
        G: GraphView<NodeId = N>,
    {
        self.tree
            .iterated_frontier(&Reversed::new(graph), scratch, blocks, filter, out);
    }
}

/// The node an optional parent names: `None` is the virtual exit.
fn lift<N>(parent: Option<N>) -> PostNode<N> {
    parent.map_or(PostNode::Exit, PostNode::Node)
}
