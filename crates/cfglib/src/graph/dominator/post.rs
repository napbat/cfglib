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
//! exit-free region a path to the virtual exit, `sink` reads exit-free sinks
//! as classes instead, and this module runs the simple Lengauer-Tarjan
//! construction on the result and holds the tree and its queries. The graph
//! the construction reads is composed from the shared views: the reverse of
//! the reachable blocks behind a virtual root,
//! `VirtualRoot::masked(&Reversed::new(&Induced::new(graph, reachable)), outlets, is_outlet)`,
//! so no node, edge, or adjacency list is copied to build it.

extern crate alloc;
use alloc::vec::Vec;

use super::construction::{ConstructionBuffers, LowHighBuffers};
use super::frontier_table::sealed::Tabulate;
use super::numbering::NumberingBuffers;
use super::{DominanceFrontiers, DominatorScratch, DominatorTree, FrontierTree, LowHighOrder};
use crate::block::BlockId;
use crate::graph::view::{DenseId, GraphView, Induced, Reversed, RootedView, VirtualRoot};
use crate::identity::{NO_INDEX, slot, word};
use certificate::without_virtual_exit;
pub(crate) use cut::{CutBuffers, ExitStructure};
use sink::{Contracted, SinkClasses};
pub use sink::{ExitPolicy, SinkId};

mod certificate;
mod cut;
mod edge_dominance;
mod sink;

/// A node of a post-dominator tree: a block of the graph, or the virtual exit
/// that every exit joins.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PostNode<N = BlockId> {
    /// A block of the graph.
    Node(N),
    /// The virtual exit `X*`, the root of the tree.
    Exit,
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
/// # Sink classes
///
/// [`compute_in`](Self::compute_in) with
/// [`ExitPolicy::SinkComponents`] reads an exit-free region the other
/// standard way: each sink component of it, a strongly connected component
/// no edge leaves, is one class joined to `X*`, and has no cut. The blocks of
/// a class post-dominate each other, and a block outside every sink
/// post-dominates another when every path from that one into the sinks passes
/// it. This is the sink post-dominance of Hecker et al., under which control
/// dependence is nontermination-insensitive.
///
/// A class is one node of the tree, named by its lowest block:
/// [`preorder`](Self::preorder), [`subtree`](Self::subtree) and
/// [`children`](Self::children) list that block for the class, and
/// [`sink_members`](Self::sink_members) expands it. The other queries take
/// any block of a class: [`ipdom`](Self::ipdom) of each is
/// [`PostNode::Exit`], [`control_dependents`](Self::control_dependents)
/// skips an arm between two blocks of one class and lists every block of a
/// class that depends, and the frontier queries read an edge into a class as
/// an edge into the class.
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
    /// The sink classes, none for every tree of the cut.
    classes: SinkClasses<N>,
}

impl<N: DenseId> PostDominatorTree<N> {
    /// Compute the post-dominator tree of the blocks a rooted graph view
    /// reaches from its root, under [`ExitPolicy::CutLoops`].
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
        Self::compute_in(
            &mut DominatorScratch::default(),
            graph,
            ExitPolicy::CutLoops,
        )
    }

    /// Compute the post-dominator tree under `policy` over caller-owned
    /// working storage.
    ///
    /// [`ExitPolicy::CutLoops`] builds exactly the tree of
    /// [`compute`](Self::compute): the allocating entry point is this
    /// function over a fresh [`DominatorScratch`]. Debug builds also build a
    /// low-high order of that tree and check it with
    /// [`verify_certificate`](Self::verify_certificate).
    /// [`ExitPolicy::SinkComponents`] makes each sink component of the blocks
    /// that reach no exit one class joined to the virtual exit (see
    /// [`sink_class`](Self::sink_class)), and has no cut roots. On a graph
    /// whose reachable blocks all reach an exit the two trees are `==`. Both
    /// cost the same O(m log n) construction plus O(n + m) to find the exits
    /// and the cut or the sinks.
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
    /// use cfglib::{Cfg, DominatorScratch, EdgeKind, ExitPolicy, PostDominatorTree};
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
    /// let mut scratch = DominatorScratch::default();
    /// let cut = PostDominatorTree::compute_in(&mut scratch, &cfg, ExitPolicy::CutLoops);
    /// assert_eq!(cut, PostDominatorTree::compute(&cfg));
    /// assert_eq!(cut.cut_roots(), [latch]);
    ///
    /// let post = PostDominatorTree::compute_in(&mut scratch, &cfg, ExitPolicy::SinkComponents);
    /// let class = post.sink_class(ok).expect("the loop is a sink");
    /// assert_eq!(post.sink_members(class), [head, ok, latch]);
    /// assert!(post.post_dominates(ok, latch) && post.post_dominates(latch, ok));
    /// assert_eq!(post.control_dependents(head, ok).count(), 0);
    /// assert!(post.cut_roots().is_empty());
    /// ```
    #[must_use]
    pub fn compute_in<G>(scratch: &mut DominatorScratch, graph: &G, policy: ExitPolicy) -> Self
    where
        G: RootedView<NodeId = N>,
    {
        if cfg!(debug_assertions) && policy == ExitPolicy::CutLoops {
            let (tree, certificate) = Self::compute_certified_in(scratch, graph);
            debug_assert_eq!(
                tree.verify_certificate(graph, &certificate),
                Ok(()),
                "the construction built a tree its own certificate rejects"
            );
            return tree;
        }
        Self::construct(scratch, graph, policy, None)
    }

    /// Build the tree under `policy`, and fill `certificate`, which only the
    /// cut asks for, when one is asked for.
    fn construct<G>(
        scratch: &mut DominatorScratch,
        graph: &G,
        policy: ExitPolicy,
        certificate: Option<&mut LowHighOrder<N>>,
    ) -> Self
    where
        G: RootedView<NodeId = N>,
    {
        let bound = graph.node_bound();
        // The virtual exit takes the index `bound`, which must stay below the
        // sentinel.
        assert!(
            bound < slot(NO_INDEX),
            "node bound plus the virtual exit exceeds the dense u32 range of a post-dominator tree"
        );
        let root = graph.root();
        assert!(
            root.index() < bound,
            "post-dominator root is not a node of the graph"
        );

        let DominatorScratch {
            construction,
            numbering,
            low_high,
            cut,
            ..
        } = scratch;
        let (structure, classes) = match policy {
            ExitPolicy::CutLoops => (cut.analyze(graph, root), SinkClasses::default()),
            ExitPolicy::SinkComponents => cut.analyze_sinks(graph, root),
        };
        let mut buffers = Buffers {
            construction,
            numbering,
            low_high,
            cut,
        };
        let tree = if classes.is_empty() {
            buffers.forest(graph, &structure.outlets, certificate)
        } else {
            let contracted = Contracted::new(graph, &classes);
            let mut tree = buffers.forest(&contracted, &structure.outlets, None);
            classes.share_numbering(&mut tree);
            tree
        };
        debug_assert_eq!(
            tree.order.len() + classes.hidden(),
            cut.reachable.iter().filter(|&&reachable| reachable).count(),
            "a reachable block has no path to the virtual exit"
        );

        let ExitStructure {
            reaches_exit,
            exits,
            cut_roots,
            ..
        } = structure;
        Self {
            tree,
            reaches_exit,
            exits,
            cut_roots,
            classes,
        }
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
    /// tree. A sink class is named by its lowest block.
    #[must_use]
    pub fn nearest_common_post_dominator(&self, first: N, second: N) -> Option<PostNode<N>> {
        (self.contains(first) && self.contains(second)).then(|| {
            lift(self.tree.nearest_common_dominator(
                self.classes.canonical(first),
                self.classes.canonical(second),
            ))
        })
    }

    /// Return the deepest node that post-dominates every block of `blocks`
    /// that the tree contains.
    ///
    /// Blocks outside the tree are skipped. The answer is `None` when no
    /// block is in the tree, and [`PostNode::Exit`] when the blocks share no
    /// tree of the forest. A sink class is named by its lowest block.
    #[must_use]
    pub fn nearest_common_post_dominator_among(
        &self,
        blocks: impl IntoIterator<Item = N>,
    ) -> Option<PostNode<N>> {
        let mut any = false;
        let meet = self.tree.nearest_common_dominator_among(
            blocks
                .into_iter()
                .filter(|&block| self.contains(block))
                .map(|block| {
                    any = true;
                    self.classes.canonical(block)
                }),
        );
        any.then(|| lift(meet))
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
        self.tree.below(match node {
            PostNode::Node(block) => Some(block),
            PostNode::Exit => None,
        })
    }

    /// Iterate over the blocks that are control dependent on the arm
    /// `from -> to`.
    ///
    /// A block `w` is control dependent on the arm when `w` post-dominates `to`
    /// and does not strictly post-dominate `from`. These are `to`, then its
    /// immediate post-dominator, and so on up the tree, stopping before the
    /// immediate post-dominator of `from`. The nearest block comes first, and
    /// nothing comes when `to` is that immediate post-dominator.
    ///
    /// The arm `u -> u` of a block that branches back to itself yields `u`
    /// first, because a block does not strictly post-dominate itself.
    ///
    /// The answer is empty when `from` or `to` is outside the tree, and when
    /// the immediate post-dominator of `from` is not a strict post-dominator
    /// of `to`. For an edge of the graph the tree was computed from, that
    /// last case is exactly `to` being the immediate post-dominator, because
    /// the parent property puts every successor at or below the tail's
    /// immediate post-dominator. The method does not check that the edge
    /// exists, so it costs O(1 + the number of blocks it yields).
    ///
    /// Under [`ExitPolicy::SinkComponents`] a block of a sink class is read as
    /// the class: an arm between two blocks of one class yields nothing, and
    /// a class that depends yields all its blocks, ascending.
    ///
    /// # Panics
    ///
    /// Panics when `from` or `to` has an index beyond the node bound the tree
    /// was computed with, as [`contains`](Self::contains) does.
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
    /// // The branch decides whether the left block runs, and the merge is
    /// // reached either way.
    /// assert_eq!(post.control_dependents(branch, left).collect::<Vec<_>>(), [left]);
    /// assert_eq!(post.control_dependents(left, merge).count(), 0);
    /// ```
    ///
    /// A block that loops back to itself decides whether it runs again, so the
    /// arm into itself is one of its own dependences:
    ///
    /// ```
    /// use cfglib::{Cfg, EdgeKind, PostDominatorTree};
    ///
    /// let mut cfg = Cfg::<u32>::new();
    /// let head = cfg.entry();
    /// let exit = cfg.new_block();
    /// cfg.add_edge(head, head, EdgeKind::ConditionalTrue);
    /// cfg.add_edge(head, exit, EdgeKind::ConditionalFalse);
    /// let post = PostDominatorTree::compute(&cfg);
    ///
    /// assert_eq!(post.control_dependents(head, head).collect::<Vec<_>>(), [head]);
    /// assert_eq!(post.control_dependents(head, exit).count(), 0);
    /// ```
    #[inline]
    pub fn control_dependents(&self, from: N, to: N) -> impl Iterator<Item = N> + '_ {
        let classes = &self.classes;
        let walk = (!classes.share(from, to))
            .then(|| self.walk_up(classes.canonical(from), classes.canonical(to)));
        ClassWalk {
            classes,
            walk,
            class: &[],
        }
    }

    /// The walk of [`control_dependents`](Self::control_dependents) between
    /// two nodes of the tree: from `to` up to the immediate post-dominator of
    /// `from`, which is left out.
    #[inline]
    fn walk_up(&self, from: N, to: N) -> impl Iterator<Item = N> + '_ {
        let stop = self.ipdom(from);
        let below_stop = self.contains(to)
            && match stop {
                None => false,
                Some(PostNode::Exit) => true,
                Some(PostNode::Node(top)) => self.strictly_post_dominates(top, to),
            };
        core::iter::successors(below_stop.then_some(to), move |&block| {
            match self.ipdom(block) {
                Some(PostNode::Node(parent)) if stop != Some(PostNode::Node(parent)) => {
                    Some(parent)
                }
                _ => None,
            }
        })
    }

    /// Collect the control-dependence sources of `block` into `out`.
    ///
    /// A block `x` is a source when `block` post-dominates a successor of `x`
    /// without strictly post-dominating `x`. This is the dominance frontier of
    /// `block` on the reverse graph, which is [`DominatorTree::frontier`] over
    /// the predecessors of `graph`. `graph` is the graph this tree was
    /// computed from.
    ///
    /// `out` is cleared first and receives each source once, in ascending
    /// dense index. A block outside the tree has no sources, and is no one's.
    /// A block that branches back to itself is its own source. The edges of
    /// [`control_dependence_graph`](crate::control_dependence_graph) are this
    /// same relation: it has the edge `x -> block` exactly when `x` is a
    /// source of `block`. Under [`ExitPolicy::SinkComponents`] a block of a
    /// sink class has the sources of its class, and no edge inside a class
    /// makes one.
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
        self.tree.frontier(
            &Reversed::new(&self.contracted(graph)),
            self.classes.canonical(block),
            out,
        );
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
    /// `out` is cleared first and receives each block once, in ascending
    /// dense index. A block of `blocks` outside the tree is ignored, and so is
    /// one named twice. Under [`ExitPolicy::SinkComponents`] a block of a
    /// sink class stands for its class, as in [`frontier`](Self::frontier).
    ///
    /// # Panics
    ///
    /// Panics when a block of `blocks`, or a predecessor of a block the search
    /// walks, has an index beyond the node bound the tree was computed with.
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
    /// let mut kept = Vec::new();
    /// post.iterated_frontier(&cfg, [left, right], None, &mut kept);
    /// assert_eq!(kept, [branch]);
    ///
    /// let nothing_live = |_| false;
    /// post.iterated_frontier(&cfg, [left], Some(&nothing_live), &mut kept);
    /// assert!(kept.is_empty());
    /// ```
    pub fn iterated_frontier<G>(
        &self,
        graph: &G,
        blocks: impl IntoIterator<Item = N>,
        filter: Option<&dyn Fn(N) -> bool>,
        out: &mut Vec<N>,
    ) where
        G: GraphView<NodeId = N>,
    {
        self.iterated_frontier_in(&mut DominatorScratch::default(), graph, blocks, filter, out);
    }

    /// [`iterated_frontier`](Self::iterated_frontier) over caller-owned
    /// working storage, the same answer: a pass that asks for many frontiers
    /// hands one [`DominatorScratch`] to every query, and a query allocates
    /// nothing once the scratch and `out` have grown.
    ///
    /// # Panics
    ///
    /// Panics under the conditions of
    /// [`iterated_frontier`](Self::iterated_frontier).
    pub fn iterated_frontier_in<G>(
        &self,
        scratch: &mut DominatorScratch,
        graph: &G,
        blocks: impl IntoIterator<Item = N>,
        filter: Option<&dyn Fn(N) -> bool>,
        out: &mut Vec<N>,
    ) where
        G: GraphView<NodeId = N>,
    {
        let blocks = blocks
            .into_iter()
            .map(|block| self.classes.canonical(block));
        self.tree.iterated_frontier_in(
            scratch,
            &Reversed::new(&self.contracted(graph)),
            blocks,
            filter,
            out,
        );
    }
}

impl<N: DenseId> FrontierTree<N> for PostDominatorTree<N> {}

/// The table of [`frontier`](PostDominatorTree::frontier): the frontiers of
/// the tree over the reverse of the contracted graph, with every block of a
/// class given the run of the block that names it.
impl<N: DenseId> Tabulate<N> for PostDominatorTree<N> {
    fn tabulate<G: GraphView<NodeId = N>>(&self, graph: &G, table: &mut DominanceFrontiers<N>) {
        table.collect(&Reversed::new(&self.contracted(graph)), &self.tree);
        self.classes.share_runs(&mut table.pairs);
        table.group(graph.node_bound());
    }
}

/// The buffers of one post-dominator construction, borrowed from a
/// [`DominatorScratch`] once the exits and the cut are known.
struct Buffers<'a> {
    construction: &'a mut ConstructionBuffers,
    numbering: &'a mut NumberingBuffers,
    low_high: &'a mut LowHighBuffers,
    cut: &'a CutBuffers,
}

impl Buffers<'_> {
    /// The forest of blocks below the virtual exit of `graph`, the graph or
    /// its contraction, with `outlets` joined to the virtual exit; and a
    /// low-high order of it in `certificate` when one is asked for.
    ///
    /// The construction runs on the reverse of the blocks the cut found
    /// reachable, from a virtual exit with an edge to every outlet. That graph
    /// is the view composition below, whose virtual exit finds its
    /// predecessors through the cut's outlet table in constant time.
    fn forest<V, N>(
        &mut self,
        graph: &V,
        outlets: &[N],
        certificate: Option<&mut LowHighOrder<N>>,
    ) -> DominatorTree<N>
    where
        V: GraphView<NodeId = N>,
        N: DenseId,
    {
        let bound = graph.node_bound();
        let live = Induced::new(graph, &self.cut.reachable);
        let reverse = Reversed::new(&live);
        let augmented = VirtualRoot::masked(&reverse, outlets, &self.cut.exit_edge);
        let parent =
            self.construction
                .immediate_dominators(&augmented, bound, certificate.is_some());
        // The virtual exit is reached and is not a block.
        let reached = self.construction.reached() - 1;
        if let Some(certificate) = certificate {
            // The order is built on the whole tree, the virtual exit at its
            // root, and then read without it.
            let whole = self
                .numbering
                .number::<usize>(&parent, &[word(bound)], reached + 1);
            let mut with_exit = LowHighOrder::empty();
            self.construction.low_high_order(
                &augmented,
                &parent,
                &whole,
                self.low_high,
                &mut with_exit,
            );
            *certificate = without_virtual_exit(&with_exit, bound);
        }
        DominatorTree::from_virtual_root(self.numbering, parent, reached)
    }
}

/// The node an optional parent names: `None` is the virtual exit.
fn lift<N>(parent: Option<N>) -> PostNode<N> {
    parent.map_or(PostNode::Exit, PostNode::Node)
}

/// A walk up the tree with each node that names a sink class expanded into
/// the class's blocks; no walk at all for an arm inside one class. Without
/// classes it is the walk itself.
struct ClassWalk<'a, N, W> {
    classes: &'a SinkClasses<N>,
    walk: Option<W>,
    /// The blocks of the class being expanded that are still to come.
    class: &'a [N],
}

impl<N: DenseId, W: Iterator<Item = N>> Iterator for ClassWalk<'_, N, W> {
    type Item = N;

    #[inline]
    fn next(&mut self) -> Option<N> {
        if let [first, rest @ ..] = self.class {
            self.class = rest;
            return Some(*first);
        }
        let node = self.walk.as_mut()?.next()?;
        if let [first, rest @ ..] = self.classes.named(node) {
            self.class = rest;
            return Some(*first);
        }
        Some(node)
    }
}
