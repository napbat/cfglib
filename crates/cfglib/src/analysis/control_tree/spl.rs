//! The series-parallel-loop decomposition of a structured control tree, and
//! the path decomposition read off it. [`SplDecomposition`] documents the
//! grammar; `build` makes the decomposition from the tree, `expand` checks it
//! against the graph by the grammar's CFG construction, and `path` reads the
//! path decomposition.

extern crate alloc;
use alloc::vec::Vec;
use core::fmt;

use super::node::ControlNodeId;
use super::tree::ControlTree;
use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::edge::EdgeId;
use crate::graph::view::DenseId;
use crate::identity::define_dense_id;

mod build;
mod expand;
mod path;

pub use path::PathDecomposition;

define_dense_id! {
    /// The handle of one node of an [`SplDecomposition`].
    ///
    /// A handle is a dense index into the decomposition that produced it.
    pub struct SplNodeId(u32);
    display = "spl";
    /// Create a handle from a dense zero-based index.
    ///
    /// # Panics
    ///
    /// Panics when `index` exceeds `u32::MAX`.
    from_index = "SPL node index exceeds u32::MAX";
}

impl DenseId for SplNodeId {
    fn from_index(index: usize) -> Self {
        Self::from_index(index)
    }

    fn index(self) -> usize {
        self.index()
    }
}

/// The ordered children of an [`SplNode`], as a run of one table of the
/// [`SplDecomposition`] that holds it.
///
/// [`SplDecomposition::children`] turns a run into its slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SplRun {
    start: u32,
    len: u32,
}

impl SplRun {
    /// The number of children.
    #[must_use]
    pub const fn len(self) -> usize {
        self.len as usize
    }

    /// Whether the run has no children.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.len == 0
    }
}

/// One node of an [`SplDecomposition`]: a production of the grammar.
///
/// [`SplDecomposition`] tells what each node does with the edges that enter
/// and leave it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SplNode {
    /// One block of the graph, including the blocks the control tree has no
    /// leaf for: empty blocks, and a loop's test blocks.
    Block(BlockId),
    /// Series composition, in execution order. An empty series is `ε`.
    Series {
        /// The statements.
        children: SplRun,
    },
    /// Parallel composition of the arms of the block before it: `if / else`,
    /// or a `switch` with one arm per target.
    Parallel {
        /// The block that branches, whose leaf comes right before this node.
        branch: BlockId,
        /// One arm per target of `branch`, each entered over its edges.
        arms: SplRun,
    },
    /// A loop that every iteration enters at `header`.
    Loop {
        /// The block each iteration begins with, the first leaf of `body`.
        header: BlockId,
        /// The body, which holds the loop's tests as branches with a `Break`
        /// arm.
        body: SplNodeId,
    },
    /// Leave the innermost loop.
    Break,
    /// Go on with the next iteration of the innermost loop, at its header.
    Continue,
}

/// Why a control tree has no [`SplDecomposition`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SplError {
    /// An unstructured transfer.
    Goto(ControlNodeId),
    /// A label that is not just the name of a loop.
    Label(ControlNodeId),
    /// A protected region with its handlers.
    TryCatch(ControlNodeId),
    /// A `break` or `continue` that leaves more than the innermost loop.
    MultiLevelJump(ControlNodeId),
    /// An exit block that does not end the procedure: a `return` other than
    /// the last statement, or one of several exit blocks.
    EarlyReturn(BlockId),
    /// A block no leaf can hold: one the root does not reach, or one without
    /// code that flow enters from more than one construct.
    UnplacedBlock(BlockId),
    /// An edge the grammar cannot make from the tree, such as an exception
    /// edge, or a `continue` that skips code before a post-tested loop's test.
    UnplacedEdge(EdgeId),
}

impl fmt::Display for SplError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Goto(node) => write!(formatter, "{node} is a goto"),
            Self::Label(node) => write!(formatter, "{node} is a label"),
            Self::TryCatch(node) => write!(formatter, "{node} is a try region"),
            Self::MultiLevelJump(node) => {
                write!(formatter, "{node} leaves more than the innermost loop")
            }
            Self::EarlyReturn(block) => {
                write!(formatter, "{block} exits before the end of the procedure")
            }
            Self::UnplacedBlock(block) => write!(formatter, "no leaf can hold {block}"),
            Self::UnplacedEdge(edge) => write!(formatter, "the grammar cannot make {edge}"),
        }
    }
}

impl core::error::Error for SplError {}

/// The series-parallel-loop decomposition of a graph: a parse tree of its
/// blocks under the grammar of structured goto-free programs.
///
/// Cai, Goharshady, Hitarth and Lam ("Series-Parallel-Loop Decompositions of
/// Control-flow Graphs", Journal of Systems Architecture, 2026) show that the
/// control-flow graphs of structured goto-free programs are exactly the graphs
/// of the grammar
///
/// ```text
/// P := ε | break | continue | P; P | if φ then P else P fi | while φ do P od
/// ```
///
/// under its CFG construction, and run dynamic programs (register allocation,
/// lifetime-optimal speculative PRE, partial constraint satisfaction)
/// bottom-up over the parse tree. [`ControlTree::spl_decomposition`] gives
/// that parse tree for a [`Cfg`] whose [`ControlTree`] is such a program.
///
/// The nodes live in one flat arena, children before their parents, and each
/// node's children are a [`SplRun`] of one table, as in [`ControlTree`].
///
/// # The grammar over blocks
///
/// A block is a statement, so every block of the graph is in exactly one
/// [`SplNode::Block`] leaf. A fragment of the decomposition is entered at its
/// first block and leaves through the edges that fall out of it, that break,
/// and that continue:
///
/// - [`SplNode::Block`] `b`: every edge falling into the fragment ends at `b`;
///   the edges out of `b` fall out.
/// - [`SplNode::Series`]: the edges that fall out of each child end at the
///   next child; an empty series (`ε`) lets them fall through.
/// - [`SplNode::Parallel`] follows the leaf of its `branch` block: each arm
///   takes one edge of `branch`, and what falls out of an arm falls out of
///   the whole. A `switch` is one `Parallel` with an arm per target, which is
///   the nested `if`s of the paper, since parallel composition is associative.
/// - [`SplNode::Loop`]: its body begins with the leaf of its `header`. The
///   edges that fall out of the body, and its continues, end at the header;
///   its breaks fall out of the loop.
/// - [`SplNode::Break`] and [`SplNode::Continue`] turn the edges falling into
///   them into breaks or continues.
///
/// Parallel edges between two blocks count once, as the grammar's parallel
/// composition merges them.
///
/// The grammar's `while φ` tests at the top of every iteration and has an
/// edge out of the loop at the test. A loop here is `while (true)`, which the
/// CFG has no such edge for, and its tests are branches with a `Break` arm: a
/// pre-tested loop tests at the end of the header's condition chain, a
/// post-tested loop at its latch, after the body, and an endless loop
/// nowhere. That is the grammar's `while` with the never-taken exit left out.
///
/// # What has no decomposition
///
/// The grammar has no `goto`, no labeled loop, no exception handling, and one
/// exit, at the end of the procedure. A tree with a `Goto`, a `Label`, a
/// `TryCatch`, a `break` or `continue` that leaves more than the innermost
/// loop, or a `Return` other than the last statement of the procedure has no
/// decomposition, and nor has a graph with more than one exit block or with a
/// block the root does not reach. [`SplError`] names the first of these.
///
/// The tree passes through a block without code, so such a block gets its
/// leaf where flow enters it, as late as the edges allow. One that edges from
/// two constructs enter, such as a tail that arms at two depths share, has no
/// place, and an exit without code that only a path inside a loop or one arm
/// of a branch reaches is an early return, as a `Return` there is.
///
/// # Width
///
/// [`path_decomposition`](Self::path_decomposition) turns the decomposition
/// into a path decomposition of the graph whose width is at most `2·d + 1`,
/// where `d` is the [`nesting_depth`](Self::nesting_depth): the deepest chain
/// of nested `Parallel` and `Loop` nodes. A straight line has width 1 and an
/// `if / else` diamond width 2. Conrado, Goharshady and Lam ("The Bounded
/// Pathwidth of Control-Flow Graphs", OOPSLA 2023) bound the pathwidth of a
/// program by twice its nesting depth; the bound here comes from the same
/// idea, driven by the decomposition rather than by source syntax: each
/// nested construct keeps its entry or branch block and the block its
/// statements fall to in every bag inside it.
///
/// # Examples
///
/// ```
/// use cfglib::{Cfg, ControlTree, EdgeKind, SplNode};
///
/// // entry; if (test) { then } else { other }; merge
/// let mut cfg = Cfg::<&str>::new();
/// let then_block = cfg.new_block();
/// let other = cfg.new_block();
/// let merge = cfg.new_block();
/// for (block, text) in [(cfg.entry(), "test"), (then_block, "then"), (other, "other"), (merge, "merge")] {
///     cfg.block_mut(block).push(text);
/// }
/// cfg.add_edge(cfg.entry(), then_block, EdgeKind::ConditionalTrue);
/// cfg.add_edge(cfg.entry(), other, EdgeKind::ConditionalFalse);
/// cfg.add_edge(then_block, merge, EdgeKind::Fallthrough);
/// cfg.add_edge(other, merge, EdgeKind::Fallthrough);
///
/// let tree = ControlTree::compute(&cfg);
/// let spl = tree.spl_decomposition(&cfg).expect("an if/else is structured");
/// assert_eq!(spl.blocks(), [cfg.entry(), then_block, other, merge]);
/// assert_eq!(spl.nesting_depth(), 1);
///
/// let SplNode::Series { children } = *spl.node(spl.root()) else {
///     panic!("the procedure is a sequence");
/// };
/// assert!(matches!(spl.node(spl.children(children)[1]), SplNode::Parallel { .. }));
///
/// // An if/else diamond has pathwidth 2.
/// assert_eq!(spl.path_decomposition().width(), 2);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SplDecomposition {
    /// Every node, children before their parents.
    nodes: Vec<SplNode>,
    /// The children of every node with children, one run each.
    children: Vec<SplNodeId>,
    /// The node of the whole procedure.
    root: SplNodeId,
    /// The block of every leaf, in preorder.
    blocks: Vec<BlockId>,
    /// The deepest chain of nested `Parallel` and `Loop` nodes.
    nesting_depth: usize,
}

impl SplDecomposition {
    /// The node of the whole procedure.
    #[must_use]
    pub const fn root(&self) -> SplNodeId {
        self.root
    }

    /// The node `id` names.
    ///
    /// # Panics
    ///
    /// Panics when `id` is not a node of this decomposition.
    #[must_use]
    pub fn node(&self, id: SplNodeId) -> &SplNode {
        &self.nodes[id.index()]
    }

    /// The number of nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// The nodes a run names, in order.
    ///
    /// # Panics
    ///
    /// Panics when `run` is not a run of this decomposition.
    #[must_use]
    pub fn children(&self, run: SplRun) -> &[SplNodeId] {
        let start = run.start as usize;
        &self.children[start..start + run.len()]
    }

    /// Every block of the graph, once, in the order of the leaves in
    /// preorder.
    #[must_use]
    pub fn blocks(&self) -> &[BlockId] {
        &self.blocks
    }

    /// The deepest chain of nested `Parallel` and `Loop` nodes: `0` for a
    /// straight line.
    #[must_use]
    pub const fn nesting_depth(&self) -> usize {
        self.nesting_depth
    }

    /// A path decomposition of the graph, as bags in order, of width at most
    /// `2·`[`nesting_depth`](Self::nesting_depth)`+ 1`.
    ///
    /// Every edge has both blocks in one bag, and the bags holding a block are
    /// consecutive. The bags hold O(n·d) blocks in all, for n blocks.
    #[must_use]
    pub fn path_decomposition(&self) -> PathDecomposition {
        path::decompose(self)
    }
}

impl ControlTree {
    /// The series-parallel-loop decomposition of `cfg`, the graph this tree
    /// was computed from, or the first construct with no counterpart in the
    /// grammar.
    ///
    /// The decomposition follows the tree's statements in their order. It
    /// adds the blocks the tree has no leaf for: a loop's condition chain and
    /// latch as the loop's tests, and every block without code where flow
    /// enters it. The result is checked against `cfg`: the grammar's CFG
    /// construction gives back exactly its blocks and its edges. That costs
    /// O(n + m) for n blocks and m edges, as does building it.
    ///
    /// # Errors
    ///
    /// Returns the [`SplError`] of the first construct in preorder with no
    /// counterpart in the grammar; see [`SplDecomposition`].
    ///
    /// # Panics
    ///
    /// Panics when `cfg` is not the graph the tree was computed from.
    pub fn spl_decomposition<I, E>(&self, cfg: &Cfg<I, E>) -> Result<SplDecomposition, SplError> {
        let decomposition = build::build(self, cfg)?;
        expand::verify(&decomposition, cfg)?;
        Ok(decomposition)
    }
}
