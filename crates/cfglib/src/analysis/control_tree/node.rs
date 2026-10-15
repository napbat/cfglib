//! The vocabulary of a control tree: node handles, run handles, instruction
//! spans, and node kinds.
//!
//! Nothing here holds an instruction. A leaf names the instructions it stands
//! for with a [`BlockSpan`], a reference into the [`Cfg`](crate::Cfg) the tree
//! was computed from.

use core::fmt;
use core::hash::{Hash, Hasher};
use core::marker::PhantomData;
use core::ops::Range;

use crate::block::BlockId;
use crate::dataflow::ProgramPoint;
use crate::dataflow::cfg_view::CfgView;
use crate::edge::EdgeId;
use crate::graph::view::DenseId;
use crate::identity::define_dense_id;
use crate::region::{HandlerKind, HandlerRef};

define_dense_id! {
    /// The handle of one node of a [`ControlTree`](super::ControlTree).
    ///
    /// A handle is a dense index into the tree that produced it, so it sizes
    /// side tables the way [`BlockId`] does. Using it with another tree reads
    /// another tree's node or panics.
    pub struct ControlNodeId(u32);
    display = "node";
    /// Create a handle from a dense zero-based index.
    ///
    /// # Panics
    ///
    /// Panics when `index` exceeds `u32::MAX`.
    from_index = "control tree node index exceeds u32::MAX";
}

impl DenseId for ControlNodeId {
    fn from_index(index: usize) -> Self {
        Self::from_index(index)
    }

    fn index(self) -> usize {
        self.index()
    }
}

/// A contiguous run of `T` stored in one table of a
/// [`ControlTree`](super::ControlTree).
///
/// A run is a handle, not a slice: the tree's accessors ([`body`], [`cases`],
/// [`handlers`], [`edges`], [`spans`]) turn it into the slice it names. The
/// type parameter keeps a run of nodes from being read as a run of edges. A
/// run is meaningful only for the tree that produced it.
///
/// [`body`]: super::ControlTree::body
/// [`cases`]: super::ControlTree::cases
/// [`handlers`]: super::ControlTree::handlers
/// [`edges`]: super::ControlTree::edges
/// [`spans`]: super::ControlTree::spans
pub struct ControlRun<T> {
    start: u32,
    len: u32,
    marker: PhantomData<fn() -> T>,
}

/// The ordered children of a node: the statements of one structured body.
pub type ControlBody = ControlRun<ControlNodeId>;

impl<T> ControlRun<T> {
    pub(super) const fn new(start: u32, len: u32) -> Self {
        Self {
            start,
            len,
            marker: PhantomData,
        }
    }

    /// The number of elements in the run.
    #[must_use]
    pub const fn len(self) -> usize {
        self.len as usize
    }

    /// Whether the run has no elements.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.len == 0
    }

    /// The positions of the run's elements in its table.
    pub(super) const fn range(self) -> Range<usize> {
        let start = self.start as usize;
        start..start + self.len as usize
    }
}

impl<T> Clone for ControlRun<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for ControlRun<T> {}

impl<T> PartialEq for ControlRun<T> {
    fn eq(&self, other: &Self) -> bool {
        self.start == other.start && self.len == other.len
    }
}

impl<T> Eq for ControlRun<T> {}

impl<T> Hash for ControlRun<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.start.hash(state);
        self.len.hash(state);
    }
}

impl<T> fmt::Debug for ControlRun<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}..{}", self.start, self.start + self.len)
    }
}

/// A half-open range of the instructions of one block.
///
/// This is the only way a tree names instructions. [`ControlTree::compute`]
/// produces whole-block spans; [`ControlTree::compute_predicated`] splits a
/// block into several spans when its instructions carry different
/// predicates. Indices are positions in the block's instruction slice, the
/// same positions as [`ProgramPoint::inst_idx`].
///
/// [`ControlTree::compute`]: super::ControlTree::compute
/// [`ControlTree::compute_predicated`]: super::ControlTree::compute_predicated
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BlockSpan {
    block: BlockId,
    start: u32,
    end: u32,
}

impl BlockSpan {
    /// The span of the instructions `start..end` of `block`.
    ///
    /// # Panics
    ///
    /// Panics when `end` exceeds `u32::MAX`.
    pub(super) fn new(block: BlockId, start: usize, end: usize) -> Self {
        debug_assert!(start <= end, "a span ends no earlier than it starts");
        Self {
            block,
            start: u32::try_from(start).expect("instruction index exceeds u32::MAX"),
            end: u32::try_from(end).expect("instruction index exceeds u32::MAX"),
        }
    }

    /// The span of every instruction of `block`, which holds `len` of them.
    pub(super) fn whole(block: BlockId, len: usize) -> Self {
        Self::new(block, 0, len)
    }

    /// The block the instructions belong to.
    #[must_use]
    pub const fn block(self) -> BlockId {
        self.block
    }

    /// The position of the first instruction.
    #[must_use]
    pub const fn start(self) -> usize {
        self.start as usize
    }

    /// The position after the last instruction.
    #[must_use]
    pub const fn end(self) -> usize {
        self.end as usize
    }

    /// The positions of the instructions, as a range of the block's slice.
    #[must_use]
    pub const fn range(self) -> Range<usize> {
        self.start as usize..self.end as usize
    }

    /// The number of instructions in the span.
    #[must_use]
    pub const fn len(self) -> usize {
        (self.end - self.start) as usize
    }

    /// Whether the span holds no instruction.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.start == self.end
    }

    /// The program point of each instruction of the span, in order.
    pub fn points(self) -> impl ExactSizeIterator<Item = ProgramPoint> {
        self.range().map(move |inst_idx| ProgramPoint {
            block: self.block,
            inst_idx,
        })
    }

    /// The instructions of the span, read from the graph the tree was
    /// computed from.
    ///
    /// # Panics
    ///
    /// Panics when `graph` is not the graph the tree was computed from, or
    /// has changed since: the block must exist and hold at least
    /// [`end`](Self::end) instructions.
    #[must_use]
    pub fn instructions<G: CfgView>(self, graph: &G) -> &[G::Inst] {
        &graph.instructions(self.block)[self.range()]
    }
}

/// The recognized control shape of one [`ControlNode::Loop`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopKind {
    /// No recognized pre- or post-test; iteration ends only through an
    /// inner `Break`, `Return`, or `Goto`.
    Endless,

    /// Pre-tested loop (`while`): every iteration enters at the header,
    /// runs the straight-line condition chain, and exits along one
    /// conditional arm.
    While {
        /// The condition chain in execution order: the header first, then
        /// each block of the straight line, ending with the block that holds
        /// the conditional branch. The spans cover whole blocks. Read them
        /// with [`ControlTree::spans`](super::ControlTree::spans).
        condition: ControlRun<BlockSpan>,
        /// Whether the `ConditionalTrue` edge leaves the loop (a source
        /// `while (!c)` shape); `false` means the true edge iterates.
        exit_on_true: bool,
    },

    /// Post-tested loop (`do/while`): the latch evaluates its condition after
    /// every iteration and either returns to the header or exits.
    DoWhile {
        /// The latch block, as a whole-block span ending in the conditional
        /// branch. For a single-block loop this is the header itself and the
        /// body is empty.
        latch: BlockSpan,
        /// Whether the `ConditionalTrue` edge returns to the header.
        continue_on_true: bool,
    },
}

/// A single non-finally handler arm inside a [`ControlNode::TryCatch`].
///
/// [`Self::kind`] distinguishes catch, catch-all, fault, and filter arms.
/// Finally handlers have no arm: their bodies are joined into
/// [`ControlNode::TryCatch::finally_body`]. Consumer handler metadata (catch types, filter predicates, ordering) stays in
/// consumer-keyed side tables such as
/// [`HandlerMetadata`](crate::HandlerMetadata), addressed by
/// [`Self::handler`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CatchHandler {
    /// Stable identity of the source CFG handler.
    pub handler: HandlerRef,
    /// The entry block of the handler.
    pub entry: BlockId,
    /// Exact catch, catch-all, fault, or filter classification.
    pub kind: HandlerKind,
    /// The body of the handler.
    pub body: ControlBody,
}

/// A single case arm inside a [`ControlNode::Switch`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SwitchCase {
    /// The case block of the CFG.
    pub block: BlockId,
    /// The dispatch edges selecting this arm, in dispatch order. Case-key
    /// metadata lives on the caller's edge payloads, so a consumer recovers
    /// exact keys through [`Cfg::edge`](crate::Cfg::edge). Read them with
    /// [`ControlTree::edges`](super::ControlTree::edges).
    pub edges: ControlRun<EdgeId>,
    /// The complete body of this case arm, beginning with the case block's
    /// own instructions.
    pub body: ControlBody,
}

/// One node of a [`ControlTree`](super::ControlTree).
///
/// A node is a small `Copy` record. Its children are [`ControlBody`] runs and
/// its instructions are [`BlockSpan`]s, so the tree stays a flat arena and
/// holds no instruction of the CFG.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControlNode {
    /// A basic block's instructions — a leaf. A block without instructions
    /// has no node.
    Block(BlockSpan),

    /// A linear sequence of statements executed one after another.
    Sequence(ControlBody),

    /// A conditional branch (`if / else`).
    IfThenElse {
        /// The condition block's instructions; the last one is the branch.
        condition: BlockSpan,
        /// The "true" arm.
        then_body: ControlBody,
        /// The "false" arm (empty if there's no `else`).
        else_body: ControlBody,
    },

    /// A loop, classified as pre-tested, post-tested, or endless.
    Loop {
        /// The loop header block in the CFG.
        header: BlockId,
        /// The recognized loop shape and its condition.
        kind: LoopKind,
        /// The body of the loop.
        body: ControlBody,
    },

    /// A multi-way branch (`switch / case`).
    Switch {
        /// The dispatch block's instructions; the last one is the dispatch.
        condition: BlockSpan,
        /// The individual case arms.
        cases: ControlRun<SwitchCase>,
        /// The arm taken when no case matches (empty when the default
        /// transfers straight to the switch continuation).
        default_body: ControlBody,
        /// The dispatch edge taken when no case matches, when one exists.
        /// Case-key metadata lives on the caller's edge payloads.
        default_edge: Option<EdgeId>,
    },

    /// Break out of a loop or switch.
    Break {
        /// The header of the enclosing loop being broken, for a multi-level
        /// break; `None` breaks the innermost loop or switch. The loop is
        /// wrapped in a [`ControlNode::Label`] for the same block.
        label: Option<BlockId>,
    },

    /// Continue to a loop's continue point (its header, or a post-tested
    /// loop's condition).
    Continue {
        /// The header of the enclosing loop being continued, for a
        /// multi-level continue; `None` continues the innermost loop.
        label: Option<BlockId>,
    },

    /// Return / terminate: the instructions of a block without successors,
    /// including the return itself.
    Return(BlockSpan),

    /// A label target (used for irreducible control flow and labeled loops).
    Label {
        /// The block the label names.
        block: BlockId,
        /// Body following the label.
        body: ControlBody,
    },

    /// An unconditional goto (used for irreducible control flow).
    Goto {
        /// The block whose [`ControlNode::Label`] the goto transfers to.
        target: BlockId,
    },

    /// A try region with catch, filter, fault, and/or finally handlers.
    TryCatch {
        /// The protected body (try block).
        try_body: ControlBody,
        /// Handler arms.
        handlers: ControlRun<CatchHandler>,
        /// Finally body (empty if no finally).
        finally_body: ControlBody,
    },

    /// A predicated/guarded region — executes only when a condition
    /// holds (ARM IT blocks, GPU wave predication, CMOV). Only
    /// [`ControlTree::compute_predicated`](super::ControlTree::compute_predicated)
    /// produces it.
    Guarded {
        /// The first instruction of the guarded run, a witness carrying the
        /// guard: its [`Predicated::predicate`](crate::Predicated::predicate)
        /// names the condition variable, and its rendering labels the region.
        predicate: ProgramPoint,
        /// Whether the body executes when the predicate is *true*.
        when_true: bool,
        /// The guarded body.
        body: ControlBody,
    },
}
