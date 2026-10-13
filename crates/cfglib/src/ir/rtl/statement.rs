//! The parts of one RTL instruction record.
//!
//! The same types serve the [`InstructionBuilder`](super::InstructionBuilder),
//! which takes them as input, and the [`RecordView`](super::RecordView),
//! which yields them. Every slice inside them is flat, so a consumer can store
//! the parts in dense pools and lend them back without rebuilding them.

use crate::graph::view::DenseId;

use super::bits::{BitSlice, Bits};
use super::dialect::{Dialect, Place};

/// A run of consecutive values, such as the results of one operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValueRange<V> {
    /// The first value of the run.
    pub first: V,
    /// The number of values in the run.
    pub len: u32,
}

impl<V: DenseId> ValueRange<V> {
    /// Creates a run of `len` values that starts at `first`.
    #[must_use]
    pub const fn new(first: V, len: u32) -> Self {
        Self { first, len }
    }

    /// Returns the value at `index` of the run, or `None` past its end.
    #[must_use]
    pub fn get(self, index: u32) -> Option<V> {
        (index < self.len).then(|| V::from_index(self.first.index() + index as usize))
    }

    /// Iterates over the values of the run in order.
    pub fn iter(self) -> impl ExactSizeIterator<Item = V> {
        let first = self.first.index();
        (first..first + self.len as usize).map(V::from_index)
    }
}

/// Where a read takes its bits from. Every read observes the state before
/// the instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Selection<'a, D: Dialect, V> {
    /// One fixed place.
    Direct(Place<D>),
    /// The place `candidates[k]` where `k` is the value of `index`.
    ///
    /// The producer normalizes the index (for example `(top + i) & 7`), so
    /// every value it can take selects one candidate. All candidates have the
    /// same width. A read and a write that share one index value select the
    /// same candidate, which is how the data and the tag of one x87 register
    /// stay coupled.
    FiniteIndexed {
        /// The selector, computed from the state before the instruction.
        index: V,
        /// The places that the selector chooses from, in index order.
        candidates: &'a [Place<D>],
    },
    /// An element of a sparse or unbounded bank, such as a model-specific
    /// register. A consumer keeps such an access opaque instead of expanding
    /// it.
    SparseIndexed {
        /// The root that stands for the bank. Its width is the width of one
        /// element.
        bank: D::Root,
        /// The element key.
        key: V,
        /// The bits of the element.
        slice: BitSlice,
    },
}

/// What one span of a written root receives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Action<D: Dialect, V> {
    /// The span keeps its value from before the instruction.
    PreservePreState,
    /// The span receives bits of `value`, starting at bit `offset` of the
    /// value. A constant or a copy of another root's earlier bits is a value
    /// like any other.
    TakeInput {
        /// The value that supplies the bits.
        value: V,
        /// The first bit of `value` that the span receives.
        offset: u16,
    },
    /// The span receives a fresh nondeterministic choice from `domain`.
    Nondeterministic(D::NondetDomain),
}

/// One span of a written root and what it receives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SpanAction<D: Dialect, V> {
    /// The bits of the root that this action covers.
    pub span: BitSlice,
    /// What those bits receive.
    pub action: Action<D, V>,
}

/// The state that one write changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WriteTarget<'a, D: Dialect, V> {
    /// One root, with its complete coverage.
    ///
    /// The spans are in ascending order, start at bit zero, end at the root
    /// width, and neither overlap nor leave a gap. The producer resolves its
    /// write policy into this plan.
    Root {
        /// The written root.
        root: D::Root,
        /// The complete coverage of `root`.
        plan: &'a [SpanAction<D, V>],
    },
    /// The place `candidates[k]` where `k` is the value of `index` receives
    /// `value`. Every other bit of every candidate root keeps its value from
    /// before the instruction.
    FiniteIndexed {
        /// The selector, computed from the state before the instruction.
        index: V,
        /// The places that the selector chooses from, in index order.
        candidates: &'a [Place<D>],
        /// The written value. Its width equals the width of every candidate.
        value: V,
    },
    /// An element of a sparse bank receives `value` in `slice`.
    SparseIndexed {
        /// The root that stands for the bank.
        bank: D::Root,
        /// The element key.
        key: V,
        /// The written bits of the element.
        slice: BitSlice,
        /// The written value.
        value: V,
    },
}

/// A non-normal outcome that one statement of the instruction raises.
///
/// When the outcome is raised, the delta of the instruction commits nothing.
/// The raise can still publish state of its own, its
/// [`binds`](Self::binds): the architecture sets some state before it
/// raises, such as the status flags of a floating-point exception that the
/// program does not mask.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Raise<'a, D: Dialect, V> {
    /// The kind of outcome.
    pub outcome: D::Outcome,
    /// The one-bit value that raises the outcome when it is one. `None`
    /// raises the outcome unconditionally, and the instruction then ends with
    /// [`Control::Terminating`].
    pub guard: Option<V>,
    /// The values that the outcome carries, such as a fault vector and an
    /// error code.
    pub payload: &'a [V],
    /// The state that the outcome publishes, as pairs of a root and the new
    /// value of the whole root.
    ///
    /// A value is computed from the state before the instruction, like every
    /// other value of the record. Its width is the width of its root. A root
    /// appears at most once. The pairs are in the order of the producer, and a
    /// consumer reports them in that order.
    ///
    /// A binding is not a write. On the normal path the root keeps the value
    /// that it has before the instruction, and the delta does not contain the
    /// binding. The bound values are uses of their definitions, and nothing
    /// else.
    pub binds: &'a [(D::Root, V)],
}

/// The single control boundary that ends an instruction.
///
/// A target is an address value. A conditional transfer that does not name
/// its other target continues at the next instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Control<V> {
    /// Execution continues at the next instruction.
    Advance,
    /// Execution continues at `target`.
    Jump {
        /// The address of the next instruction to run.
        target: V,
    },
    /// Execution continues at `taken` when `condition` is one, and otherwise
    /// at `not_taken`, or at the next instruction when `not_taken` is `None`.
    /// A repeated instruction names its own address as `taken`.
    Branch {
        /// The one-bit condition.
        condition: V,
        /// The target when the condition is one.
        taken: V,
        /// The target when the condition is zero.
        not_taken: Option<V>,
    },
    /// A call to `target` that returns to the next instruction.
    Call {
        /// The address of the callee.
        target: V,
    },
    /// A return to `target`.
    Return {
        /// The return address.
        target: V,
    },
    /// A multi-way transfer: execution continues at the decoded successor
    /// whose case value equals `scrutinee`, or at the decoded default
    /// successor. The case values belong to the decoded control flow, not to
    /// the record.
    Dispatch {
        /// The value that selects the case.
        scrutinee: V,
    },
    /// The instruction does not continue: it ended in an unconditional
    /// [`Raise`] or a terminating intrinsic.
    Terminating,
}

/// One statement of a sealed record, in the order that it was built.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Statement<'a, D: Dialect, V> {
    /// A read of the state before the instruction.
    Read {
        /// The value that holds the bits.
        result: V,
        /// Where the bits come from.
        source: Selection<'a, D, V>,
    },
    /// A constant, stored as little-endian 64-bit words.
    Constant {
        /// The constant value.
        result: V,
        /// The words of the constant. Bits past the width are zero.
        words: &'a [u64],
        /// The type of the constant.
        width: Bits,
    },
    /// A pure operator.
    Apply {
        /// The results, in operator order.
        results: ValueRange<V>,
        /// The operator.
        operator: &'a D::Operator,
        /// The operands, in operator order.
        args: &'a [V],
    },
    /// A fresh nondeterministic choice.
    Nondet {
        /// The chosen value.
        result: V,
        /// The type of the value.
        width: Bits,
        /// The domain that the value is chosen from.
        domain: D::NondetDomain,
    },
    /// An ordered effect.
    Effect {
        /// The results, in effect order.
        results: ValueRange<V>,
        /// The effect.
        operation: &'a D::EffectOp,
        /// The operands, in effect order.
        args: &'a [V],
        /// The outcomes that the environment can raise at this effect, such
        /// as a page fault of a load.
        outcomes: &'a [D::Outcome],
    },
    /// An outcome that the semantics raise.
    Raise(Raise<'a, D, V>),
    /// A write into the delta that commits at the instruction boundary.
    Write {
        /// The changed state.
        target: WriteTarget<'a, D, V>,
        /// The policy that the producer resolved into `target`.
        policy: D::WritePolicy,
    },
}
