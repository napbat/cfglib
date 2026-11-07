//! The flat, variable-based IR that lifting reads.
//!
//! A flat function is a [`Cfg`] of instructions that read and write numbered
//! variables. The consumer owns the storage. [`FlatFunction`] lends out the
//! graph, the operand context, the variable table, the signature and the
//! block labels. The instruction and edge types of the graph implement
//! [`LiftInstruction`] and [`LiftEdge`] to say what each of them stands for
//! in HLIL terms, and the operand context implements [`LiftOperands`] to say
//! what each instruction reads, writes and does.

extern crate alloc;
use alloc::borrow::Cow;

use crate::EffectOperands;
use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::flow::FlowControl;
use crate::graph::view::DenseId;

use super::dialect::LiftDialect;

/// The translation of one flat instruction, as an operation over variables
/// and constants.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InstructionShape<Op, Constant> {
    /// A constant materialized into the single definition of the instruction.
    /// It is consulted before [`Copy`](Self::Copy).
    Constant(Constant),
    /// A pure copy of the single use into the single definition.
    Copy,
    /// Applies `Op` to the lifted uses. With one definition it is a value, and
    /// with none an effect statement.
    ///
    /// HLIL has no form for an operation with several definitions, such as a
    /// call that states every register it clobbers. Lifting one is reported as
    /// [`LiftError::Unsupported`](super::LiftError::Unsupported) instead of
    /// silently dropping the extra definitions. Eliminate or split them in the
    /// flat function first.
    Operation(Op),
    /// Stores the **last** use into the place that `location` forms over the
    /// preceding uses: `store(addr, v)` becomes `location(addr) = v`.
    Store {
        /// The place operation of the written location.
        location: Op,
    },
    /// A pairwise parallel move: use `i` moves to definition `i`, and every
    /// read happens before any write (stack `dup` and `swap` families, the
    /// commit of a phi copy). A single pair is a plain
    /// [`Copy`](Self::Copy). Overlapping moves need
    /// [`LiftDialect::temporary_variable`].
    ParallelCopy,
    /// The conditional branch of the block. The first use is the condition.
    Branch,
    /// The conditional branch of the block, deciding on `Op` applied to
    /// **all** of its uses: a fused compare-and-branch whose condition exists
    /// only as an expression. An empty use list is legal when the operation
    /// embeds its whole condition.
    BranchOperation(Op),
    /// The multi-way dispatch of the block. The first use is the scrutinee.
    Switch,
    /// A function return. The uses are the returned values.
    Return,
    /// A pure control transfer with no HLIL form: a jump, a fallthrough
    /// helper, or a no-op.
    ControlFlow,
}

/// An instruction of a flat function: an operation over variables.
///
/// [`FlowControl`] supplies its flow classification. This trait adds what
/// only HLIL needs and the instruction can answer alone. What it reads,
/// writes and does comes from the [`LiftOperands`] context of its function.
pub trait LiftInstruction<H: LiftDialect>: FlowControl {
    /// Classifies this instruction. The lift calls it once per instruction
    /// per translation.
    fn lift(&self) -> InstructionShape<H::Op, H::Constant>;

    /// The use position that reads the previous value of the destination, for
    /// operations that define only part of their storage under a
    /// read-modify-write convention (masked vector lanes, sub-registers).
    ///
    /// Declaring it makes the merge visible to the lift. The definition that
    /// feeds that position never inlines into it, and the definition of this
    /// instruction never inlines forward, because its value merges with the
    /// prior state and is not a pure expression. `None` declares a
    /// whole-storage definition.
    fn previous_value_operand(&self) -> Option<usize> {
        None
    }

    /// Whether this definition must remain a visible statement.
    ///
    /// Return `true` when moving the value into its consumer would lose a
    /// relation that the expression tree cannot state. The default permits
    /// ordinary single-use inlining.
    fn materialize_value(&self) -> bool {
        false
    }
}

/// What the lift reads of an instruction through the context of its flat
/// function.
///
/// [`InstrOperands`](crate::InstrOperands) supplies the variables that it reads and writes and its
/// unwind points, and [`EffectOperands`] its effects. This trait adds the
/// types of those variables and whether the instruction may throw. A
/// consumer whose instructions own their operands implements it for `()`,
/// which already reads [`InstrInfo`](crate::InstrInfo) and
/// [`EffectInfo`](crate::EffectInfo) instructions.
pub trait LiftOperands<H: LiftDialect, I>:
    EffectOperands<I, Variable: DenseId, Effect = H::Effect>
{
    /// The type of each use occurrence, parallel to
    /// [`uses`](crate::InstrOperands::uses).
    fn use_types<'a>(&'a self, instruction: &'a I) -> &'a [H::Type];

    /// The type of each definition, parallel to
    /// [`defs`](crate::InstrOperands::defs).
    fn def_types<'a>(&'a self, instruction: &'a I) -> &'a [H::Type];

    /// Whether execution may transfer exceptionally at `instruction`.
    ///
    /// The default is [`InstrOperands::may_unwind`](crate::InstrOperands::may_unwind).
    /// Override it only when an
    /// instruction can throw without being an unwind point of the graph.
    fn may_throw(&self, instruction: &I) -> bool {
        self.may_unwind(instruction)
    }
}

/// An edge payload of a flat function.
///
/// The graph stores the [`EdgeKind`](crate::EdgeKind) of an edge beside its
/// payload, so the payload only answers what HLIL asks of it.
pub trait LiftEdge<H: LiftDialect> {
    /// The case values that this switch-dispatch edge carries; empty for any
    /// other edge.
    ///
    /// Lifting a switch needs every case edge to yield at least one value. The
    /// default edge of a dispatch is never queried.
    fn case_values(&self) -> &[H::Constant];
}

/// The flat function that is lifted: a [`Cfg`] with the context of its
/// operands, its variable table, its signature and its block labels.
pub trait FlatFunction<H: LiftDialect> {
    /// The dense variable identity of the flat function.
    type Var: DenseId;
    /// The instruction type of the graph.
    type Inst: LiftInstruction<H>;
    /// The edge payload type of the graph.
    type Edge: LiftEdge<H>;
    /// The context that answers for the operands of the instructions.
    type Operands: LiftOperands<H, Self::Inst, Variable = Self::Var>;

    /// The blocks, instructions, edges and regions.
    fn cfg(&self) -> &Cfg<Self::Inst, Self::Edge>;

    /// The context that answers for the operands of the instructions.
    fn operands(&self) -> &Self::Operands;

    /// The exclusive bound of the variable indices.
    fn variable_count(&self) -> usize;

    /// The metadata of one variable.
    fn variable(&self, variable: Self::Var) -> &H::Variable;

    /// The parameters of the function, in order.
    fn parameters(&self) -> &[Self::Var];

    /// The types of the returned values, in order.
    fn returns(&self) -> &[H::Type];

    /// The name of `block` where the lift needs one: a label, a `goto`
    /// target, or a labeled `break` or `continue`.
    ///
    /// The default is the label stored in the block. A function whose names
    /// derive from something it already holds answers here instead, so its
    /// blocks store none. An absent or empty name makes the lift name the
    /// block `.bb{index}`.
    fn block_label(&self, block: BlockId) -> Option<Cow<'_, str>> {
        self.cfg().block(block).label().map(Cow::Borrowed)
    }
}
