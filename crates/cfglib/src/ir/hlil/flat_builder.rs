//! The write-only protocol through which [`lower`](super::lower) emits a flat
//! function.
//!
//! The consumer owns the flat storage. It implements [`FlatBuilder`] over a
//! [`Cfg`](crate::Cfg) of its own instruction and edge payloads, and lowering
//! appends blocks, instructions, edges, and regions to it. Nothing is read
//! back and nothing is edited in place: handles are returned by the builder
//! and passed to it again, and the consumer seals the function in
//! [`finish`](FlatBuilder::finish).
//!
//! The builder does not receive the consumer's operations. It receives what
//! an instruction *stands for*, named in HLIL terms ([`Lowered`]), and chooses
//! the flat operation itself. The edges arrive the same way, as the role of
//! the transfer ([`LowerEdge`]), and the builder chooses the payload and the
//! [`EdgeKind`](crate::EdgeKind).

use crate::block::BlockId;
use crate::dataflow::ProgramPoint;
use crate::edge::EdgeId;
use crate::graph::view::DenseId;
use crate::region::{Region, RegionId};

use super::Dialect;

/// One flat occurrence of a variable: the variable and the type of that
/// occurrence.
///
/// The type belongs to the occurrence, not to the variable: one variable can
/// be read at several types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Occurrence<V, T> {
    /// The variable read or defined.
    pub variable: V,
    /// The type of this occurrence.
    pub ty: T,
}

/// What a flat instruction stands for, named in HLIL terms.
///
/// The builder maps it to the consumer's flat operation. It is the inverse
/// of `InstructionShape`: the shape classifies a
/// flat instruction for lifting, and this names what lowering needs the
/// flat instruction to be.
///
/// Each variant states how many uses and definitions accompany it.
#[derive(Debug, Clone, Copy)]
pub enum Lowered<'a, H: Dialect> {
    /// The operation over the flattened operands, in operand order. One
    /// definition, or none for an operation evaluated for effect.
    Operation(&'a H::Op),
    /// A constant materialized into its definition. No uses, one definition.
    Constant(&'a H::Constant),
    /// A copy. One use, one definition.
    Copy,
    /// Stores the **last** use into the place that `location` forms over
    /// the preceding uses. No definitions.
    Store {
        /// The operation forming the place.
        location: &'a H::Op,
    },
    /// A conditional branch on its one use, the condition. No definitions.
    Branch,
    /// A multi-way dispatch on its one use, the scrutinee. No definitions.
    Switch,
    /// A return carrying its uses as the returned values. No definitions.
    Return,
    /// Defines a handler binding: the delivered exception is materialized
    /// into the one definition. No uses.
    ///
    /// A builder with no caught-exception form returns an error.
    CaughtException,
    /// The enter half of one dialect region statement, over the uses that
    /// the statement's operands flatten to. No definitions.
    ///
    /// A builder with no enter/exit protocol for the operation returns an
    /// error.
    RegionEnter(&'a H::Op),
    /// The exit half of the region that [`RegionEnter`](Self::RegionEnter)
    /// opened, over the same uses. No definitions.
    RegionExit(&'a H::Op),
}

/// The role of one flat edge. The builder chooses the payload and the
/// [`EdgeKind`](crate::EdgeKind).
///
/// Lowering names the role and never the kind, because consumers disagree on
/// the kind of a synthesized transfer: one reads [`Jump`](Self::Jump) as an
/// explicit `goto`, another as a structured break. A builder that re-lifts
/// its own output is structure-faithful only if it maps `Jump` to a kind that
/// the control-tree structurer resolves as a break or continue.
#[derive(Debug)]
pub enum LowerEdge<'a, C> {
    /// The synthetic root to the first block.
    Entry,
    /// Sequential flow between consecutively lowered blocks.
    Fallthrough,
    /// A synthesized transfer: break, continue, goto, or a loop back edge.
    Jump,
    /// The taken arm of a conditional.
    True,
    /// The not-taken arm of a conditional.
    False,
    /// One switch case value.
    Case(&'a C),
    /// Taken when no switch case matches.
    Default,
    /// Exceptional: from a block that may throw to a handler entry.
    Unwind,
}

impl<C> Clone for LowerEdge<'_, C> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<C> Copy for LowerEdge<'_, C> {}

/// The result of appending one instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Appended {
    /// Where the instruction now stands.
    pub point: ProgramPoint,
    /// Whether the instruction may transfer exceptionally. Lowering places
    /// an unwind edge from the block of every such instruction inside a
    /// protected region to each handler of that region.
    pub may_throw: bool,
}

/// How a flat variable is introduced.
#[derive(Debug, Clone, Copy)]
pub enum VariableOrigin<'a, H: Dialect> {
    /// One variable of the structured function, copied one to one.
    Source {
        /// The metadata of the structured variable.
        variable: &'a H::Variable,
        /// Its declared type, when the structured function states one.
        declared_type: Option<&'a H::Type>,
    },
    /// A transient that lowering introduces to hold one flattened value.
    /// The consumer chooses its role and native identity.
    Temporary {
        /// The type of the value it holds.
        ty: &'a H::Type,
    },
}

/// The write-only protocol through which lowering emits a flat function.
///
/// # Contract
///
/// - A function starts with one block, [`entry`](Self::entry), the synthetic
///   root. It holds no instruction and has one outgoing edge, the
///   [`Entry`](LowerEdge::Entry) edge to the first block that holds code.
/// - Handles come from the builder: a block from
///   [`new_block`](Self::new_block) or [`entry`](Self::entry), a variable
///   from [`declare_variable`](Self::declare_variable). Block ids are not
///   assumed dense or sequential.
/// - Variables are declared in the order of the structured function's
///   variable table, so the `n`-th [`Source`](VariableOrigin::Source)
///   declaration is the `n`-th structured variable. Temporaries follow.
/// - An instruction is appended to the end of its block, and an edge is
///   added after both endpoints exist. A block receives at most one
///   terminating instruction ([`Branch`](Lowered::Branch),
///   [`Switch`](Lowered::Switch), [`Return`](Lowered::Return)), and it is
///   the last one appended.
/// - The consumer checks the contract at [`finish`](Self::finish) and
///   reports through [`Error`](Self::Error). A builder that has no flat form
///   for a [`Lowered`] fails the [`append`](Self::append).
pub trait FlatBuilder<H: Dialect> {
    /// Dense variable identity of the flat function.
    type Var: DenseId;
    /// The sealed flat function.
    type Function;
    /// Why the consumer refused an instruction or the sealed function.
    type Error;

    /// The synthetic root block, which already exists.
    fn entry(&self) -> BlockId;

    /// Declares one variable. Identities are issued in call order.
    fn declare_variable(&mut self, origin: VariableOrigin<'_, H>) -> Self::Var;

    /// Declares the ordered signature.
    fn set_signature(&mut self, parameters: &[Self::Var], returns: &[H::Type]);

    /// Allocates a block. `None` is a synthetic block with no authored label;
    /// the lifter names it by its index.
    fn new_block(&mut self, label: Option<&str>) -> BlockId;

    /// Appends one instruction to the end of `block`.
    ///
    /// `uses` and `defs` are the occurrences the instruction reads and
    /// writes, in order; [`Lowered`] states how many of each to expect.
    ///
    /// # Errors
    ///
    /// The consumer refuses a `lowered` that it has no flat form for.
    fn append(
        &mut self,
        block: BlockId,
        lowered: Lowered<'_, H>,
        uses: &[Occurrence<Self::Var, H::Type>],
        defs: &[Occurrence<Self::Var, H::Type>],
    ) -> Result<Appended, Self::Error>;

    /// Adds one edge of the given role.
    fn add_edge(
        &mut self,
        source: BlockId,
        target: BlockId,
        role: LowerEdge<'_, H::Constant>,
    ) -> EdgeId;

    /// Registers one exception region; `region.id` is ignored and the
    /// builder assigns the identity.
    fn add_region(&mut self, region: Region) -> RegionId;

    /// Seals the function.
    ///
    /// # Errors
    ///
    /// The consumer refuses a function that violates its contract.
    fn finish(self) -> Result<Self::Function, Self::Error>;
}
