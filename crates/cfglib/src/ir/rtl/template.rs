//! Identity-free operation templates the lift hands to dialects.
//!
//! A [`VarExpr::Read`] names positions and a constraint, never a
//! variable: the web variables live in each MLIL instruction's
//! positional `uses`/`defs` lists (one use per read in pre-order, the
//! assignment target or the effect's writes as the definitions), so the
//! templates survive generic operand rewriting. [`WebInfo`] describes
//! the typed variables the lift recovered.

extern crate alloc;

use alloc::boxed::Box;
use alloc::vec::Vec;

use crate::ir::dialect::Vocabulary;
use crate::ir::mlil::VariableId;

use super::dialect::Dialect;
use super::types::{Constraint, Shape};

/// One typed expression over lifted web variables.
///
/// Consumers embed these in their MLIL operations: the tree mirrors the
/// RTL expression, with storage reads resolved to positions within web
/// variables. The web variables themselves live in the instruction's
/// `uses` list — one entry per [`Read`](VarExpr::Read) in pre-order —
/// so the template survives operand rewriting; pair reads with operands
/// through [`ReadResolver`](super::ReadResolver).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VarExpr<D: Dialect> {
    /// A read of one web variable's positions, with the constraint the
    /// consuming operation imposes. The variable read is the matching
    /// pre-order entry of the instruction's `uses`.
    Read {
        /// Positions within the web variable, in consumption order.
        positions: Vec<u8>,
        /// The constraint the consumer imposes on each position.
        scalar: D::Constraint,
    },
    /// An immediate value, one bit pattern per lane.
    Const {
        /// Raw lane bit patterns:
        /// [`Constraint::word_count`](super::Constraint::word_count) little-endian
        /// 64-bit words per lane, lanes in order.
        bits: Vec<u64>,
        /// The shape of the constant.
        shape: Shape<D::Constraint>,
    },
    /// A pure operator application.
    Apply {
        /// The dialect operator.
        operator: D::Operator,
        /// Operand values in operator order.
        operands: Vec<VarExpr<D>>,
        /// The result shape.
        shape: Shape<D::Constraint>,
    },
    /// A bit reinterpretation to a shape with the same total bit width.
    Reinterpret {
        /// The reinterpreted value.
        operand: Box<VarExpr<D>>,
        /// The target shape.
        shape: Shape<D::Constraint>,
    },
    /// A vector composed from reads of several webs — one storage read
    /// whose lanes resolved to different variables.
    Compose {
        /// The composed parts in lane order.
        parts: Vec<VarExpr<D>>,
        /// The composed shape.
        shape: Shape<D::Constraint>,
    },
}

impl<D: Dialect> VarExpr<D> {
    /// The shape of this expression's value.
    #[must_use]
    pub fn shape(&self) -> Shape<D::Constraint> {
        match self {
            Self::Read {
                positions, scalar, ..
            } => Shape {
                scalar: scalar.clone(),
                lanes: u8::try_from(positions.len()).unwrap_or(u8::MAX),
            },
            Self::Const { shape, .. }
            | Self::Apply { shape, .. }
            | Self::Reinterpret { shape, .. }
            | Self::Compose { shape, .. } => shape.clone(),
        }
    }

    /// Evaluates this template to constant lane words, or `None` when any
    /// part is unknown or malformed.
    ///
    /// Reads resolve positionally: `next_read` yields the full constant
    /// words of the matching pre-order use (or `None` for an unknown
    /// variable), and the read's own position selection is applied here.
    /// Operator applications resolve through `fold_operator`, the only
    /// dialect-owned semantics; every other arm is checked against its
    /// declared shape so a malformed template folds to `None` rather than
    /// a wrong-width constant.
    pub fn fold_constant(
        &self,
        next_read: &mut impl FnMut() -> Option<Vec<u64>>,
        fold_operator: &mut impl FnMut(
            &D::Operator,
            &[Vec<u64>],
            &Shape<D::Constraint>,
        ) -> Option<Vec<u64>>,
    ) -> Option<Vec<u64>> {
        match self {
            Self::Read { positions, scalar } => {
                let constant = next_read()?;
                let words = scalar.word_count();
                let mut selected = Vec::with_capacity(positions.len().checked_mul(words)?);
                for &position in positions {
                    let start = usize::from(position).checked_mul(words)?;
                    selected.extend_from_slice(constant.get(start..start.checked_add(words)?)?);
                }
                Some(selected)
            }
            Self::Const { bits, shape } => shape.holds_words(bits).then(|| bits.clone()),
            Self::Apply {
                operator,
                operands,
                shape,
            } => {
                let operands = operands
                    .iter()
                    .map(|operand| operand.fold_constant(next_read, fold_operator))
                    .collect::<Option<Vec<_>>>()?;
                fold_operator(operator, &operands, shape)
            }
            Self::Reinterpret { operand, shape } => {
                let bits = operand.fold_constant(next_read, fold_operator)?;
                shape.holds_words(&bits).then_some(bits)
            }
            Self::Compose { parts, shape } => {
                let mut bits = Vec::new();
                for part in parts {
                    bits.extend(part.fold_constant(next_read, fold_operator)?);
                }
                shape.holds_words(&bits).then_some(bits)
            }
        }
    }
}

/// One lifted statement handed to the consumer's
/// [`Lift::emit`](super::Lift::emit) hook.
///
/// The statement is a template over the instruction's positional
/// variable lists: reads align with `uses` in pre-order, an
/// assignment's written variable is the instruction's sole definition,
/// and an effect's [`writes`](super::Statement::Effect::writes) align
/// with `defs` in write order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LiftedStatement<D: Dialect> {
    /// One serialized assignment of a value to positions of the
    /// instruction's defined web variable.
    Assign {
        /// Written positions within the target, in value-lane order.
        positions: Vec<u8>,
        /// The target web's full width.
        width: u8,
        /// Whether unwritten positions keep their prior value — the
        /// instruction then also uses the target as its trailing operand.
        merges: bool,
        /// The assigned value.
        value: VarExpr<D>,
        /// Observable effects attached to this instruction.
        effects: Vec<<D as Vocabulary>::Effect>,
    },
    /// An effect-bearing operation.
    ///
    /// The places it writes without expressing a value are the
    /// instruction's definitions, in write order; the operands observe
    /// the state before them.
    Effect {
        /// The dialect effect operation.
        operation: D::EffectOp,
        /// Operand values in operation order.
        operands: Vec<VarExpr<D>>,
        /// Observable effects of the operation.
        effects: Vec<<D as Vocabulary>::Effect>,
    },
    /// A conditional transfer on one scalar condition.
    Branch {
        /// The scalar branch condition.
        condition: VarExpr<D>,
    },
    /// A multi-way dispatch on one scalar scrutinee.
    Dispatch {
        /// The scalar dispatch scrutinee.
        scrutinee: VarExpr<D>,
    },
    /// A function return carrying result values.
    ///
    /// The lift materializes every non-trivial value into a temporary,
    /// so each value is a whole single-variable [`VarExpr::Read`] and
    /// the instruction's `uses` pair one-to-one with the returned
    /// values — the shape [`Lifted::Return`](crate::ir::hlil::Lifted)
    /// requires.
    Return {
        /// Returned values in signature order.
        values: Vec<VarExpr<D>>,
    },
    /// A terminating exceptional raise.
    Raise {
        /// The dialect effect operation performing the raise.
        operation: D::EffectOp,
        /// Operand values in operation order.
        operands: Vec<VarExpr<D>>,
        /// Observable effects beyond the exceptional transfer itself.
        effects: Vec<<D as Vocabulary>::Effect>,
    },
}

/// One lifted web: a typed MLIL variable recovered from storage lanes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WebInfo<D: Dialect> {
    /// The declared MLIL variable.
    pub variable: VariableId,
    /// The native storage the web lives in, or `None` for a synthetic
    /// temporary.
    pub storage: Option<<D as Vocabulary>::NativeVariable>,
    /// Ascending storage lanes the web spans.
    pub lanes: Vec<u8>,
    /// The web's inferred shape.
    pub shape: Shape<D::Constraint>,
    /// Whether the web contains the version-zero live-in value.
    pub live_in: bool,
}
