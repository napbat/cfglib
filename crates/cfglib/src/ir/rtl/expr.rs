//! Typed RTL expression trees over raw storage.

extern crate alloc;

use alloc::boxed::Box;
use alloc::vec::Vec;

use crate::ir::dialect::Vocabulary;

use super::dialect::Dialect;
use super::types::Shape;

/// One pure typed value expression.
///
/// Reads observe the state at the start of the containing statement:
/// every read in a statement sees the same pre-statement storage, no
/// matter how many transfers the statement performs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expr<D: Dialect> {
    /// A swizzled read of storage lanes, with the constraint the
    /// consuming operation imposes on the bits.
    Read {
        /// The storage location read.
        storage: <D as Vocabulary>::NativeVariable,
        /// Storage lanes selected, in consumption order; repeats allowed.
        lanes: Vec<u8>,
        /// The constraint the consumer imposes on each lane.
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
        operands: Vec<Expr<D>>,
        /// The result shape.
        shape: Shape<D::Constraint>,
    },
    /// A bit reinterpretation to a shape with the same total bit width.
    ///
    /// The source and target can have different lane counts. For example,
    /// two 32-bit lanes can become one 64-bit lane.
    Reinterpret {
        /// The reinterpreted value.
        operand: Box<Expr<D>>,
        /// The target shape.
        shape: Shape<D::Constraint>,
    },
}

impl<D: Dialect> Expr<D> {
    /// The shape of this expression's value.
    #[must_use]
    pub fn shape(&self) -> Shape<D::Constraint> {
        match self {
            Self::Read { lanes, scalar, .. } => Shape {
                scalar: scalar.clone(),
                lanes: lane_count(lanes),
            },
            Self::Const { shape, .. }
            | Self::Apply { shape, .. }
            | Self::Reinterpret { shape, .. } => shape.clone(),
        }
    }

    /// Visits every read in deterministic pre-order.
    ///
    /// This order is the contract between use collection and lowering:
    /// both walk reads identically, so SSA use positions line up with
    /// read nodes.
    pub fn for_each_read(
        &self,
        visit: &mut impl FnMut(&<D as Vocabulary>::NativeVariable, &[u8], &D::Constraint),
    ) {
        match self {
            Self::Read {
                storage,
                lanes,
                scalar,
            } => visit(storage, lanes, scalar),
            Self::Const { .. } => {}
            Self::Apply { operands, .. } => {
                for operand in operands {
                    operand.for_each_read(visit);
                }
            }
            Self::Reinterpret { operand, .. } => operand.for_each_read(visit),
        }
    }
}

/// One written destination: unique storage lanes in write order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Place<D: Dialect> {
    /// The storage location written.
    pub storage: <D as Vocabulary>::NativeVariable,
    /// Distinct storage lanes written, in value-lane order.
    pub lanes: Vec<u8>,
}

pub(super) fn lane_count(lanes: &[u8]) -> u8 {
    u8::try_from(lanes.len()).unwrap_or(u8::MAX)
}
