//! RTL statements: parallel transfers, effects, and control flow.

extern crate alloc;

use alloc::vec::Vec;

use crate::InstrInfo;
use crate::ir::dialect::Vocabulary;

use super::dialect::Dialect;
use super::expr::{Expr, Place};

/// One storage lane: a native location and a lane index within it.
pub type Lane<D> = (<D as Vocabulary>::NativeVariable, u8);

crate::identity::define_dense_id! {
    /// The dense identity of one stored RTL statement.
    ///
    /// Assigned by [`FunctionBuilder::append`](super::FunctionBuilder::append)
    /// in append order across the whole function; the lift's provenance maps
    /// key on it.
    pub struct StatementId(u32);
    raw;
}

/// One native instruction expressed at the RTL level.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Statement<D: Dialect> {
    /// Parallel transfers: every read observes pre-statement storage,
    /// no matter which assignment consumes it.
    Transfer {
        /// Destination places with their values, in native order.
        assignments: Vec<(Place<D>, Expr<D>)>,
        /// Observable effects beyond the storage writes.
        effects: Vec<<D as Vocabulary>::Effect>,
        /// Whether the instruction can transfer exceptionally.
        may_throw: bool,
    },
    /// An effect-bearing operation with pure operand values.
    Effect {
        /// The dialect effect operation.
        operation: D::EffectOp,
        /// Operand values in operation order.
        operands: Vec<Expr<D>>,
        /// Places the operation writes with values the statement does not
        /// express, in write order. Every read of the statement observes
        /// the state before these writes.
        writes: Vec<Place<D>>,
        /// Observable effects of the operation.
        effects: Vec<<D as Vocabulary>::Effect>,
        /// Whether the operation can transfer exceptionally.
        may_throw: bool,
    },
    /// A conditional transfer on one scalar condition; outcomes are the
    /// block's outgoing edges.
    Branch {
        /// The scalar branch condition.
        condition: Expr<D>,
    },
    /// A multi-way dispatch on one scalar scrutinee — a switch table, a
    /// computed goto, a `ret`-continuation dispatch. Outcomes are the
    /// block's outgoing edges; case metadata is caller-owned on each
    /// edge.
    Dispatch {
        /// The scalar dispatch scrutinee.
        scrutinee: Expr<D>,
    },
    /// A function return carrying result values.
    Return {
        /// Returned values in signature order.
        values: Vec<Expr<D>>,
    },
    /// A terminating exceptional raise — a `throw`, a deliberate trap.
    /// Control leaves through the block's exceptional edges, or unwinds
    /// out of the function when the block has none.
    ///
    /// A raise declares no writes: control never reaches the point where
    /// they would be observable.
    Raise {
        /// The dialect effect operation performing the raise.
        operation: D::EffectOp,
        /// Operand values in operation order.
        operands: Vec<Expr<D>>,
        /// Observable effects beyond the exceptional transfer itself.
        effects: Vec<<D as Vocabulary>::Effect>,
    },
}

impl<D: Dialect> Statement<D> {
    /// Visits every read in deterministic statement order.
    ///
    /// The traversal order — assignments (or operands, or values) in
    /// order, each expression in pre-order — is the contract that keeps
    /// SSA use positions aligned with read nodes during lowering.
    pub fn for_each_read(
        &self,
        visit: &mut impl FnMut(&<D as Vocabulary>::NativeVariable, &[u8], &D::Constraint),
    ) {
        match self {
            Self::Transfer { assignments, .. } => {
                for (_, value) in assignments {
                    value.for_each_read(visit);
                }
            }
            Self::Effect { operands, .. } | Self::Raise { operands, .. } => {
                for operand in operands {
                    operand.for_each_read(visit);
                }
            }
            Self::Branch { condition } => condition.for_each_read(visit),
            Self::Dispatch { scrutinee } => scrutinee.for_each_read(visit),
            Self::Return { values } => {
                for value in values {
                    value.for_each_read(visit);
                }
            }
        }
    }

    /// Visits every written place in deterministic def order.
    ///
    /// A transfer's destinations come first in assignment order, an
    /// effect's [`writes`](Self::Effect::writes) in write order, and
    /// every other form writes nothing. The order is the contract that
    /// keeps SSA definition positions aligned with places, exactly as
    /// [`for_each_read`](Self::for_each_read) does for uses.
    pub fn for_each_write(&self, visit: &mut impl FnMut(&Place<D>)) {
        match self {
            Self::Transfer { assignments, .. } => {
                for (place, _) in assignments {
                    visit(place);
                }
            }
            Self::Effect { writes, .. } => {
                for place in writes {
                    visit(place);
                }
            }
            Self::Branch { .. }
            | Self::Dispatch { .. }
            | Self::Return { .. }
            | Self::Raise { .. } => {}
        }
    }

    /// Whether the statement can transfer exceptionally.
    #[must_use]
    pub const fn may_throw(&self) -> bool {
        match self {
            Self::Transfer { may_throw, .. } | Self::Effect { may_throw, .. } => *may_throw,
            Self::Raise { .. } => true,
            Self::Branch { .. } | Self::Dispatch { .. } | Self::Return { .. } => false,
        }
    }

    /// Whether the statement terminates its block.
    #[must_use]
    pub const fn is_terminator(&self) -> bool {
        matches!(
            self,
            Self::Branch { .. } | Self::Dispatch { .. } | Self::Return { .. } | Self::Raise { .. }
        )
    }
}

/// One stored statement with its identity and cached lane-level data
/// dependencies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StatementNode<D: Dialect> {
    pub(super) id: StatementId,
    pub(super) statement: Statement<D>,
    pub(super) span: Option<<D as Vocabulary>::SourceSpan>,
    pub(super) uses: Vec<Lane<D>>,
    pub(super) defs: Vec<Lane<D>>,
}

impl<D: Dialect> StatementNode<D> {
    pub(super) fn new(
        id: StatementId,
        statement: Statement<D>,
        span: Option<<D as Vocabulary>::SourceSpan>,
    ) -> Self {
        let mut uses = Vec::new();
        statement.for_each_read(&mut |storage, lanes, _| {
            for &lane in lanes {
                uses.push((storage.clone(), lane));
            }
        });
        let mut defs = Vec::new();
        statement.for_each_write(&mut |place| {
            for &lane in &place.lanes {
                defs.push((place.storage.clone(), lane));
            }
        });
        Self {
            id,
            statement,
            span,
            uses,
            defs,
        }
    }

    /// The statement's stable identity.
    #[must_use]
    pub const fn id(&self) -> StatementId {
        self.id
    }

    /// The stored statement.
    #[must_use]
    pub const fn statement(&self) -> &Statement<D> {
        &self.statement
    }

    /// The source span the statement lowers from.
    #[must_use]
    pub const fn span(&self) -> Option<&<D as Vocabulary>::SourceSpan> {
        self.span.as_ref()
    }
}

impl<D: Dialect> InstrInfo for StatementNode<D> {
    type Variable = Lane<D>;

    fn uses(&self) -> &[Self::Variable] {
        &self.uses
    }

    fn defs(&self) -> &[Self::Variable] {
        &self.defs
    }
}
