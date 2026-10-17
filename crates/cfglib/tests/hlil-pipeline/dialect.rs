//! The `Calc` dialect: integers and booleans with named locals, a memory of
//! numbered cells, and an output channel.
//!
//! Everything the library leaves to the consumer lives here: the operations,
//! the constants, the value types, the metadata of a local, the effect
//! vocabulary of the flat bytecode, and the policy hooks of lifting,
//! recovery, verification, and printing.

use std::fmt;

use cfglib::DenseId;
use cfglib::ir::hlil::{
    Dialect, ExprShape, FunctionView, LiftDialect, Pseudocode, RecoverDialect, StmtShape,
    VerificationIssue, VerifyDialect,
};

/// The marker of the dialect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct Calc;

/// The value types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Ty {
    Int,
    Bool,
    /// What an operation evaluated only for its effects produces.
    Unit,
}

/// The constant domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Lit {
    Int(i64),
    Bool(bool),
}

/// The operations of expressions, places, and regions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Op {
    Add,
    Sub,
    Mul,
    /// Throws the code `-1` on a zero divisor.
    Div,
    /// Throws the code `-2` on a zero divisor.
    Rem,
    Lt,
    Ge,
    Eq,
    Ne,
    Not,
    /// `pick(condition, when_true, when_false)` evaluates one arm.
    Pick,
    /// Memory cell `n`: a load as a value, a store as an assignment target.
    Cell,
    /// Appends its operand to the output.
    Emit,
    /// Throws its operand.
    Raise,
    /// The code of the exception that the innermost handler is handling.
    Caught,
    Acquire,
    Release,
    /// The region operation that recovery gives an `acquire`/`release` pair.
    Guard,
}

/// The metadata of one named local.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Local {
    pub(crate) name: String,
    /// Whether lowering introduced the local.
    pub(crate) temporary: bool,
}

impl Local {
    pub(crate) fn named(name: &str) -> Self {
        Self {
            name: name.to_owned(),
            temporary: false,
        }
    }
}

/// What a flat instruction does besides computing its definitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Effect {
    Output,
    MemoryRead,
    MemoryWrite,
    /// Reads which exception is being handled.
    ExceptionState,
}

impl Op {
    pub(crate) const fn mnemonic(self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Sub => "sub",
            Self::Mul => "mul",
            Self::Div => "div",
            Self::Rem => "rem",
            Self::Lt => "lt",
            Self::Ge => "ge",
            Self::Eq => "eq",
            Self::Ne => "ne",
            Self::Not => "not",
            Self::Pick => "pick",
            Self::Cell => "cell",
            Self::Emit => "emit",
            Self::Raise => "raise",
            Self::Caught => "caught",
            Self::Acquire => "acquire",
            Self::Release => "release",
            Self::Guard => "guard",
        }
    }

    /// The operand types and the result type for operands of the given
    /// types, which fix the type of a polymorphic operation.
    pub(crate) fn signature(self, operands: &[Ty]) -> (Vec<Ty>, Ty) {
        use Ty::{Bool, Int, Unit};
        let carried = usize::from(self == Self::Pick);
        let first = operands.get(carried).copied().unwrap_or(Int);
        match self {
            Self::Add | Self::Sub | Self::Mul | Self::Div | Self::Rem => (vec![Int, Int], Int),
            Self::Lt | Self::Ge => (vec![Int, Int], Bool),
            Self::Eq | Self::Ne => (vec![first, first], Bool),
            Self::Not => (vec![Bool], Bool),
            Self::Pick => (vec![Bool, first, first], first),
            Self::Cell => (vec![Int], Int),
            Self::Emit => (vec![first], Unit),
            Self::Raise | Self::Acquire | Self::Release | Self::Guard => (vec![Int], Unit),
            Self::Caught => (vec![], Int),
        }
    }

    pub(crate) const fn may_throw(self) -> bool {
        matches!(self, Self::Div | Self::Rem | Self::Raise)
    }

    pub(crate) fn effects(self) -> &'static [Effect] {
        match self {
            Self::Emit | Self::Acquire | Self::Release => &[Effect::Output],
            Self::Cell => &[Effect::MemoryRead],
            Self::Caught => &[Effect::ExceptionState],
            _ => &[],
        }
    }
}

impl Dialect for Calc {
    type Op = Op;
    type Constant = Lit;
    type Type = Ty;
    type Variable = Local;
}

impl Pseudocode for Calc {
    fn mnemonic(op: &Op) -> &str {
        op.mnemonic()
    }

    fn fmt_constant(formatter: &mut fmt::Formatter<'_>, constant: &Lit) -> fmt::Result {
        match constant {
            Lit::Int(value) => write!(formatter, "{value}"),
            Lit::Bool(value) => write!(formatter, "{value}"),
        }
    }
}

impl LiftDialect for Calc {
    type Effect = Effect;

    fn void_type() -> Ty {
        Ty::Unit
    }

    fn logical_not() -> Option<Op> {
        Some(Op::Not)
    }

    fn negate_operation(op: &Op) -> Option<Op> {
        match op {
            Op::Lt => Some(Op::Ge),
            Op::Ge => Some(Op::Lt),
            Op::Eq => Some(Op::Ne),
            Op::Ne => Some(Op::Eq),
            _ => None,
        }
    }

    /// Two loads commute. Nothing else crosses anything.
    fn evaluation_commutes(
        moved_effects: &[Effect],
        moved_may_throw: bool,
        crossed_effects: &[Effect],
        crossed_may_throw: bool,
    ) -> bool {
        let reads_only = |effects: &[Effect]| effects.iter().all(|e| *e == Effect::MemoryRead);
        !moved_may_throw
            && !crossed_may_throw
            && reads_only(moved_effects)
            && reads_only(crossed_effects)
    }
}

impl RecoverDialect for Calc {
    fn select() -> Option<Op> {
        Some(Op::Pick)
    }

    fn region_enter(op: &Op) -> Option<Op> {
        (*op == Op::Acquire).then_some(Op::Guard)
    }

    fn releases(enter: &Op, exit: &Op) -> bool {
        *enter == Op::Acquire && *exit == Op::Release
    }

    fn is_exception_materialization(op: &Op) -> bool {
        *op == Op::Caught
    }

    fn is_throw(op: &Op) -> bool {
        *op == Op::Raise
    }
}

impl VerifyDialect for Calc {
    /// Operations apply to operands of their signature, conditions are
    /// booleans, a switch dispatches an integer, an assignment stores a value
    /// of the type of its place, and a return matches the signature.
    fn verify<F: FunctionView<Self>>(function: &F, issues: &mut Vec<VerificationIssue>) {
        for index in 0..function.expr_count() {
            let expr = F::Expr::from_index(index);
            let ExprShape::Operation { op, operands } = function.expr(expr) else {
                continue;
            };
            let actual: Vec<Ty> = operands
                .iter()
                .map(|&operand| *function.ty(operand))
                .collect();
            let (expected, result) = op.signature(&actual);
            if actual != expected {
                issues.push(VerificationIssue::new(format!(
                    "e{index} applies {} to {actual:?}, which wants {expected:?}",
                    op.mnemonic()
                )));
            } else if *function.ty(expr) != result && *function.ty(expr) != Ty::Unit {
                issues.push(VerificationIssue::new(format!(
                    "e{index} has the wrong result type for {}",
                    op.mnemonic()
                )));
            }
        }
        for index in 0..function.stmt_count() {
            let shape = function.stmt(F::Stmt::from_index(index));
            let wanted = match shape {
                StmtShape::If { condition, .. }
                | StmtShape::While { condition, .. }
                | StmtShape::DoWhile { condition, .. }
                | StmtShape::For {
                    condition: Some(condition),
                    ..
                } => Some((condition, Ty::Bool, "a condition")),
                StmtShape::Switch { scrutinee, .. } => Some((scrutinee, Ty::Int, "a scrutinee")),
                StmtShape::Assign { target, value } => {
                    Some((target, *function.ty(value), "a place of the value's type"))
                }
                _ => None,
            };
            if let Some((expr, ty, what)) = wanted {
                let actual = *function.ty(expr);
                // A fused branch states its condition with `void_type()`, so
                // a void occurrence stands for the boolean there.
                let fused = ty == Ty::Bool && actual == Ty::Unit;
                if actual != ty && !fused {
                    issues.push(VerificationIssue::new(format!(
                        "s{index} wants {what} of type {ty:?}, not {actual:?}"
                    )));
                }
            }
            if let StmtShape::Return { values } = shape {
                let returned: Vec<Ty> = values.iter().map(|&value| *function.ty(value)).collect();
                if returned != function.returns() {
                    issues.push(VerificationIssue::new(format!(
                        "s{index} returns {returned:?}, not the signature"
                    )));
                }
            }
        }
    }
}
