//! The toy HLIL dialect the tests in this module are written against.
//!
//! It lives beside them rather than inside them so the test file stays under
//! the source-size policy, and so every child module reaches one definition.

extern crate alloc;

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use crate::ir::dialect::Vocabulary;
use crate::ir::mlil;
use crate::test_util::toy::{self, Span};
use crate::{EdgeKind, FlowEffect};

use super::super::{
    Dialect, Function, LiftDialect, Lifted, LowerDialect, RecoverDialect, VerificationIssue,
    VerifyDialect,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum Type {
    Integer,
    Boolean,
    Void,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Effect {
    Read,
    Write,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Toy;

impl Vocabulary for Toy {
    type ValueType = Type;
    type Effect = Effect;
    type Source = String;
    type SourceSpan = Span;
    type SourcePoint = u32;
    type VariableRole = u8;
    type NativeVariable = u8;

    fn span_is_empty(span: &Self::SourceSpan) -> bool {
        toy::span_is_empty(*span)
    }

    fn span_contains(span: &Self::SourceSpan, point: &Self::SourcePoint) -> bool {
        toy::span_contains(*span, *point)
    }
}

/// High-level operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Operation {
    Add,
    LessThan,
    Below,
    AtLeast,
    Not,
    Load,
    Deref,
    Call,
    Select,
    Acquire,
    Release,
    Caught,
    Throw,
    /// An operation whose source spelling expands into multiple statements.
    Expanded,
}

impl Dialect for Toy {
    type Operation = Operation;
    type Constant = i64;

    fn mnemonic(operation: &Self::Operation) -> &str {
        match operation {
            Operation::Add => "add",
            Operation::LessThan => "lt",
            Operation::Below => "below",
            Operation::AtLeast => "at-least",
            Operation::Not => "not",
            Operation::Load => "load",
            Operation::Deref => "deref",
            Operation::Call => "call",
            Operation::Select => "select",
            Operation::Acquire => "acquire",
            Operation::Release => "release",
            Operation::Caught => "caught",
            Operation::Throw => "throw",
            Operation::Expanded => "expanded",
        }
    }
}

impl VerifyDialect for Toy {
    fn verify(_function: &Function<Self>, _issues: &mut Vec<VerificationIssue>) {}
}

/// Medium-level operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MediumOperation {
    Constant(i64),
    Copy,
    Add,
    LessThan,
    Not,
    Load,
    Call,
    Store,
    Exchange,
    Branch,
    CompareBranch,
    Switch,
    Jump,
    Return,
    /// A read-modify-write merge: operand 1 reads the destination's
    /// previous value.
    Merge,
    /// A value that the dialect requires as a visible statement.
    Materialized,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Edge {
    Entry,
    True,
    False,
    Fall,
    Jump,
    Case(i64),
    Except,
}

impl mlil::Dialect for Toy {
    type Operation = MediumOperation;
    type Edge = Edge;

    fn instruction_metadata(
        operation: &Self::Operation,
        may_throw: bool,
    ) -> mlil::InstructionMetadata<Self::Effect> {
        let (effects, flow) = match operation {
            MediumOperation::Call | MediumOperation::Store => {
                (vec![Effect::Write], FlowEffect::Fallthrough)
            }
            MediumOperation::Load => (vec![Effect::Read], FlowEffect::Fallthrough),
            MediumOperation::Branch | MediumOperation::CompareBranch => {
                (Vec::new(), FlowEffect::ConditionalJump)
            }
            MediumOperation::Switch => (Vec::new(), FlowEffect::IndirectJump),
            MediumOperation::Jump => (Vec::new(), FlowEffect::Jump),
            MediumOperation::Return => (Vec::new(), FlowEffect::Return),
            _ => (Vec::new(), FlowEffect::Fallthrough),
        };
        mlil::InstructionMetadata::new(effects, flow, may_throw)
    }

    fn mnemonic(operation: &Self::Operation) -> &str {
        match operation {
            MediumOperation::Constant(_) => "const",
            MediumOperation::Copy => "copy",
            MediumOperation::Add => "add",
            MediumOperation::LessThan => "lt",
            MediumOperation::Not => "not",
            MediumOperation::Load => "load",
            MediumOperation::Call => "call",
            MediumOperation::Store => "store",
            MediumOperation::Exchange => "exchange",
            MediumOperation::Branch => "branch",
            MediumOperation::CompareBranch => "compare_branch",
            MediumOperation::Switch => "switch",
            MediumOperation::Jump => "jump",
            MediumOperation::Return => "return",
            MediumOperation::Merge => "merge",
            MediumOperation::Materialized => "materialized",
        }
    }

    fn edge_kind(edge: &Self::Edge) -> EdgeKind {
        match edge {
            Edge::Entry | Edge::Fall => EdgeKind::Fallthrough,
            Edge::True => EdgeKind::ConditionalTrue,
            Edge::False => EdgeKind::ConditionalFalse,
            Edge::Jump => EdgeKind::Jump,
            Edge::Case(_) => EdgeKind::SwitchCase,
            Edge::Except => EdgeKind::ExceptionHandler,
        }
    }

    fn is_entry_edge(edge: &Self::Edge) -> bool {
        *edge == Edge::Entry
    }
}

impl mlil::AnalysisDialect for Toy {
    type Constant = i64;
    type ExpressionOperator = MediumOperation;
    type Callee = u32;

    fn is_copy(operation: &Self::Operation) -> bool {
        *operation == MediumOperation::Copy
    }

    fn expression_operator(operation: &Self::Operation) -> Option<Self::ExpressionOperator> {
        matches!(
            operation,
            MediumOperation::Add | MediumOperation::LessThan | MediumOperation::Copy
        )
        .then_some(*operation)
    }

    fn constant(operation: &Self::Operation) -> Option<Self::Constant> {
        match operation {
            MediumOperation::Constant(value) => Some(*value),
            _ => None,
        }
    }

    fn fold_constant(
        _instruction: &mlil::Instruction<Self>,
        _known: &alloc::collections::BTreeMap<mlil::VariableId, Self::Constant>,
    ) -> Option<(mlil::VariableId, Self::Constant)> {
        None
    }

    fn callee(_operation: &Self::Operation) -> Option<Self::Callee> {
        None
    }
}

impl mlil::VerifyDialect for Toy {
    fn verify(_function: &mlil::Function<Self>, _issues: &mut Vec<mlil::VerificationIssue>) {}
}

impl LiftDialect for Toy {
    fn negate_operation(operation: &Operation) -> Option<Operation> {
        match operation {
            Operation::Below => Some(Operation::AtLeast),
            Operation::AtLeast => Some(Operation::Below),
            _ => None,
        }
    }

    fn previous_value_operand(operation: &MediumOperation) -> Option<usize> {
        matches!(operation, MediumOperation::Merge).then_some(1)
    }

    fn materialize_value(operation: &MediumOperation) -> bool {
        matches!(operation, MediumOperation::Materialized)
    }

    fn lift_operation(operation: &MediumOperation) -> Lifted<Operation> {
        match operation {
            MediumOperation::Add => Lifted::Operation(Operation::Add),
            MediumOperation::LessThan => Lifted::Operation(Operation::LessThan),
            MediumOperation::Not => Lifted::Operation(Operation::Not),
            MediumOperation::Load => Lifted::Operation(Operation::Load),
            MediumOperation::Call | MediumOperation::Merge | MediumOperation::Materialized => {
                Lifted::Operation(Operation::Call)
            }
            MediumOperation::Store => Lifted::Store {
                location: Operation::Deref,
            },
            MediumOperation::Exchange => Lifted::ParallelCopy,
            MediumOperation::Branch => Lifted::Branch,
            MediumOperation::CompareBranch => Lifted::BranchOperation(Operation::Below),
            MediumOperation::Switch => Lifted::Switch,
            MediumOperation::Return => Lifted::Return,
            MediumOperation::Jump | MediumOperation::Constant(_) | MediumOperation::Copy => {
                Lifted::ControlFlow
            }
        }
    }

    fn case_values(edge: &Edge) -> Vec<i64> {
        match edge {
            Edge::Case(value) => vec![*value],
            _ => Vec::new(),
        }
    }

    fn void_type() -> Type {
        Type::Void
    }

    fn logical_not() -> Option<Operation> {
        Some(Operation::Not)
    }

    fn temporary_role() -> Option<u8> {
        Some(1)
    }

    fn evaluation_commutes(
        moved_effects: &[Effect],
        moved_may_throw: bool,
        crossed_effects: &[Effect],
        crossed_may_throw: bool,
    ) -> bool {
        // Reads pass reads; nothing passes a write or a potential throw.
        !moved_may_throw
            && !crossed_may_throw
            && moved_effects.iter().all(|effect| *effect == Effect::Read)
            && crossed_effects.iter().all(|effect| *effect == Effect::Read)
    }
}

impl RecoverDialect for Toy {
    fn select() -> Option<Operation> {
        Some(Operation::Select)
    }

    fn single_expression_operation(operation: &Operation) -> bool {
        *operation != Operation::Expanded
    }

    fn region_enter(operation: &Operation) -> Option<Operation> {
        matches!(operation, Operation::Acquire).then_some(Operation::Acquire)
    }

    fn releases(enter: &Operation, exit: &Operation) -> bool {
        matches!(enter, Operation::Acquire) && matches!(exit, Operation::Release)
    }

    fn is_exception_materialization(operation: &Operation) -> bool {
        matches!(operation, Operation::Caught)
    }

    fn is_throw(operation: &Operation) -> bool {
        matches!(operation, Operation::Throw)
    }
}

impl LowerDialect for Toy {
    fn lower_operation(operation: &Operation) -> MediumOperation {
        match operation {
            Operation::Add | Operation::Select => MediumOperation::Add,
            Operation::LessThan
            | Operation::Below
            | Operation::AtLeast
            | Operation::Acquire
            | Operation::Release
            | Operation::Caught
            | Operation::Throw => MediumOperation::LessThan,
            Operation::Not => MediumOperation::Not,
            Operation::Load | Operation::Deref => MediumOperation::Load,
            Operation::Call | Operation::Expanded => MediumOperation::Call,
        }
    }

    fn lower_constant(constant: &i64) -> MediumOperation {
        MediumOperation::Constant(*constant)
    }

    fn copy_operation() -> MediumOperation {
        MediumOperation::Copy
    }

    fn store_operation(_location: &Operation) -> MediumOperation {
        MediumOperation::Store
    }

    fn branch_operation() -> MediumOperation {
        MediumOperation::Branch
    }

    fn switch_operation() -> MediumOperation {
        MediumOperation::Switch
    }

    fn return_operation() -> MediumOperation {
        MediumOperation::Return
    }

    fn temporary_role() -> u8 {
        1
    }

    fn operation_may_throw(operation: &MediumOperation) -> bool {
        *operation == MediumOperation::Call
    }

    fn entry_edge() -> Edge {
        Edge::Entry
    }

    fn fallthrough_edge() -> Edge {
        Edge::Fall
    }

    fn jump_edge() -> Edge {
        Edge::Jump
    }

    fn true_edge() -> Edge {
        Edge::True
    }

    fn false_edge() -> Edge {
        Edge::False
    }

    fn case_edge(value: &i64) -> Edge {
        Edge::Case(*value)
    }

    fn default_edge() -> Edge {
        Edge::Fall
    }

    fn unwind_edge() -> Edge {
        Edge::Except
    }
}
