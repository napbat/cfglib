extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

use crate::ir::dialect::Vocabulary;
use crate::test_util::toy::{self, Span};
use crate::{ConstValue, EdgeKind, FlowEffect};

use super::{
    AnalysisDialect, Dialect, EntityId, FunctionBuilder, Instruction, InstructionId,
    InstructionMetadata, TypedVariable, VariableId, VerificationIssue, VerifyDialect,
};

mod canonical;
mod coalesce;
mod memory;
mod promote;
mod removal;
mod rewrite;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum Type {
    Integer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Effect {
    Control,
    Read,
    Write,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Operation {
    Constant(i64),
    Copy,
    Load(u8),
    Store(u8),
    AddressOf(u8),
    Branch,
    Return,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Edge {
    Entry,
    Next,
    Unwind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ToyDialect;

impl Vocabulary for ToyDialect {
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

impl Dialect for ToyDialect {
    type Operation = Operation;
    type Edge = Edge;

    fn instruction_metadata(
        operation: &Self::Operation,
        may_throw: bool,
    ) -> InstructionMetadata<Self::Effect> {
        match operation {
            Operation::Return => {
                InstructionMetadata::new(vec![Effect::Control], FlowEffect::Return, false)
            }
            Operation::Branch => {
                InstructionMetadata::new(vec![Effect::Control], FlowEffect::ConditionalJump, false)
            }
            Operation::Load(_) => {
                InstructionMetadata::new(vec![Effect::Read], FlowEffect::Fallthrough, may_throw)
            }
            Operation::Store(_) => {
                InstructionMetadata::new(vec![Effect::Write], FlowEffect::Fallthrough, false)
            }
            Operation::Constant(_) | Operation::Copy | Operation::AddressOf(_) => {
                InstructionMetadata::new(Vec::new(), FlowEffect::Fallthrough, false)
            }
        }
    }

    fn mnemonic(operation: &Self::Operation) -> &str {
        match operation {
            Operation::Constant(_) => "const",
            Operation::Copy => "copy",
            Operation::Load(_) => "load",
            Operation::Store(_) => "store",
            Operation::AddressOf(_) => "addressof",
            Operation::Branch => "branch",
            Operation::Return => "return",
        }
    }

    fn edge_kind(edge: &Self::Edge) -> EdgeKind {
        match edge {
            Edge::Entry | Edge::Next => EdgeKind::Fallthrough,
            Edge::Unwind => EdgeKind::ExceptionUnwind,
        }
    }

    fn is_entry_edge(edge: &Self::Edge) -> bool {
        *edge == Edge::Entry
    }
}

impl AnalysisDialect for ToyDialect {
    type Constant = i64;
    type ExpressionOperator = Operation;
    type Callee = u32;

    fn is_copy(operation: &Self::Operation) -> bool {
        *operation == Operation::Copy
    }

    fn expression_operator(operation: &Self::Operation) -> Option<Self::ExpressionOperator> {
        matches!(operation, Operation::Copy).then_some(*operation)
    }

    fn constant(operation: &Self::Operation) -> Option<Self::Constant> {
        let Operation::Constant(value) = operation else {
            return None;
        };
        Some(*value)
    }

    fn fold_constant(
        instruction: &Instruction<Self>,
        known: &BTreeMap<VariableId, Self::Constant>,
    ) -> Option<(VariableId, Self::Constant)> {
        let destination = *instruction.defs().first()?;
        let value = match instruction.operation() {
            Operation::Constant(value) => *value,
            Operation::Copy => *known.get(instruction.uses().first()?)?,
            Operation::Load(_)
            | Operation::Store(_)
            | Operation::AddressOf(_)
            | Operation::Branch
            | Operation::Return => return None,
        };
        Some((destination, value))
    }

    fn callee(_operation: &Self::Operation) -> Option<Self::Callee> {
        None
    }
}

impl super::PromoteDialect for ToyDialect {
    type Location = u8;

    fn promotion_access(instruction: &Instruction<Self>) -> super::PromotionAccess<u8> {
        match instruction.operation() {
            Operation::Load(slot) => super::PromotionAccess::Load(*slot),
            Operation::Store(slot) => super::PromotionAccess::Store(*slot),
            Operation::AddressOf(slot) => super::PromotionAccess::Escape(vec![*slot]),
            _ => super::PromotionAccess::Unrelated,
        }
    }

    fn copy_operation() -> Operation {
        Operation::Copy
    }

    fn promoted_variable(location: &u8) -> (u8, Option<u8>) {
        (9, Some(*location))
    }
}

impl VerifyDialect for ToyDialect {
    fn verify(function: &super::Function<Self>, issues: &mut Vec<VerificationIssue>) {
        for block_id in function.cfg().block_ids().skip(1) {
            let block = function.cfg().block(block_id);
            if function.cfg().outgoing(block_id).next().is_none()
                && !matches!(
                    block.instructions().last().map(Instruction::operation),
                    Some(Operation::Return)
                )
            {
                issues.push(VerificationIssue::new(format!(
                    "block {block_id} does not end in return"
                )));
            }
        }
    }
}

#[test]
fn generic_dialect_builds_verifies_and_uses_analyses() {
    let mut builder = FunctionBuilder::<ToyDialect>::new("toy::answer".into());
    let body = builder.new_block("body");
    let value = builder.declare_variable(0, Some(7)).unwrap();
    builder
        .append_instruction(
            body,
            Operation::Constant(42),
            Vec::new(),
            vec![TypedVariable::new(value, Type::Integer)],
            false,
            Some(Span { start: 4, end: 5 }),
        )
        .unwrap();
    builder
        .append_instruction(
            body,
            Operation::Return,
            vec![TypedVariable::new(value, Type::Integer)],
            Vec::new(),
            false,
            Some(Span { start: 5, end: 6 }),
        )
        .unwrap();
    builder
        .add_edge(builder.entry(), body, Edge::Entry, None)
        .unwrap();

    let function = builder.finish().unwrap();
    assert!(function.verify().is_ok());
    assert_eq!(function.source(), "toy::answer");
    assert_eq!(function.ssa().unwrap().blocks().len(), 2);
    assert_eq!(
        function.constants().fact_out(body).get(&value),
        Some(&ConstValue::Const(42))
    );
    assert_eq!(function.provenance().mappings_from(4).count(), 1);
}

#[test]
fn signature_and_region_survive_construction() {
    let mut builder = FunctionBuilder::<ToyDialect>::new("toy::signature".into());
    let body = builder.new_block("body");
    let pad = builder.new_block("pad");
    let argument = builder.declare_variable(0, None).unwrap();
    builder
        .append_instruction(
            body,
            Operation::Return,
            vec![TypedVariable::new(argument, Type::Integer)],
            Vec::new(),
            false,
            None,
        )
        .unwrap();
    builder
        .append_instruction(pad, Operation::Return, Vec::new(), Vec::new(), false, None)
        .unwrap();
    builder
        .add_edge(builder.entry(), body, Edge::Entry, None)
        .unwrap();
    builder
        .set_signature(super::Signature::<ToyDialect>::new(
            vec![argument],
            vec![Type::Integer],
        ))
        .unwrap();
    let region = builder
        .add_region(crate::Region {
            id: crate::RegionId::from_raw(0),
            protected_blocks: [body].into_iter().collect(),
            handlers: vec![crate::Handler {
                entry: pad,
                body: crate::HandlerBody::known([pad]),
                kind: crate::HandlerKind::CatchAll,
            }],
            parent: None,
        })
        .unwrap();
    let handler = crate::HandlerRef::new(region, 0);
    builder.set_cleanup_resume(handler, pad).unwrap();
    builder
        .add_continuation(
            handler,
            crate::Continuation {
                reason: crate::CompletionReason::Return,
                resume: body,
            },
        )
        .unwrap();

    let function = builder.finish().unwrap();
    assert_eq!(function.signature().parameters, vec![argument]);
    assert_eq!(function.signature().returns, vec![Type::Integer]);
    assert_eq!(function.cfg().regions().len(), 1);
    assert_eq!(function.cfg().regions()[0].id, region);
    assert_eq!(function.cfg().cleanups()[0].resume_from, Some(pad));
    assert_eq!(function.cfg().cleanups()[0].continuations[0].resume, body);
    assert_eq!(function.instructions().count(), 2);
}

#[test]
fn invalid_signatures_and_regions_are_rejected() {
    let mut builder = FunctionBuilder::<ToyDialect>::new("toy::invalid-metadata".into());
    let body = builder.new_block("body");
    let undeclared = VariableId::from_raw(7);
    let error = builder
        .set_signature(super::Signature::<ToyDialect>::new(
            vec![undeclared],
            Vec::new(),
        ))
        .unwrap_err()
        .to_string();
    assert!(error.contains("undeclared parameter"));

    let declared = builder.declare_variable(0, None).unwrap();
    let error = builder
        .set_signature(super::Signature::<ToyDialect>::new(
            vec![declared, declared],
            Vec::new(),
        ))
        .unwrap_err()
        .to_string();
    assert!(error.contains("repeats parameter"));

    let error = builder
        .add_region(crate::Region {
            id: crate::RegionId::from_raw(0),
            protected_blocks: [crate::BlockId::from_raw(9)].into_iter().collect(),
            handlers: Vec::new(),
            parent: None,
        })
        .unwrap_err()
        .to_string();
    assert!(error.contains("outside"));

    let error = builder
        .add_region(crate::Region {
            id: crate::RegionId::from_raw(0),
            protected_blocks: [body].into_iter().collect(),
            handlers: vec![crate::Handler {
                entry: body,
                body: crate::HandlerBody::known([builder.entry()]),
                kind: crate::HandlerKind::CatchAll,
            }],
            parent: None,
        })
        .unwrap_err()
        .to_string();
    assert!(error.contains("synthetic root"));
}

#[test]
fn split_variables_separates_independent_lifetimes() {
    let mut builder = FunctionBuilder::<ToyDialect>::new("toy::reuse".into());
    let body = builder.new_block("body");
    let slot = builder.declare_variable(0, Some(3)).unwrap();
    let first = builder.declare_variable(0, None).unwrap();
    let second = builder.declare_variable(0, None).unwrap();
    let typed = |variable| TypedVariable::new(variable, Type::Integer);
    builder
        .append_instruction(
            body,
            Operation::Constant(1),
            Vec::new(),
            vec![typed(slot)],
            false,
            None,
        )
        .unwrap();
    builder
        .append_instruction(
            body,
            Operation::Copy,
            vec![typed(slot)],
            vec![typed(first)],
            false,
            None,
        )
        .unwrap();
    builder
        .append_instruction(
            body,
            Operation::Constant(2),
            Vec::new(),
            vec![typed(slot)],
            false,
            None,
        )
        .unwrap();
    builder
        .append_instruction(
            body,
            Operation::Copy,
            vec![typed(slot)],
            vec![typed(second)],
            false,
            None,
        )
        .unwrap();
    builder
        .append_instruction(
            body,
            Operation::Return,
            vec![typed(second)],
            Vec::new(),
            false,
            None,
        )
        .unwrap();
    builder
        .add_edge(builder.entry(), body, Edge::Entry, None)
        .unwrap();
    let function = builder.finish().unwrap();

    let split = function.split_variables().unwrap();
    assert!(split.function.verify().is_ok());
    assert_eq!(split.splits[&slot].len(), 2, "{:?}", split.splits);
    assert_eq!(split.splits[&first].len(), 1);
    assert_eq!(split.splits[&second].len(), 1);

    let instruction = |index| {
        split
            .function
            .instruction(InstructionId::from_raw(index))
            .unwrap()
    };
    let first_lifetime = instruction(0).defs()[0];
    let second_lifetime = instruction(2).defs()[0];
    assert_ne!(first_lifetime, second_lifetime);
    assert_eq!(instruction(1).uses()[0], first_lifetime);
    assert_eq!(instruction(3).uses()[0], second_lifetime);
    assert_eq!(split.origins[first_lifetime.index()], slot);
    assert_eq!(split.origins[second_lifetime.index()], slot);
    // The split variables inherit role and native provenance.
    assert_eq!(
        split.function.variable(first_lifetime).unwrap().native,
        Some(3)
    );
}

#[test]
fn split_variables_keeps_phi_merged_values_together() {
    let mut builder = FunctionBuilder::<ToyDialect>::new("toy::diamond".into());
    let top = builder.new_block("top");
    let left = builder.new_block("left");
    let right = builder.new_block("right");
    let merge = builder.new_block("merge");
    let slot = builder.declare_variable(0, None).unwrap();
    let merged = builder.declare_variable(0, None).unwrap();
    let typed = |variable| TypedVariable::new(variable, Type::Integer);
    builder
        .append_instruction(
            top,
            Operation::Constant(0),
            Vec::new(),
            vec![typed(slot)],
            false,
            None,
        )
        .unwrap();
    builder
        .append_instruction(
            top,
            Operation::Branch,
            vec![typed(slot)],
            Vec::new(),
            false,
            None,
        )
        .unwrap();
    for (block, value) in [(left, 10), (right, 20)] {
        builder
            .append_instruction(
                block,
                Operation::Constant(value),
                Vec::new(),
                vec![typed(slot)],
                false,
                None,
            )
            .unwrap();
    }
    builder
        .append_instruction(
            merge,
            Operation::Copy,
            vec![typed(slot)],
            vec![typed(merged)],
            false,
            None,
        )
        .unwrap();
    builder
        .append_instruction(
            merge,
            Operation::Return,
            vec![typed(merged)],
            Vec::new(),
            false,
            None,
        )
        .unwrap();
    builder
        .add_edge(builder.entry(), top, Edge::Entry, None)
        .unwrap();
    builder.add_edge(top, left, Edge::Next, None).unwrap();
    builder.add_edge(top, right, Edge::Next, None).unwrap();
    builder.add_edge(left, merge, Edge::Next, None).unwrap();
    builder.add_edge(right, merge, Edge::Next, None).unwrap();
    let function = builder.finish().unwrap();

    let split = function.split_variables().unwrap();
    assert_eq!(split.splits[&slot].len(), 2, "{:?}", split.splits);
    let instruction = |index| {
        split
            .function
            .instruction(InstructionId::from_raw(index))
            .unwrap()
    };
    let top_value = instruction(0).defs()[0];
    let left_value = instruction(2).defs()[0];
    let right_value = instruction(3).defs()[0];
    let merge_use = instruction(4).uses()[0];
    assert_eq!(left_value, right_value, "phi operands share one web");
    assert_eq!(merge_use, left_value, "the merged use reads the web");
    assert_ne!(top_value, left_value, "the pre-branch lifetime is separate");
}

#[test]
fn split_variables_ignores_dead_loop_header_phis() {
    // A frontend that reuses one slot for unrelated lifetimes on both sides
    // of a loop produces a dead phi at the header (the body redefines the
    // slot before reading it). That phi must not merge the lifetimes.
    let mut builder = FunctionBuilder::<ToyDialect>::new("toy::loop".into());
    let top = builder.new_block("top");
    let header = builder.new_block("header");
    let body = builder.new_block("body");
    let exit = builder.new_block("exit");
    let slot = builder.declare_variable(0, Some(3)).unwrap();
    let first = builder.declare_variable(0, None).unwrap();
    let second = builder.declare_variable(0, None).unwrap();
    let typed = |variable| TypedVariable::new(variable, Type::Integer);
    let mut push = |block, operation, uses: Vec<_>, defs: Vec<_>| {
        builder
            .append_instruction(block, operation, uses, defs, false, None)
            .unwrap();
    };
    push(top, Operation::Constant(1), Vec::new(), vec![typed(slot)]);
    push(top, Operation::Copy, vec![typed(slot)], vec![typed(first)]);
    push(header, Operation::Branch, vec![typed(first)], Vec::new());
    push(body, Operation::Constant(2), Vec::new(), vec![typed(slot)]);
    push(
        body,
        Operation::Copy,
        vec![typed(slot)],
        vec![typed(second)],
    );
    push(exit, Operation::Return, vec![typed(first)], Vec::new());
    builder
        .add_edge(builder.entry(), top, Edge::Entry, None)
        .unwrap();
    for (source, target) in [
        (top, header),
        (header, body),
        (header, exit),
        (body, header),
    ] {
        builder.add_edge(source, target, Edge::Next, None).unwrap();
    }
    let function = builder.finish().unwrap();

    let split = function.split_variables().unwrap();
    assert!(split.function.verify().is_ok());
    assert_eq!(split.splits[&slot].len(), 2, "{:?}", split.splits);
    let instruction = |index| {
        split
            .function
            .instruction(InstructionId::from_raw(index))
            .unwrap()
    };
    let pre_loop = instruction(0).defs()[0];
    let in_loop = instruction(3).defs()[0];
    assert_ne!(pre_loop, in_loop, "the dead header phi merged lifetimes");
    assert_eq!(instruction(1).uses()[0], pre_loop);
    assert_eq!(instruction(4).uses()[0], in_loop);
    // Both lifetimes keep the original slot's provenance.
    assert_eq!(split.function.variable(pre_loop).unwrap().native, Some(3));
    assert_eq!(split.function.variable(in_loop).unwrap().native, Some(3));
}

#[test]
fn split_variables_preserves_parameters_identities_and_provenance() {
    let mut builder = FunctionBuilder::<ToyDialect>::new("toy::parameter".into());
    let body = builder.new_block("body");
    let parameter = builder.declare_variable(2, None).unwrap();
    let saved = builder.declare_variable(0, None).unwrap();
    let result = builder.declare_variable(0, None).unwrap();
    let typed = |variable| TypedVariable::new(variable, Type::Integer);
    builder
        .append_instruction(
            body,
            Operation::Copy,
            vec![typed(parameter)],
            vec![typed(saved)],
            false,
            None,
        )
        .unwrap();
    builder
        .append_instruction(
            body,
            Operation::Constant(5),
            Vec::new(),
            vec![typed(parameter)],
            false,
            Some(Span { start: 8, end: 9 }),
        )
        .unwrap();
    builder
        .append_instruction(
            body,
            Operation::Copy,
            vec![typed(parameter)],
            vec![typed(result)],
            false,
            None,
        )
        .unwrap();
    builder
        .append_instruction(
            body,
            Operation::Return,
            vec![typed(result)],
            Vec::new(),
            false,
            None,
        )
        .unwrap();
    builder
        .add_edge(builder.entry(), body, Edge::Entry, None)
        .unwrap();
    builder
        .set_signature(super::Signature::<ToyDialect>::new(
            vec![parameter],
            vec![Type::Integer],
        ))
        .unwrap();
    builder
        .map_entity(Span { start: 1, end: 2 }, EntityId::Variable(parameter))
        .unwrap();
    let function = builder.finish().unwrap();

    let split = function.split_variables().unwrap();
    assert_eq!(split.splits[&parameter].len(), 2, "{:?}", split.splits);
    let instruction = |index| {
        split
            .function
            .instruction(InstructionId::from_raw(index))
            .unwrap()
    };
    let incoming = instruction(0).uses()[0];
    let redefined = instruction(1).defs()[0];
    assert_ne!(incoming, redefined);
    assert_eq!(instruction(2).uses()[0], redefined);
    assert_eq!(
        split.function.signature().parameters,
        vec![incoming],
        "the signature names the live-in lifetime"
    );

    // Instruction identities survive, so instruction provenance does too.
    assert_eq!(
        split
            .function
            .provenance()
            .mappings_to(EntityId::Instruction(InstructionId::from_raw(1)))
            .count(),
        1
    );
    // The original variable's span now names both split lifetimes.
    assert_eq!(
        split
            .function
            .provenance()
            .mappings_from(1)
            .filter(|entry| matches!(entry.entity, EntityId::Variable(_)))
            .count(),
        2
    );
}

#[test]
fn generic_verifier_requires_one_root_entry() {
    let mut builder = FunctionBuilder::<ToyDialect>::new("toy::invalid".into());
    let body = builder.new_block("body");
    builder
        .append_instruction(body, Operation::Return, Vec::new(), Vec::new(), false, None)
        .unwrap();

    let error = builder.finish().unwrap_err().to_string();
    assert!(error.contains("synthetic root has 0 outgoing edges instead of one"));
}

#[test]
fn rejected_source_spans_do_not_partially_mutate_the_builder() {
    let mut builder = FunctionBuilder::<ToyDialect>::new("toy::atomic".into());
    let body = builder.new_block("body");
    let value = builder.declare_variable(0, None).unwrap();
    let invalid = Span { start: 3, end: 3 };

    assert!(
        builder
            .append_instruction(
                body,
                Operation::Constant(7),
                Vec::new(),
                vec![TypedVariable::new(value, Type::Integer)],
                false,
                Some(invalid),
            )
            .is_err()
    );
    assert!(
        builder
            .add_edge(builder.entry(), body, Edge::Entry, Some(invalid))
            .is_err()
    );

    let instruction = builder
        .append_instruction(
            body,
            Operation::Return,
            Vec::new(),
            Vec::new(),
            false,
            Some(Span { start: 3, end: 4 }),
        )
        .unwrap();
    builder
        .add_edge(builder.entry(), body, Edge::Entry, None)
        .unwrap();

    let function = builder.finish().unwrap();
    assert_eq!(instruction.index(), 0);
    assert_eq!(function.cfg().edge_count(), 1);
}

/// A linear protected chain for coverage extension: `body` throws into
/// `pad`, the pure `tail` and `tail2` blocks follow, and `after` throws
/// again. With `rejoin`, the handler falls back into `tail`, giving it a
/// sequential predecessor outside the protected set.
fn coverage_function(rejoin: bool) -> super::Function<ToyDialect> {
    let mut builder = FunctionBuilder::<ToyDialect>::new("toy::coverage".into());
    let body = builder.new_block("body");
    let tail = builder.new_block("tail");
    let tail2 = builder.new_block("tail2");
    let after = builder.new_block("after");
    let pad = builder.new_block("pad");
    let value = builder.declare_variable(0, None).unwrap();
    let typed = || vec![TypedVariable::new(value, Type::Integer)];
    let mut push = |block, operation, throws| {
        builder
            .append_instruction(block, operation, Vec::new(), typed(), throws, None)
            .unwrap();
    };
    push(body, Operation::Load(0), true);
    push(tail, Operation::Constant(1), false);
    push(tail2, Operation::Constant(2), false);
    push(after, Operation::Load(1), true);
    push(pad, Operation::Constant(9), false);
    builder
        .append_instruction(after, Operation::Return, typed(), Vec::new(), false, None)
        .unwrap();
    if rejoin {
        builder.add_edge(pad, tail, Edge::Next, None).unwrap();
    } else {
        builder
            .append_instruction(pad, Operation::Return, typed(), Vec::new(), false, None)
            .unwrap();
    }
    builder
        .add_edge(builder.entry(), body, Edge::Entry, None)
        .unwrap();
    builder.add_edge(body, tail, Edge::Next, None).unwrap();
    builder.add_edge(body, pad, Edge::Unwind, None).unwrap();
    builder.add_edge(tail, tail2, Edge::Next, None).unwrap();
    builder.add_edge(tail2, after, Edge::Next, None).unwrap();
    builder
        .add_region(crate::Region {
            id: crate::RegionId::from_raw(0),
            protected_blocks: [body].into_iter().collect(),
            handlers: vec![crate::Handler {
                entry: pad,
                body: crate::HandlerBody::Unknown,
                kind: crate::HandlerKind::CatchAll,
            }],
            parent: None,
        })
        .unwrap();
    builder.finish().unwrap()
}

#[test]
fn equivalent_coverage_extends_through_nonthrowing_tails() {
    let function = coverage_function(false);
    let derived = function.with_derived_cfg(super::extend_equivalent_coverage);
    let protected: Vec<&str> = derived.cfg().regions()[0]
        .protected_blocks
        .iter()
        .filter_map(|&block| derived.cfg().block(block).label())
        .collect();
    // `tail2` needs the second fixpoint round; `after` may throw, so the
    // declared coverage boundary before it is observable and stays.
    assert_eq!(protected, ["body", "tail", "tail2"]);
    // The canonical function keeps the exact declared coverage.
    assert_eq!(function.cfg().regions()[0].protected_blocks.len(), 1);
}

#[test]
fn equivalent_coverage_respects_unprotected_predecessors() {
    let function = coverage_function(true);
    let derived = function.with_derived_cfg(super::extend_equivalent_coverage);
    let protected: Vec<&str> = derived.cfg().regions()[0]
        .protected_blocks
        .iter()
        .filter_map(|&block| derived.cfg().block(block).label())
        .collect();
    // The handler rejoins at `tail`: entering it from `pad` is not covered
    // by the declared region, so absorbing it would be observable.
    assert_eq!(protected, ["body"]);
}
