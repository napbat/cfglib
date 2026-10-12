//! A consumer-owned exceptional payload must remap its precise throw site.

extern crate alloc;

use alloc::vec::Vec;

use crate::ir::dialect::Vocabulary;
use crate::ir::mlil::{
    self, Function, FunctionBuilder, InstructionId, InstructionMetadata, VerificationIssue,
};
use crate::{EdgeKind, FlowEffect};

use super::{Operation, ToyDialect};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OwnedEdge {
    Entry,
    Next,
    True,
    False,
    Exception { owner: InstructionId },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Exact;

impl Vocabulary for Exact {
    type ValueType = <ToyDialect as Vocabulary>::ValueType;
    type Effect = <ToyDialect as Vocabulary>::Effect;
    type Source = <ToyDialect as Vocabulary>::Source;
    type SourceSpan = <ToyDialect as Vocabulary>::SourceSpan;
    type SourcePoint = <ToyDialect as Vocabulary>::SourcePoint;
    type VariableRole = <ToyDialect as Vocabulary>::VariableRole;
    type NativeVariable = <ToyDialect as Vocabulary>::NativeVariable;

    fn span_is_empty(span: &Self::SourceSpan) -> bool {
        ToyDialect::span_is_empty(span)
    }
    fn span_contains(span: &Self::SourceSpan, point: &Self::SourcePoint) -> bool {
        ToyDialect::span_contains(span, point)
    }
}

impl mlil::Dialect for Exact {
    type Operation = Operation;
    type Edge = OwnedEdge;

    fn instruction_metadata(
        operation: &Operation,
        may_throw: bool,
    ) -> InstructionMetadata<Self::Effect> {
        if matches!(operation, Operation::Branch) {
            InstructionMetadata::new(Vec::new(), FlowEffect::ConditionalJump, may_throw)
        } else {
            <ToyDialect as mlil::Dialect>::instruction_metadata(operation, may_throw)
        }
    }
    fn mnemonic(operation: &Operation) -> &str {
        <ToyDialect as mlil::Dialect>::mnemonic(operation)
    }
    fn edge_kind(edge: &OwnedEdge) -> EdgeKind {
        match edge {
            OwnedEdge::Entry | OwnedEdge::Next => EdgeKind::Fallthrough,
            OwnedEdge::True => EdgeKind::ConditionalTrue,
            OwnedEdge::False => EdgeKind::ConditionalFalse,
            OwnedEdge::Exception { .. } => EdgeKind::ExceptionUnwind,
        }
    }
    fn is_entry_edge(edge: &OwnedEdge) -> bool {
        *edge == OwnedEdge::Entry
    }
}

impl mlil::VerifyDialect for Exact {
    fn verify(function: &Function<Self>, issues: &mut Vec<VerificationIssue>) {
        for edge in function.cfg().edges() {
            if let OwnedEdge::Exception { owner } = edge.payload() {
                let owned = function
                    .instruction(*owner)
                    .is_some_and(mlil::Instruction::may_throw)
                    && function
                        .instruction_point(*owner)
                        .is_some_and(|point| point.block == edge.source());
                if !owned {
                    issues.push(VerificationIssue::new(
                        "exceptional payload lost its exact throwing owner",
                    ));
                }
            }
        }
    }
}

#[test]
fn exceptional_payload_remaps_the_exact_owner_and_rejects_stale_identity() {
    let mut builder = FunctionBuilder::<Exact>::new("precise-exception".into());
    let orphan = builder.new_block("unreachable source prefix");
    let head = builder.new_block("throw then decide");
    let yes = builder.new_block("selected outcome");
    let no = builder.new_block("retired outcome");
    let handler = builder.new_block("exceptional exit");
    builder
        .append_instruction(
            orphan,
            Operation::Return,
            Vec::new(),
            Vec::new(),
            false,
            None,
        )
        .unwrap();
    let owner = builder
        .append_instruction(
            head,
            Operation::Load(0),
            Vec::new(),
            Vec::new(),
            true,
            Some(crate::test_util::toy::Span { start: 10, end: 11 }),
        )
        .unwrap();
    builder
        .append_instruction(head, Operation::Branch, Vec::new(), Vec::new(), false, None)
        .unwrap();
    for block in [yes, no, handler] {
        builder
            .append_instruction(
                block,
                Operation::Return,
                Vec::new(),
                Vec::new(),
                false,
                None,
            )
            .unwrap();
    }
    builder
        .add_edge(builder.entry(), head, OwnedEdge::Entry, None)
        .unwrap();
    builder.add_edge(head, yes, OwnedEdge::True, None).unwrap();
    builder.add_edge(head, no, OwnedEdge::False, None).unwrap();
    let unwind = builder
        .add_edge(head, handler, OwnedEdge::Exception { owner }, None)
        .unwrap();
    let function = builder.finish().unwrap();
    let decide = |instruction: &mlil::Instruction<Exact>| {
        matches!(instruction.operation(), Operation::Branch).then_some(true)
    };
    let result = function
        .reduce_constant_control_flow(decide, |edge, mapping, selected| {
            Ok(if selected {
                OwnedEdge::Next
            } else if let OwnedEdge::Exception { owner } = edge {
                OwnedEdge::Exception {
                    owner: mapping.instruction(*owner).ok_or_else(|| {
                        mlil::Error::InvalidConstruction("throw site was retired".into())
                    })?,
                }
            } else {
                *edge
            })
        })
        .unwrap()
        .unwrap();
    let current_owner = result.mapping.instruction(owner).unwrap();
    assert_ne!(owner, current_owner);
    assert_eq!(
        result.function.cfg().edge(unwind).payload(),
        &OwnedEdge::Exception {
            owner: current_owner
        }
    );
    assert_eq!(
        result
            .function
            .instruction(current_owner)
            .unwrap()
            .operation(),
        &Operation::Load(0)
    );
    assert_eq!(
        result
            .function
            .instruction_point(current_owner)
            .unwrap()
            .block,
        head
    );
    assert_eq!(result.function.source(), function.source());
    assert!(
        result
            .function
            .provenance()
            .mappings_to(mlil::EntityId::Instruction(current_owner))
            .any(|entry| entry.source.start == 10 && entry.source.end == 11)
    );
    assert!(result.function.cfg().block(no).is_empty());
    assert!(
        result
            .function
            .cfg()
            .block(handler)
            .instructions()
            .iter()
            .any(|instruction| instruction.operation() == &Operation::Return)
    );
    assert!(
        function
            .reduce_constant_control_flow(decide, |edge, _, selected| Ok(if selected {
                OwnedEdge::Next
            } else {
                *edge
            }))
            .is_err()
    );
}
