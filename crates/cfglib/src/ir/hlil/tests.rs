extern crate alloc;

use alloc::string::ToString;
use alloc::vec;
use alloc::vec::Vec;

use crate::ir::mlil;
use crate::test_util::golden::assert_golden;
use crate::test_util::toy::Span;

use super::{
    ExpressionKind, FunctionBuilder, LiftMetadata, Signature, StatementKind, lift_function,
    lift_function_with_metadata, lift_function_with_structure,
};

/// The toy dialect every test here is written against.
mod dialect;

use dialect::{Edge, MediumOperation, Operation, Toy, Type};

#[test]
fn compound_assignments_are_recognized_structurally() {
    let mut builder = FunctionBuilder::<Toy>::new("toy::compound".into());
    let counter = builder
        .declare_variable(0, Some(7), Some(Type::Integer))
        .unwrap();
    let other = builder
        .declare_variable(0, None, Some(Type::Integer))
        .unwrap();

    let read = builder
        .add_expression(ExpressionKind::Variable(counter), Type::Integer)
        .unwrap();
    let one = builder
        .add_expression(ExpressionKind::Constant(1), Type::Integer)
        .unwrap();
    let sum = builder
        .add_expression(
            ExpressionKind::Operation {
                operation: Operation::Add,
                operands: vec![read, one],
            },
            Type::Integer,
        )
        .unwrap();
    let target = builder
        .add_expression(ExpressionKind::Variable(counter), Type::Integer)
        .unwrap();
    let compound = builder
        .add_statement(StatementKind::Assign { target, value: sum }, None)
        .unwrap();

    // The same shape assigning into a different variable is not compound.
    let read_other = builder
        .add_expression(ExpressionKind::Variable(counter), Type::Integer)
        .unwrap();
    let two = builder
        .add_expression(ExpressionKind::Constant(2), Type::Integer)
        .unwrap();
    let plain_sum = builder
        .add_expression(
            ExpressionKind::Operation {
                operation: Operation::Add,
                operands: vec![read_other, two],
            },
            Type::Integer,
        )
        .unwrap();
    let other_target = builder
        .add_expression(ExpressionKind::Variable(other), Type::Integer)
        .unwrap();
    let plain = builder
        .add_statement(
            StatementKind::Assign {
                target: other_target,
                value: plain_sum,
            },
            None,
        )
        .unwrap();
    builder.set_body(vec![compound, plain]).unwrap();
    let function = builder.finish().unwrap();
    assert!(function.verify().is_ok());

    let (operation, operand) = function
        .compound_assignment(compound)
        .expect("counter = counter + 1 is compound");
    assert_eq!(operation, &Operation::Add);
    assert_eq!(operand, one);
    assert!(
        function.compound_assignment(plain).is_none(),
        "the first operand reads a different variable than the target"
    );

    assert!(function.expressions_equal(read, target));
    assert!(!function.expressions_equal(read, other_target));
    assert!(!function.expressions_equal(sum, plain_sum));
}

#[test]
fn builder_constructs_verifies_and_renders() {
    let mut builder = FunctionBuilder::<Toy>::new("toy::bump".into());
    let counter = builder
        .declare_variable(0, Some(7), Some(Type::Integer))
        .unwrap();
    let read = builder
        .add_expression(ExpressionKind::Variable(counter), Type::Integer)
        .unwrap();
    let ten = builder
        .add_expression(ExpressionKind::Constant(10), Type::Integer)
        .unwrap();
    let compare = builder
        .add_expression(
            ExpressionKind::Operation {
                operation: Operation::LessThan,
                operands: vec![read, ten],
            },
            Type::Boolean,
        )
        .unwrap();
    let read_again = builder
        .add_expression(ExpressionKind::Variable(counter), Type::Integer)
        .unwrap();
    let one = builder
        .add_expression(ExpressionKind::Constant(1), Type::Integer)
        .unwrap();
    let sum = builder
        .add_expression(
            ExpressionKind::Operation {
                operation: Operation::Add,
                operands: vec![read_again, one],
            },
            Type::Integer,
        )
        .unwrap();
    let target = builder
        .add_expression(ExpressionKind::Variable(counter), Type::Integer)
        .unwrap();
    let assign = builder
        .add_statement(
            StatementKind::Assign { target, value: sum },
            Some(Span { start: 4, end: 9 }),
        )
        .unwrap();
    let conditional = builder
        .add_statement(
            StatementKind::If {
                condition: compare,
                then_body: vec![assign],
                else_body: Vec::new(),
            },
            None,
        )
        .unwrap();
    let result = builder
        .add_expression(ExpressionKind::Variable(counter), Type::Integer)
        .unwrap();
    let return_statement = builder
        .add_statement(
            StatementKind::Return {
                values: vec![result],
            },
            None,
        )
        .unwrap();
    builder
        .set_signature(Signature::<Toy>::new(vec![counter], vec![Type::Integer]))
        .unwrap();
    builder
        .set_body(vec![conditional, return_statement])
        .unwrap();

    let function = builder.finish().unwrap();
    assert!(function.verify().is_ok());
    assert_eq!(function.source(), "toy::bump");
    assert_eq!(function.signature().parameters, vec![counter]);

    assert_golden("hlil/builder-if-loop.pseudo", &function.to_pseudocode());

    assert_eq!(function.provenance().mappings_from(5).count(), 1);
}

#[test]
fn shared_expressions_and_orphans_are_rejected() {
    let mut builder = FunctionBuilder::<Toy>::new("toy::shared".into());
    let variable = builder.declare_variable(0, None, None).unwrap();
    let read = builder
        .add_expression(ExpressionKind::Variable(variable), Type::Integer)
        .unwrap();
    builder
        .add_statement(StatementKind::Return { values: vec![read] }, None)
        .unwrap();
    let second = builder
        .add_statement(StatementKind::Return { values: vec![read] }, None)
        .unwrap();
    builder.set_body(vec![second]).unwrap();

    let error = builder.finish().unwrap_err().to_string();
    assert!(error.contains("referenced 2 times"), "{error}");
    assert!(error.contains("referenced 0 times"), "{error}");
}

#[test]
fn transfers_need_matching_context() {
    let mut builder = FunctionBuilder::<Toy>::new("toy::transfers".into());
    let stray_break = builder
        .add_statement(StatementKind::Break { label: None }, None)
        .unwrap();
    let stray_goto = builder
        .add_statement(
            StatementKind::Goto {
                label: "missing".into(),
            },
            None,
        )
        .unwrap();
    builder.set_body(vec![stray_break, stray_goto]).unwrap();

    let error = builder.finish().unwrap_err().to_string();
    assert!(
        error.contains("breaks outside any loop or switch"),
        "{error}"
    );
    assert!(error.contains("targets undefined label missing"), "{error}");
}

/// A machine-shaped counting loop:
/// `while (i < n) { i = i + 1 }; return i;` in flat MLIL.
fn counting_loop() -> mlil::Function<Toy> {
    let mut builder = mlil::FunctionBuilder::<Toy>::new("toy::count".into());
    let header = builder.new_block("header");
    let body = builder.new_block("body");
    let exit = builder.new_block("exit");
    let i = builder.declare_variable(0, None).unwrap();
    let n = builder.declare_variable(0, None).unwrap();
    let t_cond = builder.declare_variable(1, None).unwrap();
    let t_sum = builder.declare_variable(1, None).unwrap();
    let typed = |variable, value_type| mlil::TypedVariable::<Toy>::new(variable, value_type);

    builder
        .append_instruction(
            header,
            MediumOperation::LessThan,
            vec![typed(i, Type::Integer), typed(n, Type::Integer)],
            vec![typed(t_cond, Type::Boolean)],
            false,
            None,
        )
        .unwrap();
    builder
        .append_instruction(
            header,
            MediumOperation::Branch,
            vec![typed(t_cond, Type::Boolean)],
            Vec::new(),
            false,
            None,
        )
        .unwrap();
    builder
        .append_instruction(
            body,
            MediumOperation::Constant(1),
            Vec::new(),
            vec![typed(t_sum, Type::Integer)],
            false,
            None,
        )
        .unwrap();
    let add = builder
        .append_instruction(
            body,
            MediumOperation::Add,
            vec![typed(i, Type::Integer), typed(t_sum, Type::Integer)],
            vec![typed(i, Type::Integer)],
            false,
            Some(Span { start: 10, end: 20 }),
        )
        .unwrap();
    builder
        .append_instruction(
            body,
            MediumOperation::Jump,
            Vec::new(),
            Vec::new(),
            false,
            None,
        )
        .unwrap();
    builder
        .append_instruction(
            exit,
            MediumOperation::Return,
            vec![typed(i, Type::Integer)],
            Vec::new(),
            false,
            None,
        )
        .unwrap();
    builder
        .add_edge(builder.entry(), header, Edge::Entry, None)
        .unwrap();
    builder.add_edge(header, body, Edge::True, None).unwrap();
    builder.add_edge(header, exit, Edge::False, None).unwrap();
    builder.add_edge(body, header, Edge::Fall, None).unwrap();
    builder
        .set_signature(mlil::Signature::<Toy>::new(vec![i, n], vec![Type::Integer]))
        .unwrap();
    let function = builder.finish().unwrap();
    // Sanity: the lifted instruction map below keys off this id.
    assert_eq!(add.index(), 3);
    function
}

#[test]
fn lift_recovers_a_while_loop_with_inlined_expressions() {
    let source = counting_loop();
    let lifted = lift_function(&source).unwrap();
    assert!(lifted.report.is_fully_structured(), "{:?}", lifted.report);
    assert!(lifted.function.verify().is_ok());

    // The comparison and the constant are inlined into the header, so the
    // golden shows the loop with no surviving temporaries.
    assert_golden(
        "hlil/lift-while-loop.pseudo",
        &lifted.function.to_pseudocode(),
    );

    // Signature and variables carried over one-to-one.
    assert_eq!(lifted.function.signature().parameters.len(), 2);
    assert_eq!(lifted.function.variables().len(), source.variables().len());

    // The add instruction's provenance span survived onto its statement.
    assert_eq!(lifted.function.provenance().mappings_from(12).count(), 1);
    assert!(
        lifted
            .instructions
            .contains_key(&mlil::InstructionId::from_raw(3)),
        "{:?}",
        lifted.instructions
    );
}

#[test]
fn lift_reuses_owned_or_borrowed_structure_with_identical_output() {
    let source = counting_loop();
    let (owned, owned_report) = source.structured_control_flow_with_report();
    let (borrowed, borrowed_report) = crate::lift_borrowed_with_report(source.cfg());

    let from_owned =
        lift_function_with_structure(&source, &owned, &owned_report, LiftMetadata::Preserve)
            .unwrap();
    let from_borrowed =
        lift_function_with_structure(&source, &borrowed, &borrowed_report, LiftMetadata::Preserve)
            .unwrap();

    assert_eq!(from_borrowed.report, from_owned.report);
    assert_eq!(from_borrowed.function, from_owned.function);
    assert_eq!(from_borrowed.instructions, from_owned.instructions);
}

#[test]
fn lift_can_omit_correspondence_and_provenance() {
    let source = counting_loop();
    let lifted = lift_function_with_metadata(&source, LiftMetadata::Omit).unwrap();

    assert!(lifted.instructions.is_empty());
    assert!(lifted.function.provenance().is_empty());
    assert!(lifted.function.verify().is_ok());
    // Omitting the metadata changes nothing a reader sees: the same golden.
    assert_golden(
        "hlil/lift-while-loop.pseudo",
        &lifted.function.to_pseudocode(),
    );
}

#[test]
fn lift_recovers_a_switch_with_case_values_and_default() {
    let mut builder = mlil::FunctionBuilder::<Toy>::new("toy::dispatch".into());
    let dispatch = builder.new_block("dispatch");
    let case_one = builder.new_block("one");
    let case_two = builder.new_block("two");
    let fallback = builder.new_block("fallback");
    let merge = builder.new_block("merge");
    let selector = builder.declare_variable(0, None).unwrap();
    let result = builder.declare_variable(0, None).unwrap();
    let typed = |variable| mlil::TypedVariable::<Toy>::new(variable, Type::Integer);

    builder
        .append_instruction(
            dispatch,
            MediumOperation::Switch,
            vec![typed(selector)],
            Vec::new(),
            false,
            None,
        )
        .unwrap();
    for (block, value) in [(case_one, 10), (case_two, 20), (fallback, 30)] {
        builder
            .append_instruction(
                block,
                MediumOperation::Constant(value),
                Vec::new(),
                vec![typed(result)],
                false,
                None,
            )
            .unwrap();
    }
    builder
        .append_instruction(
            merge,
            MediumOperation::Return,
            vec![typed(result)],
            Vec::new(),
            false,
            None,
        )
        .unwrap();
    builder
        .add_edge(builder.entry(), dispatch, Edge::Entry, None)
        .unwrap();
    builder
        .add_edge(dispatch, case_one, Edge::Case(1), None)
        .unwrap();
    builder
        .add_edge(dispatch, case_one, Edge::Case(2), None)
        .unwrap();
    builder
        .add_edge(dispatch, case_two, Edge::Case(3), None)
        .unwrap();
    builder
        .add_edge(dispatch, fallback, Edge::Fall, None)
        .unwrap();
    builder.add_edge(case_one, merge, Edge::Fall, None).unwrap();
    builder.add_edge(case_two, merge, Edge::Fall, None).unwrap();
    builder.add_edge(fallback, merge, Edge::Fall, None).unwrap();

    let lifted = lift_function(&builder.finish().unwrap()).unwrap();
    assert!(lifted.report.is_fully_structured(), "{:?}", lifted.report);
    assert_golden("hlil/lift-switch.pseudo", &lifted.function.to_pseudocode());
}

#[test]
fn lift_structures_declared_exception_regions() {
    let mut builder = mlil::FunctionBuilder::<Toy>::new("toy::guarded".into());
    let protected = builder.new_block("protected");
    let pad = builder.new_block("pad");
    let after = builder.new_block("after");
    let x = builder.declare_variable(0, None).unwrap();
    let fallback = builder.declare_variable(0, None).unwrap();
    let typed = |variable| mlil::TypedVariable::<Toy>::new(variable, Type::Integer);

    builder
        .append_instruction(
            protected,
            MediumOperation::Call,
            Vec::new(),
            vec![typed(x)],
            true,
            None,
        )
        .unwrap();
    builder
        .append_instruction(
            pad,
            MediumOperation::Constant(7),
            Vec::new(),
            vec![typed(fallback)],
            false,
            None,
        )
        .unwrap();
    builder
        .append_instruction(
            pad,
            MediumOperation::Return,
            vec![typed(fallback)],
            Vec::new(),
            false,
            None,
        )
        .unwrap();
    builder
        .append_instruction(
            after,
            MediumOperation::Return,
            vec![typed(x)],
            Vec::new(),
            false,
            None,
        )
        .unwrap();
    builder
        .add_edge(builder.entry(), protected, Edge::Entry, None)
        .unwrap();
    builder
        .add_edge(protected, after, Edge::Fall, None)
        .unwrap();
    builder
        .add_edge(protected, pad, Edge::Except, None)
        .unwrap();
    builder
        .add_region(crate::Region {
            id: crate::RegionId::from_raw(0),
            protected_blocks: [protected].into_iter().collect(),
            handlers: vec![crate::Handler {
                entry: pad,
                body: crate::HandlerBody::known([pad]),
                kind: crate::HandlerKind::CatchAll,
            }],
            parent: None,
        })
        .unwrap();

    let lifted = lift_function(&builder.finish().unwrap()).unwrap();
    assert_golden(
        "hlil/lift-try-catch.pseudo",
        &lifted.function.to_pseudocode(),
    );
}
#[test]
fn variable_splitting_composes_with_lifting() {
    // One storage slot reused for two lifetimes. Each lifetime feeds two
    // effectful consumers, so neither definition disappears or inlines —
    // the decompiler shape variable splitting exists for.
    let mut builder = mlil::FunctionBuilder::<Toy>::new("toy::slots".into());
    let block = builder.new_block("body");
    let slot = builder.declare_variable(0, Some(7)).unwrap();
    let reads: Vec<_> = (0..4)
        .map(|_| builder.declare_variable(0, None).unwrap())
        .collect();
    let typed = |variable| mlil::TypedVariable::<Toy>::new(variable, Type::Integer);
    for (value, pair) in [(1, &reads[0..2]), (2, &reads[2..4])] {
        builder
            .append_instruction(
                block,
                MediumOperation::Constant(value),
                Vec::new(),
                vec![typed(slot)],
                false,
                None,
            )
            .unwrap();
        for &read in pair {
            builder
                .append_instruction(
                    block,
                    MediumOperation::Copy,
                    vec![typed(slot)],
                    vec![typed(read)],
                    false,
                    None,
                )
                .unwrap();
            builder
                .append_instruction(
                    block,
                    MediumOperation::Call,
                    vec![typed(read)],
                    Vec::new(),
                    false,
                    None,
                )
                .unwrap();
        }
    }
    builder
        .append_instruction(
            block,
            MediumOperation::Return,
            vec![typed(reads[3])],
            Vec::new(),
            false,
            None,
        )
        .unwrap();
    builder
        .add_edge(builder.entry(), block, Edge::Entry, None)
        .unwrap();
    let function = builder.finish().unwrap();

    let split = function.split_variables().unwrap();
    assert_eq!(split.splits[&mlil::VariableId::from_raw(0)].len(), 2);
    let lifted = lift_function(&split.function).unwrap();
    // Each lifetime of the split variable is assigned to its own local.
    assert_golden(
        "hlil/split-variables.pseudo",
        &lifted.function.to_pseudocode(),
    );
}

/// HLIL → MLIL lowering tests, split out to respect the source-size policy.
mod lowering;

/// Local dead-value pruning tests for the presentation lift.
mod dead;

/// Effect-ordered inlining tests, split out to respect the source-size
/// policy.
mod effects;

/// Parallel-copy and fused-branch translation tests, split out to respect
/// the source-size policy.
mod fused;

/// Structural recovery tests, split out to respect the source-size policy.
mod recover;

/// Pure-transfer trampoline tests, split out to respect the source-size
/// policy.
mod trampoline;
