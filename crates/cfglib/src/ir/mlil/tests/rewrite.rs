use super::*;
use crate::ir::mlil::ConstantMaterializationDialect;

impl ConstantMaterializationDialect for ToyDialect {
    fn materialize_constant(
        instruction: &Instruction<Self>,
        constant: &Self::Constant,
    ) -> Option<Self::Operation> {
        matches!(instruction.operation(), Operation::Copy).then_some(Operation::Constant(*constant))
    }
}

#[test]
fn proven_constants_materialize_without_changing_identities() {
    let mut builder = FunctionBuilder::<ToyDialect>::new("toy::constants".into());
    let body = builder.new_block("body");
    let source = builder.declare_variable(0, None).unwrap();
    let result = builder.declare_variable(0, None).unwrap();
    let literal = builder
        .append_instruction(
            body,
            Operation::Constant(42),
            Vec::new(),
            vec![TypedVariable::new(source, Type::Integer)],
            false,
            Some(Span { start: 1, end: 2 }),
        )
        .unwrap();
    let copy = builder
        .append_instruction(
            body,
            Operation::Copy,
            vec![TypedVariable::new(source, Type::Integer)],
            vec![TypedVariable::new(result, Type::Integer)],
            false,
            Some(Span { start: 2, end: 3 }),
        )
        .unwrap();
    let returned = builder
        .append_instruction(
            body,
            Operation::Return,
            vec![TypedVariable::new(result, Type::Integer)],
            Vec::new(),
            false,
            Some(Span { start: 3, end: 4 }),
        )
        .unwrap();
    builder
        .add_edge(builder.entry(), body, Edge::Entry, None)
        .unwrap();
    let function = builder.finish().unwrap();

    let materialized = function.materialize_constants().unwrap();

    assert_eq!(materialized.rewritten, 1);
    assert_eq!(
        materialized
            .function
            .instruction(literal)
            .unwrap()
            .operation(),
        &Operation::Constant(42)
    );
    let rewritten = materialized.function.instruction(copy).unwrap();
    assert_eq!(rewritten.operation(), &Operation::Constant(42));
    assert!(rewritten.uses().is_empty());
    assert_eq!(rewritten.defs(), [result]);
    assert_eq!(
        materialized.function.instruction_point(copy),
        function.instruction_point(copy)
    );
    assert_eq!(
        materialized.function.instruction_point(returned),
        function.instruction_point(returned)
    );
    assert_eq!(
        materialized
            .function
            .provenance()
            .mappings_from(2)
            .collect::<Vec<_>>(),
        function.provenance().mappings_from(2).collect::<Vec<_>>()
    );
}

#[test]
fn a_derived_graph_reindexes_the_instruction_table() {
    let mut builder = FunctionBuilder::<ToyDialect>::new("toy::derived".into());
    let body = builder.new_block("body");
    let value = builder.declare_variable(0, None).unwrap();
    let constant = builder
        .append_instruction(
            body,
            Operation::Constant(1),
            Vec::new(),
            vec![TypedVariable::new(value, Type::Integer)],
            false,
            None,
        )
        .unwrap();
    let ret = builder
        .append_instruction(
            body,
            Operation::Return,
            vec![TypedVariable::new(value, Type::Integer)],
            Vec::new(),
            false,
            None,
        )
        .unwrap();
    builder
        .add_edge(builder.entry(), body, Edge::Entry, None)
        .unwrap();
    let function = builder.finish().unwrap();

    // Dropping the leading instruction moves the one after it. A stale
    // position table would hand `constant`'s identity the return.
    let derived = function.with_derived_cfg(|cfg| {
        cfg.block_mut(body).instructions_mut().remove(0);
    });

    assert_eq!(derived.instruction(constant), None);
    assert_eq!(derived.instruction(ret).map(Instruction::id), Some(ret));
    assert_eq!(
        derived.instruction_point(ret),
        Some(crate::ProgramPoint {
            block: body,
            inst_idx: 0,
        })
    );
    assert_eq!(
        function.instruction(constant).map(Instruction::id),
        Some(constant)
    );
}
