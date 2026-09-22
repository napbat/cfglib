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

/// A call under a calling convention states that it writes every register
/// the convention does not preserve. `Store` stands for it here: three
/// definitions, one of which something reads.
fn clobbering_function() -> (crate::ir::mlil::Function<ToyDialect>, [VariableId; 4]) {
    let mut builder = FunctionBuilder::<ToyDialect>::new("toy::clobber".into());
    let body = builder.new_block("body");
    let unread = builder.declare_variable(0, Some(1)).unwrap();
    let read = builder.declare_variable(0, Some(2)).unwrap();
    let also_unread = builder.declare_variable(0, Some(3)).unwrap();
    let loaded = builder.declare_variable(0, None).unwrap();
    let typed = |variable| TypedVariable::new(variable, Type::Integer);
    builder
        .append_instruction(
            body,
            Operation::Store(0),
            Vec::new(),
            vec![typed(unread), typed(read), typed(also_unread)],
            false,
            Some(Span { start: 1, end: 2 }),
        )
        .unwrap();
    builder
        .append_instruction(
            body,
            Operation::Load(1),
            vec![typed(read)],
            vec![typed(loaded)],
            false,
            None,
        )
        .unwrap();
    builder
        .append_instruction(body, Operation::Return, Vec::new(), Vec::new(), false, None)
        .unwrap();
    builder
        .add_edge(builder.entry(), body, Edge::Entry, None)
        .unwrap();
    (
        builder.finish().unwrap(),
        [unread, read, also_unread, loaded],
    )
}

#[test]
fn an_effect_keeps_exactly_the_definitions_something_reads() {
    let (function, [_, read, _, loaded]) = clobbering_function();
    let (dropped, count) = function
        .drop_unread_definitions(|instruction| {
            matches!(instruction.operation(), Operation::Store(_))
        })
        .unwrap();

    assert_eq!(count, 2, "two of the three definitions go");
    let report = dropped.verify();
    assert!(report.is_ok(), "{:?}", report.issues);
    let store = dropped
        .instructions()
        .find(|instruction| matches!(instruction.operation(), Operation::Store(_)))
        .expect("the effect itself stays");
    assert_eq!(store.defs(), &[read]);
    assert_eq!(
        store.def_types().len(),
        1,
        "the types follow the definitions"
    );
    assert!(store.uses().is_empty(), "the operands are untouched");

    // The instruction `of` does not select keeps its own unread
    // definition, and every identity survives.
    let load = dropped
        .instructions()
        .find(|instruction| matches!(instruction.operation(), Operation::Load(_)))
        .expect("the unselected instruction stays");
    assert_eq!(load.defs(), &[loaded]);
    assert_eq!(
        dropped.instruction_count(),
        function.instruction_count(),
        "nothing is removed, so identities are preserved"
    );
    assert_eq!(
        dropped.provenance().mappings_from(1).count(),
        1,
        "the rewritten instruction keeps its provenance"
    );
}

#[test]
fn an_instruction_the_selection_misses_is_untouched() {
    let (function, _) = clobbering_function();
    let (dropped, count) = function.drop_unread_definitions(|_| false).unwrap();
    assert_eq!(count, 0);
    assert_eq!(dropped, function, "selecting nothing changes nothing");
}

#[test]
fn a_selected_instruction_loses_its_own_unread_definition() {
    let (function, _) = clobbering_function();
    let (dropped, count) = function
        .drop_unread_definitions(|instruction| {
            matches!(instruction.operation(), Operation::Load(_))
        })
        .unwrap();
    assert_eq!(count, 1, "the load's own definition is unread");
    assert!(
        dropped
            .instructions()
            .find(|instruction| matches!(instruction.operation(), Operation::Load(_)))
            .expect("the load stays")
            .defs()
            .is_empty()
    );
}
