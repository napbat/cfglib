use super::*;
use crate::ir::mlil::ConstantMaterializationDialect;
use crate::ir::mlil::InstructionEdit;
use crate::ir::mlil::InstructionReplacement;

#[test]
fn splice_preserves_order_and_remaps_source_correspondences() {
    let mut builder = FunctionBuilder::<ToyDialect>::new("toy::splice".into());
    let body = builder.new_block("body");
    let value = builder.declare_variable(0, Some(7)).unwrap();
    let copied = builder.declare_variable(0, None).unwrap();
    let original = builder
        .append_instruction(
            body,
            Operation::Copy,
            vec![TypedVariable::new(value, Type::Integer)],
            vec![TypedVariable::new(copied, Type::Integer)],
            false,
            Some(Span { start: 9, end: 10 }),
        )
        .unwrap();
    builder
        .append_instruction(
            body,
            Operation::Return,
            vec![TypedVariable::new(copied, Type::Integer)],
            Vec::new(),
            false,
            None,
        )
        .unwrap();
    builder
        .add_edge(builder.entry(), body, Edge::Entry, None)
        .unwrap();
    let function = builder.finish().unwrap();

    let result = function
        .splice_instructions_with_variables([(1, None)], |instruction, added| {
            (instruction.id() == original).then(|| {
                let temporary = TypedVariable::new(added[0], Type::Integer);
                InstructionEdit::new(
                    vec![InstructionReplacement::new(
                        Operation::Copy,
                        vec![TypedVariable::new(value, Type::Integer)],
                        vec![temporary.clone()],
                        false,
                    )],
                    Some(InstructionReplacement::new(
                        Operation::Copy,
                        vec![temporary.clone()],
                        vec![TypedVariable::new(copied, Type::Integer)],
                        false,
                    )),
                    vec![InstructionReplacement::new(
                        Operation::Copy,
                        vec![TypedVariable::new(copied, Type::Integer)],
                        vec![temporary],
                        false,
                    )],
                )
            })
        })
        .unwrap();
    assert_eq!(result.function.instruction_count(), 4);
    assert_eq!(result.before[original.index()].len(), 1);
    assert_eq!(result.after[original.index()].len(), 1);
    let mapped = result.original_instructions[original.index()];
    assert_eq!(mapped.raw(), original.raw() + 1);
    assert_eq!(
        result.function.instruction(mapped).unwrap().uses(),
        [result.added_variables[0]]
    );
    for id in result.before[original.index()]
        .iter()
        .chain(core::iter::once(&mapped))
        .chain(result.after[original.index()].iter())
    {
        assert!(
            result
                .function
                .provenance()
                .mappings_to(crate::ir::mlil::EntityId::Instruction(*id))
                .any(|entry| entry.source == Span { start: 9, end: 10 })
        );
    }
    assert!(result.function.verify().is_ok());
}

#[test]
fn splice_follows_block_order_after_a_derived_reordering() {
    let mut builder = FunctionBuilder::<ToyDialect>::new("toy::reordered".into());
    let body = builder.new_block("body");
    let input = builder.declare_variable(0, None).unwrap();
    let first_result = builder.declare_variable(0, None).unwrap();
    let second_result = builder.declare_variable(0, None).unwrap();
    for result in [first_result, second_result] {
        builder
            .append_instruction(
                body,
                Operation::Copy,
                vec![TypedVariable::new(input, Type::Integer)],
                vec![TypedVariable::new(result, Type::Integer)],
                false,
                None,
            )
            .unwrap();
    }
    builder
        .append_instruction(
            body,
            Operation::Return,
            vec![TypedVariable::new(second_result, Type::Integer)],
            Vec::new(),
            false,
            None,
        )
        .unwrap();
    builder
        .add_edge(builder.entry(), body, Edge::Entry, None)
        .unwrap();
    let function = builder.finish().unwrap().with_derived_cfg(|cfg| {
        cfg.block_mut(body).instructions_mut().swap(0, 1);
    });

    let result = function
        .splice_instructions_with_variables(core::iter::empty(), |_, _| None)
        .unwrap();
    let ordered: Vec<_> = result
        .function
        .cfg()
        .block(body)
        .instructions()
        .iter()
        .map(Instruction::id)
        .collect();
    assert_eq!(
        ordered,
        [
            result.original_instructions[1],
            result.original_instructions[0],
            result.original_instructions[2],
        ]
    );
    assert!(result.function.verify().is_ok());
}

impl ConstantMaterializationDialect for ToyDialect {
    fn materialize_constant(
        instruction: &Instruction<Self>,
        constant: &Self::Constant,
    ) -> Option<Self::Operation> {
        matches!(instruction.operation(), Operation::Copy).then_some(Operation::Constant(*constant))
    }
}

#[test]
fn added_abi_variable_keeps_existing_identities_and_provenance() {
    let mut builder = FunctionBuilder::<ToyDialect>::new("toy::abi".into());
    let body = builder.new_block("body");
    let target = builder.declare_variable(0, Some(7)).unwrap();
    let call = builder
        .append_instruction(
            body,
            Operation::Return,
            vec![TypedVariable::new(target, Type::Integer)],
            Vec::new(),
            false,
            Some(Span { start: 9, end: 10 }),
        )
        .unwrap();
    builder
        .add_edge(builder.entry(), body, Edge::Entry, None)
        .unwrap();
    let function = builder.finish().unwrap();

    let rewritten = function
        .rewrite_instructions_with_variables([(1, Some(8))], |instruction, added| {
            (instruction.id() == call).then(|| {
                InstructionReplacement::new(
                    Operation::Return,
                    vec![
                        TypedVariable::new(target, Type::Integer),
                        TypedVariable::new(added[0], Type::Integer),
                    ],
                    Vec::new(),
                    false,
                )
            })
        })
        .unwrap();
    assert_eq!(rewritten.rewritten, 1);
    assert_eq!(rewritten.added_variables.len(), 1);
    assert_eq!(
        rewritten
            .function
            .variable(target)
            .map(|variable| variable.id),
        Some(target)
    );
    assert_eq!(
        rewritten.function.instruction(call).unwrap().uses(),
        [target, rewritten.added_variables[0]]
    );
    assert_eq!(
        rewritten.function.instruction_point(call),
        function.instruction_point(call)
    );
    assert_eq!(
        rewritten
            .function
            .provenance()
            .mappings_from(9)
            .collect::<Vec<_>>(),
        function.provenance().mappings_from(9).collect::<Vec<_>>()
    );
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

/// The call defines a register nothing inside the function reads, and the
/// caller reads it back: the seed is what says so.
#[test]
fn an_exit_seed_keeps_the_definition_it_names() {
    let (function, [returned, read, _, _]) = clobbering_function();
    let exit = function
        .cfg()
        .block_ids()
        .find(|&block| function.cfg().outgoing(block).next().is_none())
        .expect("the fixture has one exit");
    let call = |instruction: &Instruction<ToyDialect>| {
        matches!(instruction.operation(), Operation::Store(_))
    };

    let (seeded, count) = function
        .drop_unread_definitions_with_exits(call, |block| {
            if block == exit {
                vec![returned]
            } else {
                Vec::new()
            }
        })
        .unwrap();
    assert_eq!(count, 1, "only the definition nothing observes goes");
    let report = seeded.verify();
    assert!(report.is_ok(), "{:?}", report.issues);
    assert_eq!(
        seeded
            .instructions()
            .find(|instruction| call(instruction))
            .expect("the call stays")
            .defs(),
        &[returned, read],
        "the seeded definition counts as read"
    );

    let (unseeded, count) = function.drop_unread_definitions(call).unwrap();
    assert_eq!(count, 2, "without the seed the same definition goes");
    assert_eq!(
        unseeded
            .instructions()
            .find(|instruction| call(instruction))
            .expect("the call stays")
            .defs(),
        &[read]
    );
}
