//! Canonical rebuilds: elimination, copy propagation, and variable
//! pruning all answer with a function that verifies.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use crate::ir::mlil::{EntityId, Function, FunctionBuilder, Signature, TypedVariable, VariableId};
use crate::test_util::toy::Span;
use crate::{BlockId, ProgramPoint};

use super::{Edge, Operation, ToyDialect, Type};

fn integer(variable: VariableId) -> TypedVariable<ToyDialect> {
    TypedVariable::new(variable, Type::Integer)
}

/// The block every fixture ends in, which is also its only exit.
fn exit_block(function: &Function<ToyDialect>) -> BlockId {
    function
        .cfg()
        .block_ids()
        .find(|&block| function.cfg().outgoing(block).next().is_none())
        .expect("every fixture has one exit block")
}

/// A function holding one unread definition, one definition an effectful
/// instruction reads, and one effectful instruction whose own definition
/// nothing reads.
fn dead_code_function() -> (Function<ToyDialect>, VariableId) {
    let mut builder = FunctionBuilder::<ToyDialect>::new("toy::dead".into());
    let body = builder.new_block("body");
    let unread = builder.declare_variable(0, None).unwrap();
    let read = builder.declare_variable(0, None).unwrap();
    let stored = builder.declare_variable(0, None).unwrap();
    builder
        .append_instruction(
            body,
            Operation::Constant(1),
            Vec::new(),
            vec![integer(unread)],
            false,
            Some(Span { start: 1, end: 2 }),
        )
        .unwrap();
    builder
        .append_instruction(
            body,
            Operation::Constant(2),
            Vec::new(),
            vec![integer(read)],
            false,
            Some(Span { start: 2, end: 3 }),
        )
        .unwrap();
    // A store declares an effect, so it survives although nothing reads
    // what it defines. A throwing instruction survives for the same
    // reason: verification refuses one that declares no effect.
    builder
        .append_instruction(
            body,
            Operation::Store(0),
            vec![integer(read)],
            vec![integer(stored)],
            false,
            Some(Span { start: 3, end: 4 }),
        )
        .unwrap();
    builder
        .append_instruction(body, Operation::Return, Vec::new(), Vec::new(), false, None)
        .unwrap();
    builder
        .add_edge(builder.entry(), body, Edge::Entry, None)
        .unwrap();
    (builder.finish().unwrap(), unread)
}

#[test]
fn elimination_drops_the_unread_definition_and_its_provenance() {
    let (function, _) = dead_code_function();
    let (eliminated, removed) = function.eliminate_dead_code(|_| Vec::new()).unwrap();

    assert_eq!(removed, 1, "only the unread constant goes");
    let report = eliminated.verify();
    assert!(report.is_ok(), "{:?}", report.issues);
    assert_eq!(eliminated.instruction_count(), 3, "identities are dense");
    assert_eq!(
        eliminated.instructions().count(),
        3,
        "every identity names a stored instruction"
    );
    assert_eq!(
        eliminated.provenance().mappings_from(1).count(),
        0,
        "the dropped instruction takes its provenance with it"
    );
    assert_eq!(
        eliminated.provenance().mappings_from(3).count(),
        1,
        "a survivor keeps its own"
    );
    assert_eq!(
        eliminated.cfg().block_count(),
        function.cfg().block_count(),
        "blocks survive"
    );
    assert_eq!(
        eliminated.variables().len(),
        function.variables().len(),
        "elimination is not the variable axis"
    );
}

#[test]
fn an_exit_seed_keeps_the_definition_it_names() {
    let (function, unread) = dead_code_function();
    let exit = exit_block(&function);
    let (eliminated, removed) = function
        .eliminate_dead_code(|block| {
            if block == exit {
                vec![unread]
            } else {
                Vec::new()
            }
        })
        .unwrap();

    assert_eq!(removed, 0, "the seed observes the only dead definition");
    assert_eq!(
        eliminated, function,
        "a function with nothing to remove comes back untouched"
    );
}

#[test]
fn elimination_keeps_an_effectful_instruction_nothing_reads() {
    let (function, _) = dead_code_function();
    let (eliminated, _) = function.eliminate_dead_code(|_| Vec::new()).unwrap();
    assert!(
        eliminated
            .instructions()
            .any(|instruction| *instruction.operation() == Operation::Store(0)),
        "a declared effect outranks an unread definition"
    );
    assert!(
        crate::DeadCode::compute(eliminated.cfg())
            .instructions
            .iter()
            .all(
                |point: &ProgramPoint| eliminated.cfg().block(point.block).instructions()
                    [point.inst_idx]
                    .effects()
                    .is_empty()
            ),
        "nothing effect-free is left dead"
    );
}

/// A function whose only reader of a constant is a copy, and whose only
/// reader of the copy is an effectful instruction.
fn copy_function() -> Function<ToyDialect> {
    let mut builder = FunctionBuilder::<ToyDialect>::new("toy::copies".into());
    let body = builder.new_block("body");
    let source = builder.declare_variable(0, None).unwrap();
    let alias = builder.declare_variable(0, None).unwrap();
    builder
        .append_instruction(
            body,
            Operation::Constant(3),
            Vec::new(),
            vec![integer(source)],
            false,
            Some(Span { start: 1, end: 2 }),
        )
        .unwrap();
    builder
        .append_instruction(
            body,
            Operation::Copy,
            vec![integer(source)],
            vec![integer(alias)],
            false,
            Some(Span { start: 2, end: 3 }),
        )
        .unwrap();
    builder
        .append_instruction(
            body,
            Operation::Store(0),
            vec![integer(alias)],
            Vec::new(),
            false,
            Some(Span { start: 3, end: 4 }),
        )
        .unwrap();
    builder
        .append_instruction(body, Operation::Return, Vec::new(), Vec::new(), false, None)
        .unwrap();
    builder
        .add_edge(builder.entry(), body, Edge::Entry, None)
        .unwrap();
    builder.finish().unwrap()
}

#[test]
fn copy_propagation_rebuilds_a_function_that_verifies() {
    let function = copy_function();
    let (propagated, statistics) = function.propagate_copies().unwrap();

    assert_eq!(statistics.copies_removed, 1);
    assert_eq!(statistics.uses_rewritten, 1);
    let report = propagated.verify();
    assert!(report.is_ok(), "{:?}", report.issues);
    assert_eq!(propagated.instruction_count(), 3, "identities are dense");
    assert_eq!(
        propagated.provenance().mappings_from(2).count(),
        0,
        "the removed copy takes its provenance with it"
    );
    let store = propagated
        .instructions()
        .find(|instruction| *instruction.operation() == Operation::Store(0))
        .expect("the store survives");
    assert_eq!(
        store.uses(),
        &[VariableId::from_raw(0)],
        "the reader now reads the copy's source"
    );
}

#[test]
fn copy_propagation_leaves_a_copy_free_function_untouched() {
    let (function, _) = dead_code_function();
    let (propagated, statistics) = function.propagate_copies().unwrap();
    assert_eq!(statistics.copies_removed, 0);
    assert_eq!(statistics.uses_rewritten, 0);
    assert_eq!(propagated, function);
}

/// `mov rax, rcx; add rax, rdx; ret`: the function's result is a copy of
/// one of its inputs, and the only reader inside the function is the
/// instruction that consumes it.
fn returned_copy_function() -> (Function<ToyDialect>, VariableId, VariableId) {
    let mut builder = FunctionBuilder::<ToyDialect>::new("toy::returned-copy".into());
    let body = builder.new_block("body");
    let argument = builder.declare_variable(0, None).unwrap();
    let result = builder.declare_variable(0, None).unwrap();
    builder
        .append_instruction(
            body,
            Operation::Copy,
            vec![integer(argument)],
            vec![integer(result)],
            false,
            None,
        )
        .unwrap();
    builder
        .append_instruction(
            body,
            Operation::Store(0),
            vec![integer(result)],
            Vec::new(),
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
    builder
        .set_signature(Signature::<ToyDialect>::new(
            vec![argument],
            vec![Type::Integer],
        ))
        .unwrap();
    (builder.finish().unwrap(), argument, result)
}

#[test]
fn an_unseeded_propagation_drops_the_definition_a_caller_reads_back() {
    let (function, _, _) = returned_copy_function();
    let (propagated, statistics) = function.propagate_copies().unwrap();

    assert_eq!(statistics.copies_removed, 1);
    assert!(
        propagated
            .instructions()
            .all(|instruction| *instruction.operation() != Operation::Copy),
        "nothing defines the result any more"
    );
}

#[test]
fn an_exit_seed_keeps_the_copy_that_defines_the_result() {
    let (function, argument, result) = returned_copy_function();
    let exit = exit_block(&function);
    let (propagated, statistics) = function
        .propagate_copies_with_exits(|block| {
            if block == exit {
                vec![result]
            } else {
                Vec::new()
            }
        })
        .unwrap();

    assert_eq!(
        statistics.uses_rewritten, 1,
        "the reader inside still reads through to the source"
    );
    assert_eq!(
        statistics.copies_removed, 0,
        "the definition the caller reads back stays"
    );
    let report = propagated.verify();
    assert!(report.is_ok(), "{:?}", report.issues);
    let copy = propagated
        .instructions()
        .find(|instruction| *instruction.operation() == Operation::Copy)
        .expect("the copy survives");
    assert_eq!(copy.defs(), &[result]);
    let store = propagated
        .instructions()
        .find(|instruction| *instruction.operation() == Operation::Store(0))
        .expect("the store survives");
    assert_eq!(store.uses(), &[argument]);
}

/// A function with one parameter, one occurring variable, one variable
/// only the provenance names, and one nothing names at all.
fn pruning_function() -> (Function<ToyDialect>, [VariableId; 4]) {
    let mut builder = FunctionBuilder::<ToyDialect>::new("toy::prune".into());
    let body = builder.new_block("body");
    let parameter = builder.declare_variable(0, None).unwrap();
    let spare = builder.declare_variable(0, Some(9)).unwrap();
    let occurring = builder.declare_variable(0, None).unwrap();
    let documented = builder.declare_variable(0, None).unwrap();
    builder
        .append_instruction(
            body,
            Operation::Copy,
            vec![integer(parameter)],
            vec![integer(occurring)],
            false,
            Some(Span { start: 1, end: 2 }),
        )
        .unwrap();
    builder
        .append_instruction(
            body,
            Operation::Store(0),
            vec![integer(occurring)],
            Vec::new(),
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
    builder
        .set_signature(Signature::<ToyDialect>::new(
            vec![parameter],
            vec![Type::Integer],
        ))
        .unwrap();
    builder
        .map_entity(Span { start: 7, end: 8 }, EntityId::Variable(documented))
        .unwrap();
    (
        builder.finish().unwrap(),
        [parameter, spare, occurring, documented],
    )
}

#[test]
fn pruning_drops_only_the_variable_nothing_names() {
    let (function, [parameter, spare, occurring, documented]) = pruning_function();
    let (pruned, pruning) = function.prune_variables().unwrap();

    let report = pruned.verify();
    assert!(report.is_ok(), "{:?}", report.issues);
    assert_eq!(pruned.variables().len(), 3, "the spare declaration goes");
    assert_eq!(pruning.origins, vec![parameter, occurring, documented]);
    assert!(!pruning.kept.contains_key(&spare));
    for (index, &origin) in pruning.origins.iter().enumerate() {
        let renumbered = VariableId::from_raw(u32::try_from(index).unwrap());
        assert_eq!(
            pruning.kept.get(&origin),
            Some(&renumbered),
            "the two directions agree about {origin}"
        );
    }
}

#[test]
fn pruning_renumbers_every_occurrence_and_keeps_graph_identities() {
    let (function, [parameter, _, occurring, documented]) = pruning_function();
    let (pruned, pruning) = function.prune_variables().unwrap();

    assert_eq!(
        pruned.signature().parameters,
        vec![pruning.kept[&parameter]]
    );
    assert_eq!(
        pruned.instruction_count(),
        function.instruction_count(),
        "instruction identities are untouched"
    );
    let copy = pruned
        .instructions()
        .find(|instruction| *instruction.operation() == Operation::Copy)
        .expect("the copy survives");
    assert_eq!(copy.uses(), &[pruning.kept[&parameter]]);
    assert_eq!(copy.defs(), &[pruning.kept[&occurring]]);
    assert_eq!(
        pruned
            .provenance()
            .mappings_from(7)
            .map(|entry| entry.entity)
            .collect::<Vec<_>>(),
        vec![EntityId::Variable(pruning.kept[&documented])],
        "a provenance entry follows its variable"
    );
    // The native provenance of a survivor is the one it was declared
    // with; only the spare carried a native slot, and it is gone.
    assert!(
        pruned
            .variables()
            .iter()
            .all(|variable| variable.native.is_none())
    );
}

#[test]
fn pruning_a_fully_named_function_returns_it_unchanged() {
    let function = copy_function();
    let (pruned, pruning) = function.prune_variables().unwrap();
    assert_eq!(pruned, function);
    assert_eq!(
        pruning.origins,
        function
            .variables()
            .iter()
            .map(|variable| variable.id)
            .collect::<Vec<_>>()
    );
    assert!(
        pruning.kept.iter().all(|(old, new)| old == new,),
        "an identity map maps every variable to itself"
    );
}

/// The three rebuilds compose: eliminate, propagate, then prune, and the
/// result still verifies.
#[test]
fn the_canonical_rebuilds_compose() {
    let function = copy_function();
    let (eliminated, _) = function.eliminate_dead_code(|_| Vec::new()).unwrap();
    let (propagated, _) = eliminated.propagate_copies().unwrap();
    let (pruned, pruning) = propagated.prune_variables().unwrap();

    let report = pruned.verify();
    assert!(report.is_ok(), "{:?}", report.issues);
    assert_eq!(
        pruned.variables().len(),
        1,
        "the propagated copy's target stops occurring: {:?}",
        pruning.origins
    );
    assert!(pruned.ssa().is_ok(), "a canonical function still lifts");
}
