//! The closed loop of the three algorithms over the toy dialects: lift a
//! `FlatFunction` into HLIL, recover its structure, lower it into a flat
//! function again, and lift that.
//!
//! The toys disagree on the kind of a synthesized jump and of an exceptional
//! edge, so every loop here closes under both conventions.

extern crate alloc;

use alloc::string::String;

use crate::BlockId;
use crate::dataflow::ProgramPoint;
use crate::graph::view::DenseId;
use crate::ir::hlil::{
    Entity, FlatBuilder, FunctionBuilder, FunctionView, StmtShape, has_recovery_candidates, lower,
    pseudocode, recover_structure,
};
use crate::test_util::golden::assert_golden;

use super::fixtures::structured_counting_loop;
use super::flat::{FlatKit, ToyEdge, ToyFlat, ToyFlatBuilder};
use super::flat_fixtures::{counting_loop, dispatch_with_default, guarded_call};
use super::flat_lift::{boolean, int, lifted};
use super::{Prim, ToyHlilBuilder, Ty};

over_toys!(
    lowering_and_relifting_round_trips_a_while_loop,
    lowering_and_relifting_round_trips_a_switch,
    lowering_registers_declared_exception_regions_and_relifts_them,
    a_lifted_function_survives_lowering_and_lifting_again,
    recovery_composes_with_lifting_and_the_recovered_loop_lowers,
    an_endless_loop_without_a_body_round_trips,
);

fn lowering_and_relifting_round_trips_a_while_loop<D: FlatKit>() {
    let (function, _) = structured_counting_loop::<D>();
    let lowered = lower::<D, _, _>(&function, ToyFlatBuilder::<D>::new()).expect("the loop lowers");
    assert!(lowered.function.check().is_ok());
    assert_ne!(lowered.origins, []);

    let relifted = lifted(&lowered.function);
    assert!(
        relifted.report.is_fully_structured(),
        "{:?}",
        relifted.report
    );
    assert_golden(
        "hlil/relift-while-loop.pseudo",
        &pseudocode(&relifted.function),
    );
}

/// `loop { }`: the structure that lifting gives a block that jumps to itself,
/// and a function that iterates forever, which lowering accepts.
fn an_endless_loop_without_a_body_round_trips<D: FlatKit>() {
    let mut builder = ToyHlilBuilder::<D>::new();
    let spin = builder.loop_(&[]);
    let function = builder.finish(&[spin]).expect("an empty loop verifies");

    let lowered = lower::<D, _, _>(&function, ToyFlatBuilder::<D>::new())
        .expect("a function may iterate forever without doing anything");
    assert!(lowered.function.check().is_ok());
    let relifted = lifted(&lowered.function);
    assert!(
        relifted.report.is_fully_structured(),
        "{:?}",
        relifted.report
    );
    assert_eq!(pseudocode(&relifted.function), "loop {\n}\n");
}

fn lowering_and_relifting_round_trips_a_switch<D: FlatKit>() {
    let mut builder = ToyHlilBuilder::<D>::new();
    let selector = builder.int_variable(0);
    let result = builder.int_variable(0);
    let first = {
        let value = builder.number(10);
        builder.store(result, value)
    };
    let second = {
        let value = builder.number(20);
        builder.store(result, value)
    };
    let fallback = {
        let value = builder.number(30);
        builder.store(result, value)
    };
    let scrutinee = builder.read(selector);
    let switch = builder.switch(
        scrutinee,
        &[
            crate::ir::hlil::SwitchArm {
                values: &[D::constant(1), D::constant(2)],
                body: &[first],
            },
            crate::ir::hlil::SwitchArm {
                values: &[D::constant(3)],
                body: &[second],
            },
        ],
        &[fallback],
    );
    let read_result = builder.read(result);
    let ret = builder.return_(&[read_result]);
    let function = builder.finish(&[switch, ret]).expect("the switch verifies");

    let lowered =
        lower::<D, _, _>(&function, ToyFlatBuilder::<D>::new()).expect("the switch lowers");
    let relifted = lifted(&lowered.function);
    assert!(
        relifted.report.is_fully_structured(),
        "{:?}",
        relifted.report
    );
    assert_golden("hlil/relift-switch.pseudo", &pseudocode(&relifted.function));
}

fn lowering_registers_declared_exception_regions_and_relifts_them<D: FlatKit>() {
    let mut builder = ToyHlilBuilder::<D>::new();
    let x = builder.int_variable(0);
    let call = builder.call(Prim::Call, &[], Ty::Integer);
    let target = builder.read(x);
    let protected = builder.assign(target, call);
    let seven = builder.number(7);
    let handler_return = builder.return_(&[seven]);
    let try_statement = builder.try_(
        &[protected],
        &[crate::ir::hlil::Handler {
            kind: crate::ir::hlil::HandlerKind::CatchAll,
            binding: None,
            caught_types: &[],
            body: &[handler_return],
        }],
        &[],
    );
    let read_x = builder.read(x);
    let ret = builder.return_(&[read_x]);
    let function = builder
        .finish(&[try_statement, ret])
        .expect("the guarded call verifies");

    let lowered =
        lower::<D, _, _>(&function, ToyFlatBuilder::<D>::new()).expect("the guarded call lowers");
    assert_eq!(lowered.function.cfg.regions().len(), 1);
    let region = &lowered.function.cfg.regions()[0];
    assert_eq!(region.handlers.len(), 1);
    assert!(region.handlers[0].body.is_known());

    let relifted = lifted(&lowered.function);
    assert_golden(
        "hlil/relift-try-catch.pseudo",
        &pseudocode(&relifted.function),
    );
}

/// The pseudocode of `source` lifted, and of the function that lowering and
/// lifting again give back.
fn lifted_and_relifted<D: FlatKit>(source: &ToyFlat<D>) -> (String, String) {
    let first = lifted(source);
    let lowered =
        lower::<D, _, _>(&first.function, ToyFlatBuilder::<D>::new()).expect("the function lowers");
    assert!(lowered.function.check().is_ok());
    let again = lifted(&lowered.function);
    assert!(again.report.is_fully_structured(), "{:?}", again.report);
    (pseudocode(&first.function), pseudocode(&again.function))
}

fn a_lifted_function_survives_lowering_and_lifting_again<D: FlatKit>() {
    for source in [
        counting_loop::<D>(),
        dispatch_with_default::<D>(),
        guarded_call::<D>(),
    ] {
        let (first, again) = lifted_and_relifted(&source);
        assert_eq!(first, again, "{}", source.dump());
    }
}

/// `v1 = 0; while (lt(v1, v0)) { v2 = add(v2, v1); v1 = add(v1, 1); } return v2;`
/// in flat form, with the bound and the total as parameters.
fn counted_sum<D: FlatKit>() -> ToyFlat<D> {
    let mut flat = ToyFlatBuilder::<D>::new();
    let init = flat.new_block(Some("init"));
    let header = flat.new_block(Some("header"));
    let body = flat.new_block(Some("body"));
    let exit = flat.new_block(Some("exit"));
    let bound = flat.int_variable(0);
    let index = flat.int_variable(1);
    let total = flat.int_variable(2);
    let condition = flat.int_variable(3);
    let one = flat.int_variable(3);

    flat.literal(init, 0, index);
    flat.operate(
        header,
        Prim::LessThan,
        &[int::<D>(index), int::<D>(bound)],
        &[boolean::<D>(condition)],
    );
    flat.branch(header, condition);
    flat.operate(
        body,
        Prim::Add,
        &[int::<D>(total), int::<D>(index)],
        &[int::<D>(total)],
    );
    flat.literal(body, 1, one);
    flat.operate(
        body,
        Prim::Add,
        &[int::<D>(index), int::<D>(one)],
        &[int::<D>(index)],
    );
    flat.give_back(exit, &[total]);
    flat.start(init);
    flat.connect(init, header, ToyEdge::Fall);
    flat.connect(header, body, ToyEdge::True);
    flat.connect(header, exit, ToyEdge::False);
    flat.connect(body, header, ToyEdge::Fall);
    flat.takes(&[bound, total]);
    flat.finish().expect("the counted sum is well formed")
}

fn recovery_composes_with_lifting_and_the_recovered_loop_lowers<D: FlatKit>() {
    let source = counted_sum::<D>();
    let first = lifted(&source);
    assert_eq!(
        pseudocode(&first.function),
        "v1 = 0;\nwhile (lt(v1, v0)) {\n    v2 = add(v2, v1);\n    v1 = add(v1, 1);\n}\nreturn v2;\n"
    );

    // The caller composes the algorithms: lift, then recover when the lifted
    // function has a candidate, then move the correspondence along.
    assert!(has_recovery_candidates(&first.function));
    let recovery = recover_structure::<D, _, _>(&first.function, ToyHlilBuilder::<D>::new())
        .expect("the function recovers");
    assert_eq!(recovery.for_loops, 1);
    // The lift reproduces the shape that the recovery golden was written from.
    assert_golden(
        "hlil/recover-for-loop.pseudo",
        &pseudocode(&recovery.function),
    );

    let correspondence = first.correspondence.clone().remap(&recovery);
    let recovered = &recovery.function;
    let row = |block: usize, inst_idx: usize| {
        let point = ProgramPoint {
            block: BlockId::from_index(block),
            inst_idx,
        };
        correspondence
            .instructions
            .iter()
            .find(|(candidate, _)| *candidate == point)
            .map(|(_, entity)| *entity)
    };
    // The branch of the loop header now stands for the `for` that consumed
    // the `while`, and the initialization is its initializer.
    let Some(Entity::Statement(counted)) = row(2, 1) else {
        panic!("{correspondence:?}");
    };
    let StmtShape::For { initializer, .. } = recovered.stmt(counted) else {
        panic!("the header branch is the for statement");
    };
    assert_eq!(
        row(1, 0),
        initializer.first().map(|&init| Entity::Statement(init))
    );
    // Every entity of every row exists in the recovered function.
    for (_, entity) in &correspondence.instructions {
        match *entity {
            Entity::Statement(statement) => assert!(statement.index() < recovered.stmt_count()),
            Entity::Expression(expression) => assert!(expression.index() < recovered.expr_count()),
            Entity::Variable(variable) => assert!(variable.index() < recovered.variable_count()),
        }
    }
    assert_eq!(correspondence.variables.len(), source.variables.len());

    // The recovered loop lowers, and the lowered function lifts back to the
    // loop that the first lift produced: the `for` is a source-level
    // construct that the flat form does not keep.
    let lowered =
        lower::<D, _, _>(recovered, ToyFlatBuilder::<D>::new()).expect("the recovered loop lowers");
    assert_eq!(
        pseudocode(&lifted(&lowered.function).function),
        pseudocode(&first.function)
    );
}
