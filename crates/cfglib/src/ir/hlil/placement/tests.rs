//! Declaration placement over the toy dialects.

use crate::ir::hlil::VariablePlacements;
use crate::ir::hlil::tests::{Kit, ToyHlilBuilder, over_toys};

over_toys!(
    shared_branch_variable_anchors_before_if,
    branch_local_variable_stays_in_the_branch_body,
    loop_local_variable_stays_in_the_loop_body,
    for_clause_variable_anchors_before_the_loop,
    switch_arm_variables_anchor_in_their_arm,
    unreferenced_variables_have_no_anchor,
);

fn shared_branch_variable_anchors_before_if<D: Kit>() {
    let mut builder = ToyHlilBuilder::<D>::new();
    let value = builder.int_variable(0);
    let then_use = builder.use_of(value);
    let else_use = builder.use_of(value);
    let condition = builder.condition();
    let branch = builder.if_(condition, &[then_use], &[else_use]);
    let function = builder.finish(&[branch]).unwrap();

    let placements = VariablePlacements::compute(&function);

    assert_eq!(placements.placement(value), Some(branch));
    assert_eq!(placements.before(branch), [value]);
}

fn branch_local_variable_stays_in_the_branch_body<D: Kit>() {
    let mut builder = ToyHlilBuilder::<D>::new();
    let value = builder.int_variable(0);
    let unrelated = builder.noop();
    let first_use = builder.use_of(value);
    let second_use = builder.use_of(value);
    let condition = builder.condition();
    let branch = builder.if_(condition, &[unrelated, first_use, second_use], &[]);
    let function = builder.finish(&[branch]).unwrap();

    let placements = VariablePlacements::compute(&function);

    assert_eq!(placements.placement(value), Some(first_use));
    assert_eq!(placements.before(branch), []);
    assert_eq!(placements.before(first_use), [value]);
}

fn loop_local_variable_stays_in_the_loop_body<D: Kit>() {
    let mut builder = ToyHlilBuilder::<D>::new();
    let value = builder.int_variable(0);
    let first_use = builder.use_of(value);
    let second_use = builder.use_of(value);
    let condition = builder.condition();
    let looped = builder.while_(condition, &[first_use, second_use]);
    let function = builder.finish(&[looped]).unwrap();

    let placements = VariablePlacements::compute(&function);

    assert_eq!(placements.placement(value), Some(first_use));
    assert_eq!(placements.before(looped), []);
}

fn for_clause_variable_anchors_before_the_loop<D: Kit>() {
    let mut builder = ToyHlilBuilder::<D>::new();
    let value = builder.int_variable(0);
    let initializer = builder.use_of(value);
    let condition = builder.condition();
    let looped = builder.for_(&[initializer], Some(condition), &[], &[]);
    let function = builder.finish(&[looped]).unwrap();

    let placements = VariablePlacements::compute(&function);

    assert_eq!(placements.placement(value), Some(looped));
    assert_eq!(placements.before(looped), [value]);
    assert_eq!(placements.before(initializer), []);
}

fn switch_arm_variables_anchor_in_their_arm<D: Kit>() {
    use crate::ir::hlil::SwitchArm;

    let mut builder = ToyHlilBuilder::<D>::new();
    let local = builder.int_variable(0);
    let shared = builder.int_variable(1);
    let first_local = builder.use_of(local);
    let second_local = builder.use_of(local);
    let first_shared = builder.use_of(shared);
    let second_shared = builder.use_of(shared);
    let scrutinee = builder.number(0);
    let values = [D::constant(1)];
    let switch = builder.switch(
        scrutinee,
        &[SwitchArm {
            values: &values,
            body: &[first_local, second_local, first_shared],
        }],
        &[second_shared],
    );
    let function = builder.finish(&[switch]).unwrap();

    let placements = VariablePlacements::compute(&function);

    assert_eq!(placements.placement(local), Some(first_local));
    assert_eq!(placements.placement(shared), Some(switch));
    assert_eq!(placements.before(switch), [shared]);
}

fn unreferenced_variables_have_no_anchor<D: Kit>() {
    let mut builder = ToyHlilBuilder::<D>::new();
    let unused = builder.int_variable(0);
    let statement = builder.noop();
    let function = builder.finish(&[statement]).unwrap();

    let placements = VariablePlacements::compute(&function);

    assert_eq!(placements.placement(unused), None);
    assert_eq!(placements.before(statement), []);
}
