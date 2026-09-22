//! The lowering direction for an effect that declares the places it
//! writes: an MLIL effect instruction with definitions becomes an RTL
//! effect whose `writes` are those variables' planned places.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use super::super::super::{Function, FunctionBuilder, Place, Statement, lift, lower};
use super::{
    Effect, EffectOp, JvmConstraint, JvmRtlEdge, Managed, hierarchy, slot_read, word_const,
};

/// A function whose invoke writes two slots it does not read.
fn invoking_function() -> Function<Managed> {
    let mut builder = FunctionBuilder::<Managed>::new("test".into());
    let entry = builder.entry();
    let body = builder.new_block("body");
    builder.add_edge(entry, body, JvmRtlEdge::Entry).unwrap();
    builder
        .append(body, super::slot_write(4, word_const(7)), None)
        .unwrap();
    builder
        .append(
            body,
            Statement::Effect {
                operation: EffectOp::Invoke,
                operands: vec![slot_read(4, JvmConstraint::Unknown)],
                writes: vec![
                    Place {
                        storage: 5,
                        lanes: vec![0],
                    },
                    Place {
                        storage: 6,
                        lanes: vec![0],
                    },
                ],
                effects: vec![Effect::Call],
                may_throw: false,
            },
            None,
        )
        .unwrap();
    builder
        .append(
            body,
            Statement::Return {
                values: vec![slot_read(5, JvmConstraint::Unknown)],
            },
            None,
        )
        .unwrap();
    builder.finish().unwrap()
}

#[test]
fn lowering_round_trips_the_places_an_effect_writes() {
    let function = invoking_function();
    let mlil_function = lift(&function, &hierarchy())
        .unwrap()
        .builder
        .finish()
        .unwrap();

    let lowered = lower::<Managed>(&mlil_function).unwrap();
    let effect_instruction = mlil_function
        .instructions()
        .find(|instruction| instruction.defs().len() == 2)
        .expect("the invoke defines both written slots");
    let expected: Vec<Place<Managed>> = effect_instruction
        .defs()
        .iter()
        .map(|&variable| {
            lowered
                .placement
                .place(variable)
                .expect("the plan places every definition")
                .clone()
        })
        .collect();

    let statement = lowered.statements(effect_instruction.id())[0];
    let lowered_statement = lowered
        .function
        .cfg()
        .blocks()
        .flat_map(|block| block.instructions().iter())
        .find(|node| node.id() == statement)
        .expect("the lowered statement is stored");
    let Statement::Effect { writes, .. } = lowered_statement.statement() else {
        panic!("an effect lowers to an effect");
    };
    assert_eq!(writes, &expected, "the writes keep the planned places");

    // The relift recovers one web per written place, so the round trip is
    // closed over definitions as well as reads.
    let relifted = lift(&lowered.function, &hierarchy())
        .unwrap()
        .builder
        .finish()
        .unwrap();
    assert!(
        relifted
            .instructions()
            .any(|instruction| instruction.defs().len() == 2),
        "the relifted invoke still defines both places"
    );
}
