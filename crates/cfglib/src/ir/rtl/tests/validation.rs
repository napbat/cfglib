//! Construction and completion validation tests.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use super::{
    Edge, Effect, Expr, FunctionBuilder, Place, ScalarType, Statement, TestDialect, ValueShape,
    assign, constant, lift, read,
};

/// A transfer with no assignments is rejected: it would silently drop
/// its effects and exceptional behavior in lowering.
#[test]
fn empty_transfer_is_rejected() {
    let mut builder = FunctionBuilder::<TestDialect>::new("test".into());
    let entry = builder.entry();
    let body = builder.new_block("body");
    builder.add_edge(entry, body, Edge::Entry).unwrap();
    let error = builder.append(
        body,
        Statement::Transfer {
            assignments: Vec::new(),
            effects: vec![Effect::Emit],
            may_throw: true,
        },
        None,
    );
    assert!(error.is_err(), "an empty transfer must not validate");
}

/// One lane written by two places of one transfer is rejected: the
/// parallel semantics leave no defined result.
#[test]
fn duplicate_lane_across_assignments_is_rejected() {
    let mut builder = FunctionBuilder::<TestDialect>::new("test".into());
    let entry = builder.entry();
    let body = builder.new_block("body");
    builder.add_edge(entry, body, Edge::Entry).unwrap();
    let error = builder.append(
        body,
        Statement::Transfer {
            assignments: vec![
                (
                    Place {
                        storage: 0,
                        lanes: vec![0],
                    },
                    constant(1, ScalarType::U32),
                ),
                (
                    Place {
                        storage: 0,
                        lanes: vec![0],
                    },
                    constant(2, ScalarType::U32),
                ),
            ],
            effects: Vec::new(),
            may_throw: false,
        },
        None,
    );
    assert!(error.is_err(), "two writes of one lane must not validate");
}

/// Reinterpretation preserves lane width, not just lane count.
#[test]
fn cross_width_reinterpret_is_rejected() {
    let mut builder = FunctionBuilder::<TestDialect>::new("test".into());
    let entry = builder.entry();
    let body = builder.new_block("body");
    builder.add_edge(entry, body, Edge::Entry).unwrap();
    let error = builder.append(
        body,
        assign(
            1,
            &[0],
            Expr::Reinterpret {
                operand: alloc::boxed::Box::new(constant(1, ScalarType::F32)),
                shape: ValueShape::scalar(ScalarType::F64),
            },
        ),
        None,
    );
    assert!(error.is_err(), "a 32-to-64-bit reinterpretation must fail");
    builder
        .append(
            body,
            assign(
                1,
                &[0],
                Expr::Reinterpret {
                    operand: alloc::boxed::Box::new(constant(1, ScalarType::F32)),
                    shape: ValueShape::scalar(ScalarType::U32),
                },
            ),
            None,
        )
        .expect("a same-width reinterpretation validates");
}

/// A wide scalar lane carries multiple constant words.
#[test]
fn wide_lane_constants_validate_by_word_count() {
    let mut builder = FunctionBuilder::<TestDialect>::new("test".into());
    let entry = builder.entry();
    let body = builder.new_block("body");
    builder.add_edge(entry, body, Edge::Entry).unwrap();
    builder
        .append(
            body,
            assign(
                0,
                &[0],
                Expr::Const {
                    bits: vec![1, 2, 3, 4],
                    shape: ValueShape::scalar(ScalarType::U256),
                },
            ),
            None,
        )
        .expect("a 256-bit lane carries four words");
    let error = builder.append(
        body,
        assign(
            0,
            &[0],
            Expr::Const {
                bits: vec![1, 2],
                shape: ValueShape::scalar(ScalarType::U256),
            },
        ),
        None,
    );
    assert!(error.is_err(), "a short word count must not validate");
}

/// A block ending in a return must not continue anywhere.
#[test]
fn return_block_with_successor_is_rejected() {
    let mut builder = FunctionBuilder::<TestDialect>::new("test".into());
    let entry = builder.entry();
    let body = builder.new_block("body");
    let extra = builder.new_block("extra");
    builder.add_edge(entry, body, Edge::Entry).unwrap();
    builder.add_edge(body, extra, Edge::Fall).unwrap();
    builder
        .append(body, Statement::Return { values: Vec::new() }, None)
        .unwrap();
    builder
        .append(extra, Statement::Return { values: Vec::new() }, None)
        .unwrap();
    assert!(builder.finish().is_err());
}

/// A branch decides between at least two outgoing edges.
#[test]
fn branch_with_single_successor_is_rejected() {
    let mut builder = FunctionBuilder::<TestDialect>::new("test".into());
    let entry = builder.entry();
    let body = builder.new_block("body");
    let exit = builder.new_block("exit");
    builder.add_edge(entry, body, Edge::Entry).unwrap();
    builder.add_edge(body, exit, Edge::True).unwrap();
    builder
        .append(
            body,
            Statement::Branch {
                condition: read(9, &[0], ScalarType::U32),
            },
            None,
        )
        .unwrap();
    builder
        .append(exit, Statement::Return { values: Vec::new() }, None)
        .unwrap();
    assert!(builder.finish().is_err());
}

/// Unreachable blocks are legal and lift faithfully — dead code after a
/// return or throw is source-faithful in managed bytecode.
#[test]
fn unreachable_blocks_are_preserved() {
    let mut builder = FunctionBuilder::<TestDialect>::new("test".into());
    let entry = builder.entry();
    let body = builder.new_block("body");
    let orphan = builder.new_block("orphan");
    builder.add_edge(entry, body, Edge::Entry).unwrap();
    builder
        .append(body, Statement::Return { values: Vec::new() }, None)
        .unwrap();
    let dead = builder
        .append(orphan, assign(0, &[0], constant(1, ScalarType::U32)), None)
        .unwrap();
    builder
        .append(orphan, Statement::Return { values: Vec::new() }, None)
        .unwrap();
    let function = builder.finish().expect("unreachable blocks are legal");

    let lifting = lift(&function, &()).unwrap();
    assert_eq!(
        lifting.maps.instructions(dead).len(),
        1,
        "the unreachable statement still emits"
    );
    let function = lifting.builder.finish().unwrap();
    assert_eq!(function.cfg().block_count(), 3, "the orphan block survives");
}

/// A terminal block without statements retains a control-flow path whose
/// destination is outside the recovered function.
#[test]
fn an_empty_terminal_block_lifts_as_an_unresolved_exit() {
    let mut builder = FunctionBuilder::<TestDialect>::new("test".into());
    let entry = builder.entry();
    let branch = builder.new_block("branch");
    let resolved = builder.new_block("resolved");
    let unresolved = builder.new_block("unresolved");
    builder.add_edge(entry, branch, Edge::Entry).unwrap();
    builder.add_edge(branch, resolved, Edge::False).unwrap();
    builder.add_edge(branch, unresolved, Edge::True).unwrap();
    builder
        .append(
            branch,
            Statement::Branch {
                condition: read(9, &[0], ScalarType::U32),
            },
            None,
        )
        .unwrap();
    builder
        .append(resolved, Statement::Return { values: Vec::new() }, None)
        .unwrap();
    let function = builder.finish().expect("the RTL exit is valid");

    let lifting = lift(&function, &()).expect("the unresolved exit lifts");
    let lifted_unresolved = lifting
        .maps
        .block(unresolved)
        .expect("the empty exit keeps its block mapping");
    let function = lifting.builder.finish().expect("the MLIL exit is valid");

    assert!(function.cfg().block(lifted_unresolved).is_empty());
    assert!(function.cfg().successor_edges(lifted_unresolved).is_empty());
}

#[test]
fn an_empty_forwarding_block_remains_invalid_mlil() {
    let mut builder = FunctionBuilder::<TestDialect>::new("test".into());
    let entry = builder.entry();
    let forwarding = builder.new_block("forwarding");
    let exit = builder.new_block("exit");
    builder.add_edge(entry, forwarding, Edge::Entry).unwrap();
    builder.add_edge(forwarding, exit, Edge::Fall).unwrap();
    builder
        .append(exit, Statement::Return { values: Vec::new() }, None)
        .unwrap();
    let function = builder.finish().expect("the RTL forwarding block is valid");

    let lifting = lift(&function, &()).expect("the RTL structure lifts");
    assert!(
        lifting.builder.finish().is_err(),
        "a nonterminal semantic block must contain an instruction"
    );
}
