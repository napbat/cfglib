//! Pure-transfer trampoline tests, split out to respect the source-size
//! policy.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use super::{Edge, MediumOperation, Toy, Type};
use crate::ir::hlil::lift_function;
use crate::ir::mlil;
use crate::test_util::golden::assert_golden;

/// A pre-tested loop whose conditional break routes through a block
/// holding an explicit jump instruction, with that block claimed by an
/// exception region's protected set — javac shapes both ways.
fn claimed_trampoline_loop() -> mlil::Function<Toy> {
    let mut builder = mlil::FunctionBuilder::<Toy>::new("toy::trampoline".into());
    let header = builder.new_block("header");
    let body = builder.new_block("body");
    let latch = builder.new_block("latch");
    let tramp = builder.new_block("tramp");
    let exit = builder.new_block("exit");
    let pad = builder.new_block("pad");
    let i = builder.declare_variable(0, None).unwrap();
    let n = builder.declare_variable(0, None).unwrap();
    let t_loop = builder.declare_variable(1, None).unwrap();
    let t_break = builder.declare_variable(1, None).unwrap();
    let t_one = builder.declare_variable(1, None).unwrap();
    let int = |variable| mlil::TypedVariable::<Toy>::new(variable, Type::Integer);
    let cond = |variable| mlil::TypedVariable::<Toy>::new(variable, Type::Boolean);
    let mut push = |block, operation, uses: Vec<_>, defs: Vec<_>| {
        builder
            .append_instruction(block, operation, uses, defs, false, None)
            .unwrap();
    };
    push(
        header,
        MediumOperation::LessThan,
        vec![int(i), int(n)],
        vec![cond(t_loop)],
    );
    push(
        header,
        MediumOperation::Branch,
        vec![cond(t_loop)],
        Vec::new(),
    );
    push(
        body,
        MediumOperation::LessThan,
        vec![int(n), int(i)],
        vec![cond(t_break)],
    );
    push(
        body,
        MediumOperation::Branch,
        vec![cond(t_break)],
        Vec::new(),
    );
    push(
        latch,
        MediumOperation::Constant(1),
        Vec::new(),
        vec![int(t_one)],
    );
    push(
        latch,
        MediumOperation::Add,
        vec![int(i), int(t_one)],
        vec![int(i)],
    );
    push(tramp, MediumOperation::Jump, Vec::new(), Vec::new());
    push(exit, MediumOperation::Return, vec![int(i)], Vec::new());
    push(pad, MediumOperation::Return, vec![int(i)], Vec::new());
    builder
        .add_edge(builder.entry(), header, Edge::Entry, None)
        .unwrap();
    builder.add_edge(header, body, Edge::True, None).unwrap();
    builder.add_edge(header, exit, Edge::False, None).unwrap();
    builder.add_edge(body, tramp, Edge::True, None).unwrap();
    builder.add_edge(body, latch, Edge::False, None).unwrap();
    builder.add_edge(latch, header, Edge::Fall, None).unwrap();
    builder.add_edge(tramp, exit, Edge::Jump, None).unwrap();
    builder
        .add_region(crate::Region {
            id: crate::RegionId::from_raw(0),
            protected_blocks: [tramp].into_iter().collect(),
            handlers: vec![crate::Handler {
                entry: pad,
                body: crate::HandlerBody::Unknown,
                kind: crate::HandlerKind::CatchAll,
            }],
            parent: None,
        })
        .unwrap();
    builder
        .set_signature(mlil::Signature::<Toy>::new(vec![i, n], vec![Type::Integer]))
        .unwrap();
    builder.finish().unwrap()
}

/// The region claim refuses out-of-bound inlining and a non-empty block
/// refuses the trampoline hop, so structuring depends on the lift
/// emptying dialect-declared pure transfers in its working view.
#[test]
fn a_region_claimed_jump_trampoline_resolves_as_a_break() {
    let lifted = lift_function(&claimed_trampoline_loop()).unwrap();
    assert!(lifted.report.gotos.is_empty(), "{:?}", lifted.report);
    // The trampoline resolved to a break; no goto survives.
    assert_golden(
        "hlil/trampoline-break.pseudo",
        &lifted.function.to_pseudocode(),
    );
}
