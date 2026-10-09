//! The straight-block join: forwarding blocks leave, straight runs become
//! one block, and a throw site keeps its exceptional edges.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use crate::ir::mlil::{EntityId, Function, FunctionBuilder, TypedVariable, VariableId};
use crate::test_util::toy::Span;

use super::{Edge, Operation, ToyDialect, Type};

fn integer(variable: VariableId) -> TypedVariable<ToyDialect> {
    TypedVariable::new(variable, Type::Integer)
}

/// `v = 1` in `head`, an empty `jump`, `w = v` in `tail`, and `return w`
/// in `exit`, each block the only successor of the one before it.
fn straight_run() -> Function<ToyDialect> {
    let mut builder = FunctionBuilder::<ToyDialect>::new("toy::straight".into());
    let value = builder.declare_variable(0, None).unwrap();
    let copy = builder.declare_variable(0, None).unwrap();
    let head = builder.new_block("head");
    let jump = builder.new_block("jump");
    let tail = builder.new_block("tail");
    let exit = builder.new_block("exit");
    builder
        .append_instruction(
            head,
            Operation::Constant(1),
            Vec::new(),
            vec![integer(value)],
            false,
            Some(Span { start: 1, end: 2 }),
        )
        .unwrap();
    builder
        .append_instruction(
            tail,
            Operation::Copy,
            vec![integer(value)],
            vec![integer(copy)],
            false,
            Some(Span { start: 3, end: 4 }),
        )
        .unwrap();
    builder
        .append_instruction(
            exit,
            Operation::Return,
            vec![integer(copy)],
            Vec::new(),
            false,
            Some(Span { start: 4, end: 5 }),
        )
        .unwrap();
    builder
        .add_edge(builder.entry(), head, Edge::Entry, None)
        .unwrap();
    builder.add_edge(head, jump, Edge::Next, None).unwrap();
    builder.add_edge(jump, tail, Edge::Next, None).unwrap();
    builder.add_edge(tail, exit, Edge::Next, None).unwrap();
    builder.finish().unwrap()
}

#[test]
fn a_straight_run_becomes_one_block_under_the_root() {
    let (joined, removed) = straight_run().merge_straight_blocks().unwrap();

    assert_eq!(removed, 3, "the jump leaves, and head takes tail and exit");
    let cfg = joined.cfg();
    assert_eq!(cfg.block_count(), 2, "the root and one semantic block");
    assert!(cfg.block(cfg.entry()).is_empty(), "the root stays empty");
    let body = cfg
        .successors(cfg.entry())
        .next()
        .expect("the root keeps its entry edge");
    assert_eq!(body.index(), 1, "the entry block keeps its identity");
    let operations: Vec<_> = cfg
        .block(body)
        .instructions()
        .iter()
        .map(|instruction| *instruction.operation())
        .collect();
    assert_eq!(
        operations,
        [Operation::Constant(1), Operation::Copy, Operation::Return]
    );
    let mapped = joined
        .provenance()
        .entries()
        .iter()
        .filter(|entry| matches!(entry.entity, EntityId::Instruction(_)))
        .count();
    assert_eq!(mapped, 3, "each instruction keeps its source span");
}

/// `load` in `head`, which nothing protects, and a protected `load` in
/// `guarded`, whose exceptional edge reaches `pad`.
fn protected_run() -> Function<ToyDialect> {
    let mut builder = FunctionBuilder::<ToyDialect>::new("toy::protected".into());
    let first = builder.declare_variable(0, None).unwrap();
    let second = builder.declare_variable(0, None).unwrap();
    let head = builder.new_block("head");
    let guarded = builder.new_block("guarded");
    let exit = builder.new_block("exit");
    let pad = builder.new_block("pad");
    for (block, variable) in [(head, first), (guarded, second)] {
        builder
            .append_instruction(
                block,
                Operation::Load(0),
                Vec::new(),
                vec![integer(variable)],
                true,
                None,
            )
            .unwrap();
    }
    for block in [exit, pad] {
        builder
            .append_instruction(
                block,
                Operation::Return,
                Vec::new(),
                Vec::new(),
                false,
                None,
            )
            .unwrap();
    }
    builder
        .add_edge(builder.entry(), head, Edge::Entry, None)
        .unwrap();
    builder.add_edge(head, guarded, Edge::Next, None).unwrap();
    builder.add_edge(guarded, exit, Edge::Next, None).unwrap();
    builder.add_edge(guarded, pad, Edge::Unwind, None).unwrap();
    builder.finish().unwrap()
}

#[test]
fn a_throwing_block_takes_no_block_with_an_exceptional_edge() {
    let source = protected_run();
    let (joined, removed) = source.merge_straight_blocks().unwrap();

    assert_eq!(removed, 0, "each load keeps its own block");
    assert_eq!(joined, source);
}

/// A rebuild after a join mirrors the slots that the join left unused, so
/// every block and edge that it copies keeps its identity.
#[test]
fn a_rebuild_after_a_join_keeps_the_identities_of_the_join() {
    let mut builder = FunctionBuilder::<ToyDialect>::new("toy::rejoined".into());
    let unread = builder.declare_variable(0, None).unwrap();
    let head = builder.new_block("head");
    let jump = builder.new_block("jump");
    let exit = builder.new_block("exit");
    builder
        .append_instruction(
            head,
            Operation::Constant(1),
            Vec::new(),
            vec![integer(unread)],
            false,
            None,
        )
        .unwrap();
    builder
        .append_instruction(exit, Operation::Return, Vec::new(), Vec::new(), false, None)
        .unwrap();
    builder
        .add_edge(builder.entry(), head, Edge::Entry, None)
        .unwrap();
    builder.add_edge(head, jump, Edge::Next, None).unwrap();
    builder.add_edge(jump, exit, Edge::Next, None).unwrap();
    let (joined, removed) = builder.finish().unwrap().merge_straight_blocks().unwrap();
    assert_eq!(removed, 2);

    let (rebuilt, dropped) = joined
        .remove_instructions(|instruction| *instruction.operation() == Operation::Constant(1))
        .unwrap();

    assert_eq!(dropped, 1);
    let blocks: Vec<_> = rebuilt.cfg().block_ids().collect();
    assert_eq!(blocks, joined.cfg().block_ids().collect::<Vec<_>>());
    let edges: Vec<_> = rebuilt.cfg().edge_ids().collect();
    assert_eq!(edges, joined.cfg().edge_ids().collect::<Vec<_>>());
}

/// An empty arm of a branch keeps its block, so a fact about the path that
/// the arm takes still has a block to name.
#[test]
fn an_empty_branch_arm_keeps_its_block() {
    let mut builder = FunctionBuilder::<ToyDialect>::new("toy::arms".into());
    let condition = builder.declare_variable(0, None).unwrap();
    let head = builder.new_block("head");
    let taken = builder.new_block("taken");
    let skipped = builder.new_block("skipped");
    let exit = builder.new_block("exit");
    builder
        .append_instruction(
            head,
            Operation::Constant(1),
            Vec::new(),
            vec![integer(condition)],
            false,
            None,
        )
        .unwrap();
    builder
        .append_instruction(
            head,
            Operation::Branch,
            vec![integer(condition)],
            Vec::new(),
            false,
            None,
        )
        .unwrap();
    builder
        .append_instruction(exit, Operation::Return, Vec::new(), Vec::new(), false, None)
        .unwrap();
    builder
        .add_edge(builder.entry(), head, Edge::Entry, None)
        .unwrap();
    for (source, target) in [
        (head, taken),
        (head, skipped),
        (taken, exit),
        (skipped, exit),
    ] {
        builder.add_edge(source, target, Edge::Next, None).unwrap();
    }
    let function = builder.finish().unwrap();

    let (joined, removed) = function.merge_straight_blocks().unwrap();

    assert_eq!(removed, 0, "each arm keeps its block");
    let arms: Vec<_> = joined.cfg().successors(head).collect();
    assert_eq!(arms, [taken, skipped]);
}

/// An empty jump inside a protected region leaves the region, and so does
/// an empty jump after a throw site, whose exceptional edge is no branch.
/// The handler entry and the protected throw site keep their blocks.
#[test]
fn a_protected_empty_block_leaves_its_region() {
    let mut builder = FunctionBuilder::<ToyDialect>::new("toy::region".into());
    let value = builder.declare_variable(0, None).unwrap();
    let loaded = builder.declare_variable(0, None).unwrap();
    let head = builder.new_block("head");
    let jump = builder.new_block("jump");
    let guarded = builder.new_block("guarded");
    let after = builder.new_block("after");
    let tail = builder.new_block("tail");
    let pad = builder.new_block("pad");
    builder
        .append_instruction(
            head,
            Operation::Constant(1),
            Vec::new(),
            vec![integer(value)],
            false,
            None,
        )
        .unwrap();
    builder
        .append_instruction(
            guarded,
            Operation::Load(0),
            Vec::new(),
            vec![integer(loaded)],
            true,
            None,
        )
        .unwrap();
    for block in [tail, pad] {
        builder
            .append_instruction(
                block,
                Operation::Return,
                Vec::new(),
                Vec::new(),
                false,
                None,
            )
            .unwrap();
    }
    builder
        .add_edge(builder.entry(), head, Edge::Entry, None)
        .unwrap();
    builder.add_edge(head, jump, Edge::Next, None).unwrap();
    builder.add_edge(jump, guarded, Edge::Next, None).unwrap();
    builder.add_edge(guarded, after, Edge::Next, None).unwrap();
    builder.add_edge(after, tail, Edge::Next, None).unwrap();
    builder.add_edge(guarded, pad, Edge::Unwind, None).unwrap();
    builder
        .add_region(crate::Region {
            id: crate::RegionId::from_raw(0),
            protected_blocks: [jump, guarded, after].into_iter().collect(),
            handlers: vec![crate::Handler {
                entry: pad,
                body: crate::HandlerBody::known([pad]),
                kind: crate::HandlerKind::CatchAll,
            }],
            parent: None,
        })
        .unwrap();
    let function = builder.finish().unwrap();

    let (joined, removed) = function.merge_straight_blocks().unwrap();

    assert_eq!(removed, 2, "the two empty jumps leave");
    let cfg = joined.cfg();
    assert_eq!(cfg.successors(head).collect::<Vec<_>>(), [guarded]);
    assert!(cfg.successors(guarded).any(|block| block == tail));
    let region = &cfg.regions()[0];
    assert_eq!(
        region.protected_blocks.iter().copied().collect::<Vec<_>>(),
        [guarded]
    );
    assert_eq!(region.handlers[0].entry, pad);
}
