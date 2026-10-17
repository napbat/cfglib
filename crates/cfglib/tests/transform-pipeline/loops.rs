//! Loop passes: rotation, invariant discovery with a hand-written hoist,
//! preheader insertion and canonicalization, irreducible-flow repair, and
//! CFG reversal.

use std::collections::BTreeSet;

use cfglib::{
    BlockId, Cfg, DominatorTree, EdgeKind, NaturalLoop, canonicalize_loops, detect_loops,
    find_loop_invariants, insert_preheader, is_reducible, loop_exit_blocks, make_reducible,
    rotate_loop,
};

use crate::oracles::assert_valid;
use crate::programs::random_program;
use crate::toy::{BinaryOp, Builder, Inst, assert_same_behavior};

/// A counted while-loop: the header tests, the body accumulates.
struct WhileLoop {
    cfg: Cfg<Inst>,
    header: BlockId,
    body: BlockId,
    exit: BlockId,
}

fn while_loop() -> WhileLoop {
    let mut builder = Builder::<()>::new();
    let entry = builder.entry();
    for instruction in [
        Inst::constant(5, 0),
        Inst::constant(4, 1),
        Inst::constant(6, 0),
    ] {
        builder.push(entry, instruction);
    }
    let header = builder.block([Inst::less(9, 5, 1), Inst::branch(9)]);
    let body = builder.block([
        Inst::binary(BinaryOp::Multiply, 7, 0, 2),
        Inst::add(8, 7, 7),
        Inst::add(6, 6, 8),
        Inst::add(5, 5, 4),
    ]);
    let exit = builder.block([Inst::print(6), Inst::print(5), Inst::ret()]);
    builder.edge(entry, header, EdgeKind::Fallthrough);
    builder.edge(header, body, EdgeKind::ConditionalTrue);
    builder.edge(header, exit, EdgeKind::ConditionalFalse);
    builder.edge(body, header, EdgeKind::Back);
    WhileLoop {
        cfg: builder.finish(),
        header,
        body,
        exit,
    }
}

fn loops_of(cfg: &Cfg<Inst>) -> Vec<NaturalLoop> {
    detect_loops(cfg, &DominatorTree::compute(cfg))
}

#[test]
fn rotation_makes_a_bottom_tested_loop_that_runs_like_the_original() {
    let WhileLoop {
        cfg: before,
        header,
        body,
        exit,
    } = while_loop();
    let mut cfg = before.clone();
    let lp = loops_of(&cfg).remove(0);

    let rotation = rotate_loop(&mut cfg, &lp).expect("a conditional header with one latch rotates");

    assert_eq!(rotation.new_header, body);
    let copy = rotation.bottom_test;
    assert_eq!(
        cfg.block(copy).instructions(),
        before.block(header).instructions()
    );
    let successors: BTreeSet<BlockId> = cfg.successors(copy).collect();
    assert_eq!(
        successors,
        BTreeSet::from([body, exit]),
        "the test copy decides between the body and the exit"
    );
    assert_eq!(
        cfg.successors(body).collect::<Vec<_>>(),
        [copy],
        "the latch now reaches the copy"
    );
    assert_eq!(
        cfg.incoming(header).count(),
        1,
        "the original test runs once, from the entry"
    );
    assert_valid(&cfg, "rotated loop");
    assert_same_behavior(&before, &cfg, "rotated while-loop");

    let rotated = loops_of(&cfg);
    assert_eq!(rotated.len(), 1);
    assert_eq!(rotated[0].header, body, "the body is the new loop header");
    assert_eq!(rotated[0].body, BTreeSet::from([body, copy]));
}

#[test]
fn rotation_preserves_behavior_of_every_loop_of_random_programs() {
    let mut rotated = 0;
    for seed in 0..60 {
        let before = random_program::<()>(seed, false);
        for lp in loops_of(&before) {
            let mut cfg = before.clone();
            if rotate_loop(&mut cfg, &lp).is_some() {
                rotated += 1;
                let context = format!("seed {seed}, loop at {}", lp.header);
                assert_valid(&cfg, &context);
                assert_same_behavior(&before, &cfg, &context);
            }
        }
    }
    assert!(
        rotated > 20,
        "only {rotated} loops rotated; the generator lost its loops"
    );
}

#[test]
fn rotation_declines_loops_it_cannot_rotate_and_leaves_them_untouched() {
    let mut builder = Builder::<()>::new();
    let entry = builder.entry();
    let header = builder.block([Inst::less(9, 5, 1), Inst::branch(9)]);
    let left = builder.block([Inst::print(2)]);
    let right = builder.block([Inst::print(3)]);
    let exit = builder.block([Inst::ret()]);
    builder.edge(entry, header, EdgeKind::Fallthrough);
    builder.edge(header, left, EdgeKind::ConditionalTrue);
    builder.edge(header, exit, EdgeKind::ConditionalFalse);
    builder.edge(left, right, EdgeKind::ConditionalTrue);
    builder.edge(left, header, EdgeKind::Back);
    builder.edge(right, header, EdgeKind::Back);
    let two_latches = builder.finish();
    let lp = loops_of(&two_latches).remove(0);
    assert_eq!(lp.latches.len(), 2);
    let mut cfg = two_latches.clone();
    assert!(rotate_loop(&mut cfg, &lp).is_none(), "two latches");
    assert_eq!(cfg, two_latches);

    let mut builder = Builder::<()>::new();
    let entry = builder.entry();
    let spin = builder.block([Inst::print(2)]);
    builder.edge(entry, spin, EdgeKind::Fallthrough);
    builder.edge(spin, spin, EdgeKind::Back);
    let unconditional = builder.finish();
    let lp = loops_of(&unconditional).remove(0);
    let mut cfg = unconditional.clone();
    assert!(
        rotate_loop(&mut cfg, &lp).is_none(),
        "the header does not branch"
    );
    assert_eq!(cfg, unconditional);

    let mut builder = Builder::<()>::new();
    let entry = builder.entry();
    let header = builder.block([Inst::branch(9)]);
    let first = builder.block([Inst::print(2)]);
    let second = builder.block([Inst::print(3)]);
    builder.edge(entry, header, EdgeKind::Fallthrough);
    builder.edge(header, first, EdgeKind::ConditionalTrue);
    builder.edge(header, second, EdgeKind::ConditionalFalse);
    builder.edge(first, header, EdgeKind::Back);
    builder.edge(second, header, EdgeKind::Back);
    let no_exit = builder.finish();
    let lp = loops_of(&no_exit).remove(0);
    let mut cfg = no_exit.clone();
    assert!(
        rotate_loop(&mut cfg, &lp).is_none(),
        "both header successors stay in the loop"
    );
    assert_eq!(cfg, no_exit);
}

#[test]
fn invariants_are_the_instructions_whose_inputs_the_loop_never_changes() {
    let WhileLoop {
        cfg, header, body, ..
    } = while_loop();
    let lp = loops_of(&cfg).remove(0);

    let invariants = find_loop_invariants(&cfg, &lp);

    assert_eq!(
        invariants,
        [(body, 0), (body, 1)],
        "r7 = r0 * r2 reads nothing the loop writes, and r8 = r7 + r7 reads only r7"
    );
    assert!(
        !invariants.iter().any(|&(block, _)| block == header),
        "the header's test reads the counter the loop increments"
    );
}

#[test]
fn hoisting_the_reported_invariants_into_a_preheader_keeps_the_behavior() {
    let WhileLoop {
        cfg: before,
        header,
        body,
        ..
    } = while_loop();
    let mut cfg = before.clone();
    let dom = DominatorTree::compute(&cfg);
    let canonical = canonicalize_loops(&mut cfg, &dom);
    let [only] = canonical.as_slice() else {
        panic!("one loop: {canonical:?}")
    };
    assert_eq!(only.natural_loop.header, header);

    let invariants = find_loop_invariants(&cfg, &only.natural_loop);
    assert_ne!(invariants, Vec::new());
    let hoisted: Vec<Inst> = invariants
        .iter()
        .map(|&(block, index)| cfg.block(block).instructions()[index].clone())
        .collect();
    let mut remaining = cfg.block(body).instructions().to_vec();
    for &(block, index) in invariants.iter().rev() {
        assert_eq!(block, body);
        remaining.remove(index);
    }
    *cfg.block_mut(body).instructions_mut() = remaining;
    cfg.block_mut(only.preheader)
        .instructions_mut()
        .extend(hoisted);

    assert_valid(&cfg, "after hoisting");
    assert_same_behavior(&before, &cfg, "loop-invariant code motion");
}

struct TwoEntries {
    cfg: Cfg<Inst>,
    header: BlockId,
    branches: [BlockId; 2],
}

fn two_entry_loop() -> TwoEntries {
    let mut builder = Builder::<()>::new();
    let entry = builder.entry();
    for instruction in [
        Inst::constant(5, 0),
        Inst::constant(4, 1),
        Inst::less(9, 0, 4),
        Inst::branch(9),
    ] {
        builder.push(entry, instruction);
    }
    let first = builder.block([Inst::print(0)]);
    let second = builder.block([Inst::print(2)]);
    let header = builder.block([Inst::less(9, 5, 1), Inst::branch(9)]);
    let body = builder.block([Inst::add(5, 5, 4), Inst::print(5)]);
    let exit = builder.block([Inst::ret()]);
    builder.edge(entry, first, EdgeKind::ConditionalTrue);
    builder.edge(entry, second, EdgeKind::ConditionalFalse);
    builder.edge(first, header, EdgeKind::Jump);
    builder.edge(second, header, EdgeKind::Fallthrough);
    builder.edge(header, body, EdgeKind::ConditionalTrue);
    builder.edge(header, exit, EdgeKind::ConditionalFalse);
    builder.edge(body, header, EdgeKind::Back);
    TwoEntries {
        cfg: builder.finish(),
        header,
        branches: [first, second],
    }
}

#[test]
fn a_preheader_collects_every_outside_entry_into_the_loop() {
    let TwoEntries {
        cfg: before,
        header,
        branches,
    } = two_entry_loop();
    let mut cfg = before.clone();
    let lp = loops_of(&cfg).remove(0);

    let preheader = insert_preheader(&mut cfg, &lp).expect("two outside predecessors need one");

    assert_eq!(cfg.successors(preheader).collect::<Vec<_>>(), [header]);
    assert!(cfg.block(preheader).is_empty());
    let predecessors: BTreeSet<BlockId> = cfg.predecessors(header).collect();
    assert_eq!(
        predecessors,
        BTreeSet::from([preheader, *lp.latches.first().unwrap()])
    );
    for branch in branches {
        assert_eq!(cfg.successors(branch).collect::<Vec<_>>(), [preheader]);
    }
    let kinds: BTreeSet<&str> = branches
        .iter()
        .map(|&branch| cfg.edge(cfg.outgoing(branch).next().unwrap()).kind().name())
        .collect();
    assert_eq!(
        kinds,
        BTreeSet::from([EdgeKind::Jump.name(), EdgeKind::Fallthrough.name()]),
        "entry edge kinds survive"
    );
    assert_valid(&cfg, "preheader");
    assert_same_behavior(&before, &cfg, "preheader insertion");
}

#[test]
fn no_preheader_is_inserted_when_the_loop_has_a_single_outside_entry() {
    let WhileLoop { cfg: before, .. } = while_loop();
    let mut cfg = before.clone();
    let lp = loops_of(&cfg).remove(0);

    assert_eq!(insert_preheader(&mut cfg, &lp), None);
    assert_eq!(cfg, before);

    let mut builder = Builder::<()>::new();
    let entry = builder.entry();
    builder.push(entry, Inst::branch(0));
    let exit = builder.block([Inst::ret()]);
    builder.edge(entry, entry, EdgeKind::Back);
    builder.edge(entry, exit, EdgeKind::ConditionalFalse);
    let mut entry_loop = builder.finish();
    let lp = loops_of(&entry_loop).remove(0);
    assert_eq!(lp.header, entry);
    assert_eq!(
        insert_preheader(&mut entry_loop, &lp),
        None,
        "the entry has no outside entry to collect"
    );
}

fn nested_loops() -> Cfg<Inst> {
    let mut builder = Builder::<()>::new();
    let entry = builder.entry();
    for instruction in [
        Inst::constant(5, 0),
        Inst::constant(4, 1),
        Inst::less(9, 0, 4),
        Inst::branch(9),
    ] {
        builder.push(entry, instruction);
    }
    let side = builder.block([Inst::print(0), Inst::constant(5, 0)]);
    let outer = builder.block([Inst::less(9, 5, 1), Inst::branch(9)]);
    let enter_inner = builder.block([Inst::constant(6, 0)]);
    let inner = builder.block([Inst::less(10, 6, 1), Inst::branch(10)]);
    let inner_body = builder.block([Inst::add(6, 6, 4), Inst::print(6)]);
    let outer_latch = builder.block([Inst::add(5, 5, 4)]);
    let exit = builder.block([Inst::print(5), Inst::ret()]);
    builder.edge(entry, side, EdgeKind::ConditionalTrue);
    builder.edge(entry, outer, EdgeKind::ConditionalFalse);
    builder.edge(side, outer, EdgeKind::Jump);
    builder.edge(outer, enter_inner, EdgeKind::ConditionalTrue);
    builder.edge(outer, exit, EdgeKind::ConditionalFalse);
    builder.edge(enter_inner, inner, EdgeKind::Fallthrough);
    builder.edge(inner, inner_body, EdgeKind::ConditionalTrue);
    builder.edge(inner, outer_latch, EdgeKind::ConditionalFalse);
    builder.edge(inner_body, inner, EdgeKind::Back);
    builder.edge(outer_latch, outer, EdgeKind::Back);
    builder.finish()
}

#[test]
fn canonicalization_gives_every_loop_one_outside_entry_and_reports_its_exits() {
    let before = nested_loops();
    let dom = DominatorTree::compute(&before);
    let expected = detect_loops(&before, &dom);
    assert_eq!(expected.len(), 2);

    let mut cfg = before.clone();
    let canonical = canonicalize_loops(&mut cfg, &dom);

    assert_eq!(canonical.len(), expected.len());
    for loop_info in &canonical {
        let lp = &loop_info.natural_loop;
        let outside: Vec<BlockId> = cfg
            .predecessors(lp.header)
            .filter(|predecessor| !lp.body.contains(predecessor))
            .collect();
        assert_eq!(outside, [loop_info.preheader], "loop at {}", lp.header);
        let brute_force: BTreeSet<BlockId> = lp
            .body
            .iter()
            .flat_map(|&block| before.successors(block))
            .filter(|successor| !lp.body.contains(successor))
            .collect();
        assert_eq!(
            loop_info.exits, brute_force,
            "exits of the loop at {}",
            lp.header
        );
        assert_eq!(loop_info.exits, loop_exit_blocks(&before, lp));
    }
    let preheaders: BTreeSet<BlockId> = canonical
        .iter()
        .map(|loop_info| loop_info.preheader)
        .collect();
    assert_eq!(preheaders.len(), 2, "loops do not share a preheader");
    assert_valid(&cfg, "canonical loops");
    assert_same_behavior(&before, &cfg, "canonicalization");
}

#[test]
fn canonicalization_of_random_programs_preserves_behavior_and_loop_count() {
    for seed in 0..60 {
        let before = random_program::<()>(seed, false);
        let dom = DominatorTree::compute(&before);
        let expected = detect_loops(&before, &dom).len();
        let mut cfg = before.clone();
        let canonical = canonicalize_loops(&mut cfg, &dom);
        let context = format!("seed {seed}");

        assert_eq!(canonical.len(), expected, "{context}");
        assert_valid(&cfg, &context);
        assert_same_behavior(&before, &cfg, &context);
        for loop_info in &canonical {
            let lp = &loop_info.natural_loop;
            let outside = cfg
                .predecessors(lp.header)
                .filter(|predecessor| !lp.body.contains(predecessor))
                .count();
            assert_eq!(outside, 1, "{context}: loop at {}", lp.header);
        }
    }
}

fn irreducible() -> Cfg<Inst> {
    let mut builder = Builder::<()>::new();
    let entry = builder.entry();
    for instruction in [
        Inst::constant(5, 0),
        Inst::constant(4, 1),
        Inst::less(9, 0, 4),
        Inst::branch(9),
    ] {
        builder.push(entry, instruction);
    }
    let first = builder.block([
        Inst::add(5, 5, 4),
        Inst::print(5),
        Inst::less(9, 5, 1),
        Inst::branch(9),
    ]);
    let second = builder.block([
        Inst::add(5, 5, 4),
        Inst::add(5, 5, 4),
        Inst::print(5),
        Inst::less(9, 5, 1),
        Inst::branch(9),
    ]);
    let exit = builder.block([Inst::ret()]);
    builder.edge(entry, first, EdgeKind::ConditionalTrue);
    builder.edge(entry, second, EdgeKind::ConditionalFalse);
    builder.edge(first, second, EdgeKind::ConditionalTrue);
    builder.edge(first, exit, EdgeKind::ConditionalFalse);
    builder.edge(second, first, EdgeKind::ConditionalTrue);
    builder.edge(second, exit, EdgeKind::ConditionalFalse);
    builder.finish()
}

#[test]
fn node_splitting_makes_an_irreducible_cycle_reducible_without_changing_output() {
    let before = irreducible();
    assert!(!is_reducible(&before, &DominatorTree::compute(&before)));
    assert!(
        loops_of(&before).is_empty(),
        "neither entry of the cycle dominates the other"
    );

    let mut cfg = before.clone();
    let duplicated = make_reducible(&mut cfg);

    assert!(duplicated > 0);
    assert_eq!(
        cfg.block_count(),
        before.block_count() + duplicated,
        "the count is blocks duplicated"
    );
    assert!(is_reducible(&cfg, &DominatorTree::compute(&cfg)));
    assert!(
        !loops_of(&cfg).is_empty(),
        "the cycle is now a natural loop"
    );
    assert_valid(&cfg, "reduced");
    assert_same_behavior(&before, &cfg, "node splitting");
    assert_eq!(
        make_reducible(&mut cfg),
        0,
        "a reducible graph is left alone"
    );
}

#[test]
fn reducible_programs_are_not_touched_by_make_reducible() {
    for seed in 0..40 {
        let before = random_program::<()>(seed, true);
        assert!(
            is_reducible(&before, &DominatorTree::compute(&before)),
            "seed {seed}"
        );
        let mut cfg = before.clone();
        assert_eq!(make_reducible(&mut cfg), 0, "seed {seed}");
        assert_eq!(cfg, before, "seed {seed}");
    }
}
