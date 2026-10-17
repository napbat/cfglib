//! Dead-code passes: instruction removal against a brute-force liveness
//! oracle, caller-declared live-outs, and the DCE-plus-cleanup fixpoint.

use cfglib::{
    EdgeKind, dead_code_elimination, dead_code_elimination_with_exits, remove_dead_code,
    remove_dead_code_mapped, simplify,
};

use crate::oracles::{
    Placement, assert_fully_simplified, assert_rewrite_accounts_for_identities,
    assert_surviving_edges_unchanged, assert_valid, classic_dead_code_survivors,
    instruction_multiset, live_blocks,
};
use crate::programs::random_program;
use crate::toy::{Builder, Inst, OBSERVED, Payload, Route, assert_same_behavior_observing, run};

const SEEDS: u64 = 80;

fn instruction_count<E>(cfg: &cfglib::Cfg<Inst, E>) -> usize {
    cfg.block_ids()
        .map(|block| cfg.block(block).instructions().len())
        .sum()
}

#[test]
fn dead_code_elimination_removes_a_whole_dead_chain_but_no_effect() {
    let mut builder = Builder::<()>::new();
    let entry = builder.entry();
    for instruction in [
        Inst::add(10, 0, 2),
        Inst::add(11, 10, 10),
        Inst::add(12, 11, 11),
        Inst::print(0),
        Inst::constant(13, 4),
        Inst::print(13),
    ] {
        builder.push(entry, instruction);
    }
    let mut cfg = builder.finish();

    assert_eq!(dead_code_elimination(&mut cfg), 3);
    assert_eq!(
        cfg.block(entry).instructions(),
        [Inst::print(0), Inst::constant(13, 4), Inst::print(13)],
        "the chain feeding only a dead result goes in one call; used values stay"
    );
}

#[test]
fn dead_code_elimination_removes_a_chain_that_spans_blocks_in_one_call() {
    let mut builder = Builder::<Route>::new();
    let entry = builder.entry();
    builder.push(entry, Inst::add(10, 0, 2));
    let middle = builder.block([Inst::add(11, 10, 10)]);
    let last = builder.block([Inst::add(12, 11, 11), Inst::print(0), Inst::ret()]);
    builder.edge(entry, middle, EdgeKind::Fallthrough);
    builder.edge(middle, last, EdgeKind::Fallthrough);
    let mut cfg = builder.finish();

    assert_eq!(
        dead_code_elimination(&mut cfg),
        3,
        "a chain feeding only a dead instruction is dead with it, wherever its links sit"
    );
    assert_eq!(cfg.block(entry).instructions(), []);
    assert_eq!(cfg.block(middle).instructions(), []);
    assert_eq!(
        cfg.block(last).instructions(),
        [Inst::print(0), Inst::ret()]
    );
}

fn elimination_case<E: Payload>(seed: u64) {
    let before = random_program::<E>(seed, true);
    let mut cfg = before.clone();
    let removed = dead_code_elimination(&mut cfg);
    let context = format!("seed {seed}");

    assert_eq!(
        instruction_count(&before) - instruction_count(&cfg),
        removed,
        "{context}"
    );
    assert_same_behavior_observing(&before, &cfg, &[], &context);
    assert_eq!(
        live_blocks(&cfg),
        live_blocks(&before),
        "{context}: blocks stay"
    );
    for edge in before.edge_ids() {
        let (old, new) = (before.edge(edge), cfg.edge(edge));
        assert_eq!(
            (old.source(), old.target()),
            (new.source(), new.target()),
            "{context}"
        );
    }
    assert_surviving_edges_unchanged(&before, &cfg, &context);
    assert_valid(&cfg, &context);
}

#[test]
fn dead_code_elimination_preserves_behavior_and_leaves_the_graph_shape_alone() {
    for seed in 0..SEEDS {
        elimination_case::<()>(seed);
        elimination_case::<Route>(seed);
    }
}

fn exits_case<E: Payload>(seed: u64) {
    let before = random_program::<E>(seed, false);
    let exits: Vec<_> = before.exit_blocks().collect();
    let mut cfg = before.clone();
    dead_code_elimination_with_exits(&mut cfg, |block| {
        if exits.contains(&block) {
            OBSERVED.to_vec()
        } else {
            Vec::new()
        }
    });

    let context = format!("seed {seed}");
    assert_same_behavior_observing(&before, &cfg, &OBSERVED, &context);
    let mut alone = before.clone();
    dead_code_elimination(&mut alone);
    assert!(
        instruction_count(&alone) <= instruction_count(&cfg),
        "{context}: declaring live-outs can only keep more"
    );
}

#[test]
fn dead_code_elimination_with_exits_preserves_the_registers_it_was_told_about() {
    for seed in 0..SEEDS {
        exits_case::<()>(seed);
        exits_case::<Route>(seed);
    }
}

#[test]
fn dead_code_elimination_with_exits_keeps_values_the_caller_reads_afterwards() {
    let build = || {
        let mut builder = Builder::<()>::new();
        let entry = builder.entry();
        let exit = builder.block([Inst::add(7, 0, 2), Inst::constant(8, 5), Inst::add(9, 7, 7)]);
        builder.edge(entry, exit, EdgeKind::Fallthrough);
        (builder.finish(), exit)
    };

    let (mut alone, _) = build();
    assert_eq!(
        dead_code_elimination(&mut alone),
        3,
        "nothing leaves the graph"
    );

    let (mut with_exits, exit) = build();
    let removed = dead_code_elimination_with_exits(&mut with_exits, |block| {
        if block == exit { vec![7] } else { vec![] }
    });
    assert_eq!(removed, 2, "r8 and r9 are dead; r7 is read by the caller");
    assert_eq!(with_exits.block(exit).instructions(), [Inst::add(7, 0, 2)]);

    let (original, _) = build();
    let inputs = [(0, 3), (2, 4)];
    let before = run(&original, &inputs, &[7]).unwrap();
    assert_eq!(run(&with_exits, &inputs, &[7]), Ok(before));
}

fn fixpoint_case<E: Payload>(seed: u64) {
    let before = random_program::<E>(seed, true);
    let mut cfg = before.clone();
    let total = remove_dead_code(&mut cfg);
    let context = format!("seed {seed}");

    assert_eq!(
        total > 0,
        cfg != before,
        "{context}: the count reports whether anything changed"
    );
    assert_valid(&cfg, &context);
    assert_same_behavior_observing(&before, &cfg, &[], &context);
    assert_eq!(
        instruction_multiset(&cfg, &live_blocks(&cfg)),
        classic_dead_code_survivors(&before),
        "{context}: exactly the instructions that classic iterated DCE keeps survive"
    );
    assert_fully_simplified(&cfg, &context);

    let settled = cfg.clone();
    assert_eq!(dead_code_elimination(&mut cfg), 0, "{context}");
    assert_eq!(simplify(&mut cfg), 0, "{context}");
    assert_eq!(
        cfg, settled,
        "{context}: the result is a fixpoint of both passes"
    );
}

#[test]
fn remove_dead_code_reaches_the_fixpoint_of_elimination_and_cleanup() {
    for seed in 0..SEEDS {
        fixpoint_case::<()>(seed);
        fixpoint_case::<Route>(seed);
    }
}

#[test]
fn remove_dead_code_collapses_blocks_that_elimination_empties() {
    let mut builder = Builder::<Route>::new();
    let entry = builder.entry();
    builder.push(entry, Inst::constant(9, 1));
    let middle = builder.block([Inst::add(10, 9, 9)]);
    let last = builder.block([Inst::print(0), Inst::ret()]);
    builder.edge(entry, middle, EdgeKind::Fallthrough);
    builder.edge(middle, last, EdgeKind::Fallthrough);
    let before = builder.finish();
    let mut cfg = before.clone();

    assert!(remove_dead_code(&mut cfg) > 0);
    assert_eq!(cfg.block_count(), 1);
    assert_eq!(
        cfg.block(entry).instructions(),
        [Inst::print(0), Inst::ret()]
    );
    assert_same_behavior_observing(&before, &cfg, &[], "collapsed chain");
}

fn mapped_case<E: Payload>(seed: u64) {
    let before = random_program::<E>(seed, true);
    let mut cfg = before.clone();
    let (_, rewrite) = remove_dead_code_mapped(&mut cfg);
    let context = format!("seed {seed}");

    let placement = Placement {
        rewrite: &rewrite,
        renumbering: None,
    };
    assert_rewrite_accounts_for_identities(&before, &cfg, &placement, &context);
    assert_surviving_edges_unchanged(&before, &cfg, &context);

    let renumbering = cfg.compact();
    let carried = rewrite.then_renumbered(&renumbering);
    let compacted = Placement {
        rewrite: &carried,
        renumbering: Some(&renumbering),
    };
    assert_rewrite_accounts_for_identities(&before, &cfg, &compacted, &context);
    assert_same_behavior_observing(&before, &cfg, &[], &context);
}

#[test]
fn remove_dead_code_mapped_composes_the_cleanup_rewrites_across_rounds() {
    for seed in 0..SEEDS {
        mapped_case::<()>(seed);
        mapped_case::<Route>(seed);
    }
}
