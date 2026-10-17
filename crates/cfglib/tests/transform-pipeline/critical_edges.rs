//! Critical-edge splitting: the unit-payload wrapper, the default-payload
//! form, and the caller-payload form on payload-bearing CFGs.

use cfglib::{
    BlockId, Cfg, DominatorTree, EdgeKind, cfg_block_nesting_depths, detect_loops_tagged,
    split_critical_edges, split_critical_edges_mapped, split_critical_edges_with,
};

use crate::oracles::{
    Placement, assert_rewrite_accounts_for_identities, assert_surviving_payloads_unchanged,
    assert_valid, critical_edges,
};
use crate::programs::random_program;
use crate::toy::{Builder, Inst, Payload, Route, assert_same_behavior};

const SEEDS: u64 = 80;

#[test]
fn split_critical_edges_leaves_no_edge_from_a_branch_to_a_join() {
    let mut total = 0;
    for seed in 0..SEEDS {
        let before = random_program::<()>(seed, true);
        let critical = critical_edges(&before);
        let mut cfg = before.clone();
        let split = split_critical_edges(&mut cfg);
        let context = format!("seed {seed}");

        assert_eq!(split, critical.len(), "{context}");
        assert!(
            critical_edges(&cfg).is_empty(),
            "{context}: critical edge remains"
        );
        assert_eq!(cfg.block_count(), before.block_count() + split, "{context}");
        assert_valid(&cfg, &context);
        assert_same_behavior(&before, &cfg, &context);
        assert_eq!(split_critical_edges(&mut cfg), 0, "{context}: second run");
        total += split;
    }
    assert!(total > 0, "the generator must produce critical edges");
}

fn mapped_case<E: Payload>(seed: u64) {
    let before = random_program::<E>(seed, true);
    let critical = critical_edges(&before);
    let mut cfg = before.clone();
    let (split, rewrite) = split_critical_edges_mapped(&mut cfg);
    let context = format!("seed {seed}");

    assert_eq!(split, critical.len(), "{context}");
    assert_eq!(critical_edges(&cfg), [], "{context}");
    assert_valid(&cfg, &context);
    assert_same_behavior(&before, &cfg, &context);
    assert_surviving_payloads_unchanged(&before, &cfg, &context);
    let placement = Placement {
        rewrite: &rewrite,
        renumbering: None,
    };
    assert_rewrite_accounts_for_identities(&before, &cfg, &placement, &context);

    assert_eq!(rewrite.created_blocks().len(), split, "{context}");
    for &edge in &critical {
        let [first, second] = rewrite.edges(edge).expect("a split edge is recorded") else {
            panic!("{context}: {edge} maps to both halves");
        };
        let middle = cfg.edge(*first).target();
        assert_eq!(
            *first, edge,
            "{context}: the original identity stays on the first half"
        );
        let original_kind = before.edge(edge).kind();
        if original_kind == EdgeKind::Back {
            assert_eq!(
                cfg.edge(*second).kind(),
                EdgeKind::Back,
                "{context}: the back tag enters the header"
            );
            assert_ne!(
                cfg.edge(*first).kind(),
                EdgeKind::Back,
                "{context}: and only there"
            );
        } else {
            assert_eq!(cfg.edge(*first).kind(), original_kind, "{context}");
            assert_eq!(cfg.edge(*second).kind(), EdgeKind::Fallthrough, "{context}");
        }
        assert!(rewrite.created_blocks().contains(&middle), "{context}");
        assert!(rewrite.created_edges().contains(second), "{context}");
        assert_eq!(cfg.edge(*second).source(), middle, "{context}");
        assert_eq!(
            cfg.edge(*second).target(),
            before.edge(edge).target(),
            "{context}"
        );
        assert_eq!(cfg.edge(*second).payload(), &E::default(), "{context}");
        assert!(
            cfg.block(middle).is_empty(),
            "{context}: the inserted block holds no code"
        );
        assert_eq!(cfg.outgoing(middle).count(), 1, "{context}");
        assert_eq!(cfg.incoming(middle).count(), 1, "{context}");
    }
}

#[test]
fn split_critical_edges_mapped_records_both_halves_and_defaults_the_new_payload() {
    for seed in 0..SEEDS {
        mapped_case::<()>(seed);
        mapped_case::<Route>(seed);
    }
}

#[test]
fn split_critical_edges_with_asks_for_the_payload_of_each_synthetic_half() {
    for seed in 0..SEEDS {
        let before = random_program::<Route>(seed, false);
        let critical = critical_edges(&before);
        let mut cfg = before.clone();
        let mut asked = Vec::new();
        let (split, rewrite) = split_critical_edges_with(&mut cfg, |edge, original| {
            asked.push(edge);
            match original.payload() {
                Route::Tagged(tag) => Route::Tagged(tag + 5_000),
                Route::Synthetic => Route::Synthetic,
            }
        });

        assert_eq!(split, critical.len(), "seed {seed}");
        assert_eq!(
            asked.len(),
            split,
            "seed {seed}: one payload request per critical edge"
        );
        for &edge in &critical {
            assert!(
                asked.contains(&edge),
                "seed {seed}: {edge} was never asked about"
            );
            let [first, second] = rewrite.edges(edge).unwrap() else {
                unreachable!()
            };
            assert_eq!(
                cfg.edge(*first).payload(),
                before.edge(edge).payload(),
                "seed {seed}"
            );
            let Route::Tagged(tag) = before.edge(edge).payload().clone() else {
                unreachable!("generated edges are tagged");
            };
            assert_eq!(
                cfg.edge(*second).payload(),
                &Route::Tagged(tag + 5_000),
                "seed {seed}: the closure saw the original edge's payload"
            );
        }
        assert_same_behavior(&before, &cfg, &format!("seed {seed}"));
    }
}

#[test]
fn parallel_switch_edges_into_a_join_are_split_one_by_one() {
    let mut builder = Builder::<Route>::new();
    let entry = builder.entry();
    builder.push(entry, Inst::branch(0));
    let target = builder.block([Inst::print(2)]);
    let other = builder.block([Inst::print(3)]);
    let third = builder.block([Inst::print(0)]);
    let first = builder.edge(entry, target, EdgeKind::SwitchCase);
    let second = builder.edge(entry, target, EdgeKind::SwitchCase);
    builder.edge(entry, other, EdgeKind::SwitchCase);
    builder.edge(other, target, EdgeKind::Fallthrough);
    builder.edge(target, third, EdgeKind::Fallthrough);
    let mut cfg = builder.finish();

    let (split, rewrite) = split_critical_edges_mapped(&mut cfg);

    assert_eq!(split, 2, "each parallel edge is its own critical edge");
    assert_eq!(rewrite.created_blocks().len(), 2);
    let middles: Vec<BlockId> = [first, second].map(|edge| cfg.edge(edge).target()).into();
    assert_ne!(middles[0], middles[1]);
    for (middle, edge) in middles.into_iter().zip([first, second]) {
        assert_eq!(cfg.edge(edge).kind(), EdgeKind::SwitchCase);
        assert_eq!(cfg.incoming(middle).count(), 1);
        assert_eq!(cfg.successors(middle).collect::<Vec<_>>(), [target]);
    }
    assert_eq!(cfg.incoming(target).count(), 3);
    assert_eq!(critical_edges(&cfg), []);
    assert_valid(&cfg, "parallel switch edges");
}

fn self_loop_case<E: Payload>() -> (Cfg<Inst, E>, BlockId, BlockId) {
    let mut builder = Builder::<E>::new();
    let entry = builder.entry();
    builder.push(entry, Inst::constant(5, 0));
    builder.push(entry, Inst::constant(4, 1));
    let body = builder.block([
        Inst::add(5, 5, 4),
        Inst::print(5),
        Inst::less(6, 5, 1),
        Inst::branch(6),
    ]);
    let exit = builder.block([Inst::ret()]);
    builder.edge(entry, body, EdgeKind::Fallthrough);
    builder.edge(body, body, EdgeKind::Back);
    builder.edge(body, exit, EdgeKind::ConditionalFalse);
    (builder.finish(), entry, body)
}

#[test]
fn splitting_a_critical_self_loop_keeps_the_loop_and_its_behavior() {
    let (before, entry, body) = self_loop_case::<Route>();
    let mut cfg = before.clone();
    let (split, rewrite) = split_critical_edges_mapped(&mut cfg);

    assert_eq!(split, 1);
    let [middle] = rewrite.created_blocks() else {
        unreachable!()
    };
    assert_eq!(cfg.successors(*middle).collect::<Vec<_>>(), [body]);
    assert_eq!(critical_edges(&cfg), []);
    assert_same_behavior(&before, &cfg, "self-loop");
    assert_ne!(cfg.entry(), *middle);
    assert_eq!(cfg.entry(), entry);
}

#[test]
fn loop_structure_survives_critical_edge_splitting_with_back_tags() {
    let (before, entry, body) = self_loop_case::<()>();
    let loops_before = detect_loops_tagged(&before, &DominatorTree::compute(&before));
    assert_eq!(loops_before.len(), 1);
    assert_eq!(loops_before[0].header, body);

    let mut cfg = before.clone();
    split_critical_edges(&mut cfg);
    let loops = detect_loops_tagged(&cfg, &DominatorTree::compute(&cfg));

    assert_eq!(
        loops.iter().map(|lp| lp.header).collect::<Vec<_>>(),
        [body],
        "a back-tagged edge that was split must not invent a second loop"
    );
    assert_eq!(loops[0].latches.len(), 1);
    assert_eq!(
        cfg_block_nesting_depths(&cfg)[entry.index()],
        0,
        "code before the loop is not inside it"
    );
}
