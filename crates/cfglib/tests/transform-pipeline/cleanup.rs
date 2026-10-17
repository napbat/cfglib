//! Cleanup passes: simplify, merge, empty-block bypass, unreachable removal,
//! edge contraction, and node splitting.

use cfglib::{
    Cfg, EdgeKind, SplitPointError, contract_edge, contract_edge_mapped, merge_blocks,
    merge_blocks_mapped, remove_empty_blocks, remove_empty_blocks_mapped, remove_unreachable,
    remove_unreachable_mapped, simplify, simplify_mapped, split_node, split_node_at_points,
    split_node_with_payload_mapped,
};

use crate::oracles::{
    Placement, assert_fully_simplified, assert_images_hold_original_instructions,
    assert_rewrite_accounts_for_identities, assert_surviving_edges_unchanged, assert_valid,
    instruction_multiset, live_blocks, reachable_blocks,
};
use crate::programs::random_program;
use crate::toy::{Builder, Inst, Payload, Route, assert_same_behavior};

const SEEDS: u64 = 60;

fn simplify_case<E: Payload>(seed: u64) {
    let before = random_program::<E>(seed, true);
    let reachable = reachable_blocks(&before);
    let mut cfg = before.clone();
    let total = simplify(&mut cfg);

    let context = format!("seed {seed}");
    assert_valid(&cfg, &context);
    assert_eq!(
        before.block_count() - cfg.block_count(),
        total,
        "{context}: every counted transformation removes exactly one block"
    );
    assert_same_behavior(&before, &cfg, &context);
    assert_fully_simplified(&cfg, &context);
    assert_eq!(
        instruction_multiset(&cfg, &live_blocks(&cfg)),
        instruction_multiset(&before, &reachable),
        "{context}: simplify only moves the instructions of reachable blocks"
    );

    let settled = cfg.clone();
    assert_eq!(simplify(&mut cfg), 0, "{context}: second run finds work");
    assert_eq!(cfg, settled, "{context}: second run changed the graph");
}

#[test]
fn simplify_preserves_behavior_and_reaches_the_documented_fixpoint() {
    for seed in 0..SEEDS {
        simplify_case::<()>(seed);
        simplify_case::<Route>(seed);
    }
}

fn simplify_mapped_case<E: Payload>(seed: u64) {
    let before = random_program::<E>(seed, true);
    let mut cfg = before.clone();
    let (total, rewrite) = simplify_mapped(&mut cfg);
    let context = format!("seed {seed}");

    let placement = Placement {
        rewrite: &rewrite,
        renumbering: None,
    };
    assert_rewrite_accounts_for_identities(&before, &cfg, &placement, &context);
    assert_images_hold_original_instructions(&before, &cfg, &placement, &context);
    assert_surviving_edges_unchanged(&before, &cfg, &context);
    assert_eq!(
        total == 0,
        rewrite.is_empty(),
        "{context}: record and count disagree"
    );

    let renumbering = cfg.compact();
    let carried = rewrite.then_renumbered(&renumbering);
    let compacted = Placement {
        rewrite: &carried,
        renumbering: Some(&renumbering),
    };
    assert_valid(&cfg, &context);
    assert_rewrite_accounts_for_identities(&before, &cfg, &compacted, &context);
    assert_images_hold_original_instructions(&before, &cfg, &compacted, &context);
    assert_same_behavior(&before, &cfg, &context);
}

#[test]
fn simplify_mapped_accounts_for_every_identity_before_and_after_compaction() {
    for seed in 0..SEEDS {
        simplify_mapped_case::<()>(seed);
        simplify_mapped_case::<Route>(seed);
    }
}

fn chain<E: Payload>(length: usize) -> (Cfg<Inst, E>, Vec<cfglib::BlockId>) {
    let mut builder = Builder::<E>::new();
    let mut blocks = vec![builder.entry()];
    builder.push(blocks[0], Inst::constant(2, 0));
    for index in 1..length {
        let block = builder.block([Inst::constant(2, i64::try_from(index).unwrap())]);
        builder.edge(blocks[index - 1], block, EdgeKind::Fallthrough);
        blocks.push(block);
    }
    (builder.finish(), blocks)
}

fn merge_chain_case<E: Payload>() {
    let (mut cfg, blocks) = chain::<E>(4);
    let before = cfg.clone();
    let edges: Vec<_> = before.edge_ids().collect();
    let (merged, rewrite) = merge_blocks_mapped(&mut cfg);

    assert_eq!(merged, 3);
    assert_eq!(live_blocks(&cfg), [blocks[0]].into());
    let expected: Vec<Inst> = (0..4).map(|value| Inst::constant(2, value)).collect();
    assert_eq!(cfg.block(blocks[0]).instructions(), expected);
    assert_eq!(cfg.edge_count(), 0);
    for &block in &blocks[1..] {
        assert_eq!(rewrite.blocks(block), Some([blocks[0]].as_slice()));
    }
    for edge in edges {
        assert_eq!(rewrite.edges(edge), Some([].as_slice()), "{edge}");
    }
    assert_valid(&cfg, "merged chain");
}

#[test]
fn merge_blocks_folds_a_straight_chain_into_its_head_in_order() {
    merge_chain_case::<()>();
    merge_chain_case::<Route>();
}

#[test]
fn merge_blocks_leaves_branch_arms_and_the_entry_alone() {
    let mut builder = Builder::<Route>::new();
    let entry = builder.entry();
    builder.push(entry, Inst::constant(2, 1));
    builder.push(entry, Inst::branch(2));
    let left = builder.block([Inst::print(2)]);
    let right = builder.block([Inst::print(3)]);
    let join = builder.block([Inst::print(2)]);
    let exit = builder.block([Inst::ret()]);
    builder.edge(entry, left, EdgeKind::ConditionalTrue);
    builder.edge(entry, right, EdgeKind::ConditionalFalse);
    builder.edge(left, join, EdgeKind::Fallthrough);
    builder.edge(right, join, EdgeKind::Fallthrough);
    builder.edge(join, exit, EdgeKind::Fallthrough);
    let mut cfg = builder.finish();

    assert_eq!(merge_blocks(&mut cfg), 1, "only the join absorbs its exit");
    assert_eq!(
        cfg.block(join).instructions(),
        [Inst::print(2), Inst::ret()]
    );
    assert!(!cfg.contains_block(exit));
    for block in [entry, left, right, join] {
        assert!(cfg.contains_block(block));
    }
    assert_eq!(cfg.edge_count(), 4);
}

#[test]
fn merge_blocks_never_consumes_an_entry_that_a_loop_returns_to() {
    let mut builder = Builder::<Route>::new();
    let entry = builder.entry();
    builder.push(entry, Inst::constant(2, 1));
    builder.push(entry, Inst::branch(2));
    let body = builder.block([Inst::print(2)]);
    let exit = builder.block([Inst::ret()]);
    builder.edge(entry, body, EdgeKind::ConditionalTrue);
    builder.edge(entry, exit, EdgeKind::ConditionalFalse);
    builder.edge(body, entry, EdgeKind::Back);
    let mut cfg = builder.finish();
    let before = cfg.clone();

    assert_eq!(merge_blocks(&mut cfg), 0);
    assert_eq!(cfg, before);
}

#[test]
fn merge_blocks_keeps_the_label_of_the_block_it_folds_in() {
    let (mut cfg, blocks) = chain::<()>(3);
    cfg.block_mut(blocks[1]).set_label("loop");
    cfg.block_mut(blocks[2]).set_label("tail");

    assert_eq!(merge_blocks(&mut cfg), 2);
    assert_eq!(
        cfg.block(blocks[0]).label(),
        Some("loop"),
        "an unlabeled head takes the first label, exactly as contract_edge does"
    );
}

fn bypass_chain_case<E: Payload>() {
    let mut builder = Builder::<E>::new();
    let entry = builder.entry();
    builder.push(entry, Inst::constant(2, 7));
    let first = builder.block([]);
    let second = builder.block([]);
    let target = builder.block([Inst::print(2)]);
    let into_first = builder.edge(entry, first, EdgeKind::Fallthrough);
    let between = builder.edge(first, second, EdgeKind::Unconditional);
    let out_of_second = builder.edge(second, target, EdgeKind::Fallthrough);
    let mut cfg = builder.finish();
    let before = cfg.clone();

    let (removed, rewrite) = remove_empty_blocks_mapped(&mut cfg);

    assert_eq!(removed, 2);
    assert_eq!(live_blocks(&cfg), [entry, target].into());
    assert_eq!(cfg.edge(into_first).source(), entry);
    assert_eq!(cfg.edge(into_first).target(), target);
    assert_eq!(rewrite.edges(into_first), Some([into_first].as_slice()));
    assert_eq!(rewrite.edges(between), Some([].as_slice()));
    assert_eq!(rewrite.edges(out_of_second), Some([].as_slice()));
    assert_eq!(rewrite.blocks(first), Some([].as_slice()));
    assert_eq!(rewrite.blocks(second), Some([].as_slice()));
    assert_surviving_edges_unchanged(&before, &cfg, "bypass chain");
    assert_valid(&cfg, "bypass chain");
}

#[test]
fn remove_empty_blocks_bypasses_a_chain_of_empty_blocks_keeping_edge_payloads() {
    bypass_chain_case::<()>();
    bypass_chain_case::<Route>();
}

#[test]
fn remove_empty_blocks_keeps_empty_blocks_that_decide_or_never_leave() {
    let mut builder = Builder::<Route>::new();
    let entry = builder.entry();
    let conditional_exit = builder.block([]);
    let two_ways = builder.block([]);
    let spin = builder.block([]);
    let left = builder.block([Inst::print(2)]);
    let right = builder.block([Inst::print(3)]);
    builder.edge(entry, conditional_exit, EdgeKind::ConditionalTrue);
    builder.edge(conditional_exit, two_ways, EdgeKind::ConditionalTrue);
    builder.edge(two_ways, left, EdgeKind::ConditionalTrue);
    builder.edge(two_ways, right, EdgeKind::ConditionalFalse);
    builder.edge(left, spin, EdgeKind::Fallthrough);
    builder.edge(right, spin, EdgeKind::Fallthrough);
    builder.edge(spin, spin, EdgeKind::Fallthrough);
    let mut cfg = builder.finish();
    let before = cfg.clone();

    assert_eq!(remove_empty_blocks(&mut cfg), 0);
    assert_eq!(
        cfg, before,
        "the entry, a conditional exit, a two-way block and a self-loop stay"
    );
}

fn unreachable_case<E: Payload>() {
    let mut builder = Builder::<E>::new();
    let entry = builder.entry();
    builder.push(entry, Inst::print(0));
    let live = builder.block([Inst::ret()]);
    let first = builder.block([Inst::print(2)]);
    let second = builder.block([Inst::print(3)]);
    builder.edge(entry, live, EdgeKind::Fallthrough);
    let first_edge = builder.edge(first, second, EdgeKind::Fallthrough);
    let feeding = builder.edge(second, live, EdgeKind::Fallthrough);
    let mut cfg = builder.finish();
    let bound = cfg.block_bound();

    let (removed, rewrite) = remove_unreachable_mapped(&mut cfg);

    assert_eq!(removed, 2);
    assert_eq!(live_blocks(&cfg), reachable_blocks(&cfg));
    assert_eq!(
        cfg.incoming(live).count(),
        1,
        "the dead feeder's edge is gone"
    );
    assert_eq!(rewrite.blocks(first), Some([].as_slice()));
    assert_eq!(rewrite.blocks(second), Some([].as_slice()));
    assert_eq!(rewrite.edges(first_edge), Some([].as_slice()));
    assert_eq!(rewrite.edges(feeding), Some([].as_slice()));
    assert_eq!(
        cfg.block_bound(),
        bound,
        "slots stay reserved until compaction"
    );
    assert_eq!(
        cfg.block(first).instructions(),
        [Inst::print(2)],
        "a removed block stays readable until compaction"
    );
    assert_eq!(remove_unreachable(&mut cfg), 0);
}

#[test]
fn remove_unreachable_drops_exactly_the_dead_blocks_and_the_edges_they_fed() {
    unreachable_case::<()>();
    unreachable_case::<Route>();
}

fn contract_case<E: Payload>() {
    let mut builder = Builder::<E>::new();
    let entry = builder.entry();
    builder.push(entry, Inst::constant(2, 1));
    let middle = builder.block([Inst::print(2)]);
    let exit = builder.block([Inst::ret()]);
    let connecting = builder.edge(entry, middle, EdgeKind::Fallthrough);
    let leaving = builder.edge(middle, exit, EdgeKind::Jump);
    let mut cfg = builder.finish();
    cfg.block_mut(middle).set_label("body");
    let before = cfg.clone();

    let rewrite = contract_edge_mapped(&mut cfg, entry, middle).expect("a sole edge contracts");

    assert_eq!(
        cfg.block(entry).instructions(),
        [Inst::constant(2, 1), Inst::print(2)]
    );
    assert_eq!(cfg.block(entry).label(), Some("body"));
    assert!(!cfg.contains_block(middle));
    assert!(!cfg.contains_edge(connecting));
    assert_eq!(cfg.edge(leaving).source(), entry);
    assert_eq!(cfg.edge(leaving).kind(), EdgeKind::Jump);
    assert_eq!(rewrite.blocks(middle), Some([entry].as_slice()));
    assert_eq!(rewrite.edges(connecting), Some([].as_slice()));
    assert_eq!(rewrite.edges(leaving), Some([leaving].as_slice()));
    assert_surviving_edges_unchanged(&before, &cfg, "contract");
    assert_valid(&cfg, "contract");
}

#[test]
fn contract_edge_folds_target_into_source_with_label_and_outgoing_edges() {
    contract_case::<()>();
    contract_case::<Route>();
}

#[test]
fn contract_edge_keeps_the_source_label_when_both_blocks_have_one() {
    let (mut cfg, blocks) = chain::<()>(2);
    cfg.block_mut(blocks[0]).set_label("head");
    cfg.block_mut(blocks[1]).set_label("tail");

    assert!(contract_edge(&mut cfg, blocks[0], blocks[1]));
    assert_eq!(cfg.block(blocks[0]).label(), Some("head"));
}

#[test]
fn contract_edge_refuses_everything_but_a_sole_edge_between_distinct_blocks() {
    let mut builder = Builder::<Route>::new();
    let entry = builder.entry();
    let branch = builder.block([Inst::branch(2)]);
    let left = builder.block([Inst::print(2)]);
    let join = builder.block([Inst::print(3)]);
    let spin = builder.block([]);
    builder.edge(entry, branch, EdgeKind::Fallthrough);
    builder.edge(branch, left, EdgeKind::ConditionalTrue);
    builder.edge(branch, join, EdgeKind::ConditionalFalse);
    builder.edge(left, join, EdgeKind::Fallthrough);
    builder.edge(join, entry, EdgeKind::Back);
    builder.edge(join, spin, EdgeKind::Fallthrough);
    builder.edge(spin, spin, EdgeKind::Fallthrough);
    let mut cfg = builder.finish();
    let before = cfg.clone();

    assert!(
        !contract_edge(&mut cfg, branch, left),
        "source has two exits"
    );
    assert!(
        !contract_edge(&mut cfg, left, join),
        "target has two entries"
    );
    assert!(
        !contract_edge(&mut cfg, join, entry),
        "the entry is never a target"
    );
    assert!(
        !contract_edge(&mut cfg, spin, spin),
        "self-loops do not contract"
    );
    assert!(
        !contract_edge(&mut cfg, entry, join),
        "no edge runs entry to join"
    );
    assert!(contract_edge_mapped(&mut cfg, left, spin).is_none());
    assert_eq!(cfg, before);
}

fn block_with_exits<E: Payload>() -> (Cfg<Inst, E>, cfglib::BlockId, [cfglib::EdgeId; 2]) {
    let mut builder = Builder::<E>::new();
    let entry = builder.entry();
    let body: Vec<Inst> = (0..5).map(|value| Inst::constant(2, value)).collect();
    let block = builder.block(body);
    let left = builder.block([Inst::print(2)]);
    let right = builder.block([Inst::print(3)]);
    builder.edge(entry, block, EdgeKind::Fallthrough);
    let exits = [
        builder.edge(block, left, EdgeKind::ConditionalTrue),
        builder.edge(block, right, EdgeKind::ConditionalFalse),
    ];
    (builder.finish(), block, exits)
}

fn split_points_case<E: Payload>() {
    let (mut cfg, block, exits) = block_with_exits::<E>();
    let original: Vec<Inst> = cfg.block(block).instructions().to_vec();
    let (parts, rewrite) = split_node_at_points(
        &mut cfg,
        block,
        [
            (0, E::tagged(100)),
            (2, E::tagged(101)),
            (5, E::tagged(102)),
        ],
    )
    .expect("strictly increasing in-bounds points");

    assert_eq!(parts.len(), 4);
    assert_eq!(parts[0], block);
    let pieces: Vec<&[Inst]> = parts
        .iter()
        .map(|&part| cfg.block(part).instructions())
        .collect();
    assert_eq!(
        pieces[0],
        &original[..0],
        "a point at 0 leaves the head empty"
    );
    assert_eq!(pieces[1], &original[..2]);
    assert_eq!(pieces[2], &original[2..]);
    assert_eq!(
        pieces[3],
        &original[5..],
        "a point at the end leaves the tail empty"
    );
    let flattened: Vec<Inst> = pieces.concat();
    assert_eq!(flattened, original);

    for exit in exits {
        assert_eq!(
            cfg.edge(exit).source(),
            parts[3],
            "exits move to the last part"
        );
    }
    for (index, pair) in parts.windows(2).enumerate() {
        let joining: Vec<_> = cfg
            .outgoing(pair[0])
            .filter(|&edge| cfg.edge(edge).target() == pair[1])
            .collect();
        assert_eq!(joining.len(), 1, "one fallthrough joins consecutive parts");
        assert_eq!(cfg.edge(joining[0]).kind(), EdgeKind::Fallthrough);
        assert_eq!(
            cfg.edge(joining[0]).payload(),
            &E::tagged(100 + u32::try_from(index).unwrap())
        );
    }
    assert_eq!(rewrite.blocks(block), Some(parts.as_slice()));
    assert_eq!(rewrite.created_blocks(), &parts[1..]);
    assert_eq!(rewrite.created_edges().len(), 3);
    for exit in exits {
        assert_eq!(rewrite.edges(exit), Some([exit].as_slice()));
    }
    assert_valid(&cfg, "split at points");
}

#[test]
fn split_node_at_points_partitions_a_block_and_hands_its_exits_to_the_last_part() {
    split_points_case::<()>();
    split_points_case::<Route>();
}

#[test]
fn split_node_at_points_rejects_bad_points_before_touching_the_graph() {
    let (mut cfg, block, _) = block_with_exits::<Route>();
    let before = cfg.clone();
    let points = |points: &[usize]| {
        points
            .iter()
            .map(|&point| (point, Route::Synthetic))
            .collect::<Vec<_>>()
    };

    assert_eq!(
        split_node_at_points(&mut cfg, block, points(&[1, 6])).unwrap_err(),
        SplitPointError::OutOfBounds {
            point: 6,
            instruction_count: 5
        }
    );
    assert_eq!(
        split_node_at_points(&mut cfg, block, points(&[3, 3])).unwrap_err(),
        SplitPointError::NotStrictlyIncreasing {
            previous: 3,
            point: 3
        }
    );
    assert_eq!(
        split_node_at_points(&mut cfg, block, points(&[4, 2])).unwrap_err(),
        SplitPointError::NotStrictlyIncreasing {
            previous: 4,
            point: 2
        }
    );
    assert_eq!(cfg, before);
}

fn split_everywhere_case<E: Payload>(seed: u64) {
    let before = random_program::<E>(seed, false);
    for block in before.block_ids() {
        for at in 0..=before.block(block).instructions().len() {
            let mut cfg = before.clone();
            let tail = split_node(&mut cfg, block, at);
            let context = format!("seed {seed}, {block} at {at}");
            assert_valid(&cfg, &context);
            assert_same_behavior(&before, &cfg, &context);
            let fallthrough: Vec<_> = cfg
                .outgoing(block)
                .filter(|&edge| cfg.edge(edge).target() == tail)
                .collect();
            assert_eq!(fallthrough.len(), 1, "{context}");
            assert_eq!(
                cfg.edge(fallthrough[0]).payload(),
                &E::default(),
                "{context}"
            );
        }
    }
}

#[test]
fn split_node_preserves_behavior_at_every_instruction_boundary() {
    for seed in 0..12 {
        split_everywhere_case::<()>(seed);
        split_everywhere_case::<Route>(seed);
    }
}

#[test]
fn split_node_with_payload_mapped_reports_the_new_tail_and_edge() {
    let (mut cfg, block, exits) = block_with_exits::<Route>();
    let (tail, rewrite) = split_node_with_payload_mapped(&mut cfg, block, 3, Route::Tagged(77));

    assert_eq!(rewrite.blocks(block), Some([block, tail].as_slice()));
    assert_eq!(rewrite.created_blocks(), [tail]);
    let [created] = rewrite.created_edges() else {
        panic!(
            "one fallthrough edge is created: {:?}",
            rewrite.created_edges()
        );
    };
    assert_eq!(cfg.edge(*created).payload(), &Route::Tagged(77));
    assert_eq!(cfg.edge(*created).target(), tail);
    assert_eq!(cfg.block(block).instructions().len(), 3);
    assert_eq!(cfg.block(tail).instructions().len(), 2);
    for exit in exits {
        assert_eq!(cfg.edge(exit).source(), tail);
    }
}
