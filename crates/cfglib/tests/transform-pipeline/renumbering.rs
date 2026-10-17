//! Identity bookkeeping across passes: composing `Rewrite` records, and
//! carrying them over compactions with `then_renumbered` and `Renumbering`.

use cfglib::{
    BlockId, Cfg, CfgRenumbering, EdgeId, Rewrite, merge_blocks_mapped, remove_empty_blocks_mapped,
    remove_unreachable_mapped, simplify_mapped, split_critical_edges_mapped,
};

use crate::oracles::{
    Placement, assert_images_hold_original_instructions, assert_rewrite_accounts_for_identities,
    assert_surviving_payloads_unchanged, assert_valid, live_blocks,
};
use crate::programs::random_program;
use crate::toy::{Builder, Inst, Payload, Route, assert_same_behavior};

const SEEDS: u64 = 60;

type Pass<E> = fn(&mut Cfg<Inst, E>) -> Rewrite;

/// The structural passes of a cleanup-then-split pipeline.
fn pipeline<E: Payload>() -> [Pass<E>; 4] {
    [
        |cfg| remove_unreachable_mapped(cfg).1,
        |cfg| remove_empty_blocks_mapped(cfg).1,
        |cfg| merge_blocks_mapped(cfg).1,
        |cfg| split_critical_edges_mapped(cfg).1,
    ]
}

/// Runs every pass in order, one record each.
fn run_passes<E: Payload>(cfg: &mut Cfg<Inst, E>) -> Vec<Rewrite> {
    pipeline::<E>().iter().map(|pass| pass(cfg)).collect()
}

fn composed(records: &[Rewrite]) -> Rewrite {
    let mut all = Rewrite::new();
    for record in records {
        all.compose(record.clone());
    }
    all
}

fn composition_case<E: Payload>(seed: u64) {
    let before = random_program::<E>(seed, true);
    let mut cfg = before.clone();
    let records = run_passes(&mut cfg);
    let all = composed(&records);
    let context = format!("seed {seed}");

    let placement = Placement {
        rewrite: &all,
        renumbering: None,
    };
    assert_rewrite_accounts_for_identities(&before, &cfg, &placement, &context);
    assert_images_hold_original_instructions(&before, &cfg, &placement, &context);
    assert_surviving_payloads_unchanged(&before, &cfg, &context);
    assert_valid(&cfg, &context);
    assert_same_behavior(&before, &cfg, &context);

    let mut left = records[0].clone();
    left.compose(records[1].clone());
    left.compose(records[2].clone());
    let mut tail = records[1].clone();
    tail.compose(records[2].clone());
    let mut right = records[0].clone();
    right.compose(tail);
    assert_eq!(
        left, right,
        "{context}: composition does not depend on grouping"
    );

    for record in &records {
        let mut after_empty = Rewrite::new();
        after_empty.compose(record.clone());
        assert_eq!(
            &after_empty, record,
            "{context}: an empty record composes as the identity"
        );
        let mut before_empty = record.clone();
        before_empty.compose(Rewrite::new());
        assert_eq!(&before_empty, record, "{context}");
    }
}

#[test]
fn composed_records_map_original_identities_straight_to_the_final_graph() {
    for seed in 0..SEEDS {
        composition_case::<()>(seed);
        composition_case::<Route>(seed);
    }
}

#[test]
fn simplify_reports_the_same_map_as_composing_its_passes_by_hand() {
    for seed in 0..SEEDS {
        let before = random_program::<()>(seed, true);
        let mut by_simplify = before.clone();
        let (_, from_simplify) = simplify_mapped(&mut by_simplify);

        let mut by_hand = before.clone();
        let mut record = Rewrite::new();
        loop {
            let mut round = Rewrite::new();
            let (unreachable, step) = remove_unreachable_mapped(&mut by_hand);
            round.compose(step);
            let (empty, step) = remove_empty_blocks_mapped(&mut by_hand);
            round.compose(step);
            let (merged, step) = merge_blocks_mapped(&mut by_hand);
            round.compose(step);
            record.compose(round);
            if unreachable + empty + merged == 0 {
                break;
            }
        }
        assert_eq!(by_simplify, by_hand, "seed {seed}");
        assert_eq!(from_simplify, record, "seed {seed}");
    }
}

/// Follows one old block through a sequence of (rewrite, compaction) stages,
/// the way a consumer holding identities from before the pipeline must.
fn follow(stages: &[(Rewrite, CfgRenumbering)], old: BlockId) -> Vec<BlockId> {
    let mut current = vec![old];
    for (rewrite, renumbering) in stages {
        let mut next = Vec::new();
        for block in current {
            let images = match rewrite.blocks(block) {
                Some(images) => images.to_vec(),
                None => vec![block],
            };
            for image in images {
                if let Some(renumbered) = renumbering.node(image)
                    && !next.contains(&renumbered)
                {
                    next.push(renumbered);
                }
            }
        }
        current = next;
    }
    current
}

#[test]
fn a_compaction_between_every_pass_keeps_the_identity_trail_intact() {
    for seed in 0..SEEDS {
        let before = random_program::<Route>(seed, true);
        let mut cfg = before.clone();
        let mut stages = Vec::new();
        for _ in 0..2 {
            for pass in pipeline::<Route>() {
                let record = pass(&mut cfg);
                let renumbering = cfg.compact();
                stages.push((record, renumbering));
            }
        }
        let context = format!("seed {seed}");
        assert!(cfg.is_compact(), "{context}");

        for old in before.block_ids() {
            let images = follow(&stages, old);
            assert!(
                images.iter().all(|&image| cfg.contains_block(image)),
                "{context}"
            );
            let original = before.block(old).instructions();
            if images.is_empty() || original.is_empty() {
                continue;
            }
            let held: Vec<Inst> = images
                .iter()
                .flat_map(|&image| cfg.block(image).instructions().iter().cloned())
                .collect();
            assert!(
                held.windows(original.len())
                    .any(|window| window == original),
                "{context}: instructions of {old} are lost; it follows to {images:?}"
            );
        }
        assert_valid(&cfg, &context);
        assert_same_behavior(&before, &cfg, &context);
    }
}

#[test]
fn then_renumbered_translates_every_replacement_and_creation_to_the_compacted_ids() {
    for seed in 0..SEEDS {
        let before = random_program::<Route>(seed, true);
        let mut cfg = before.clone();
        let records = run_passes(&mut cfg);
        let all = composed(&records);
        let renumbering = cfg.compact();
        let carried = all.then_renumbered(&renumbering);
        let context = format!("seed {seed}");

        for (old, images) in all.block_mappings() {
            let expected: Vec<BlockId> = images
                .iter()
                .filter_map(|&image| renumbering.node(image))
                .collect();
            assert_eq!(
                carried.blocks(old),
                Some(expected.as_slice()),
                "{context}: {old}"
            );
        }
        for (old, images) in all.edge_mappings() {
            let expected: Vec<_> = images
                .iter()
                .filter_map(|&image| renumbering.edge(image))
                .collect();
            assert_eq!(
                carried.edges(old),
                Some(expected.as_slice()),
                "{context}: {old}"
            );
        }
        assert!(
            carried
                .created_blocks()
                .iter()
                .all(|&block| cfg.contains_block(block)),
            "{context}"
        );
        assert!(
            carried
                .created_edges()
                .iter()
                .all(|&edge| cfg.contains_edge(edge)),
            "{context}"
        );
        assert_eq!(
            carried.created_blocks().len(),
            all.created_blocks()
                .iter()
                .filter(|&&block| renumbering.node(block).is_some())
                .count(),
            "{context}: a created block that was removed again is no longer created"
        );
    }
}

#[test]
fn compaction_renumbers_survivors_densely_in_order_and_reports_the_rest_as_gone() {
    for seed in 0..SEEDS {
        let mut cfg = random_program::<Route>(seed, true);
        simplify_mapped(&mut cfg);
        let removed_something = !cfg.is_compact();
        let before = cfg.clone();
        let survivors: Vec<BlockId> = live_blocks(&before).into_iter().collect();
        let renumbering = cfg.compact();
        let context = format!("seed {seed}");

        assert!(cfg.is_compact(), "{context}");
        assert_eq!(renumbering.is_identity(), !removed_something, "{context}");
        assert_eq!(renumbering.node_bound(), before.block_bound(), "{context}");
        assert_eq!(renumbering.edge_bound(), before.edge_bound(), "{context}");
        for (rank, &old) in survivors.iter().enumerate() {
            let new = renumbering.node(old).expect("a live block survives");
            assert_eq!(
                new.index(),
                rank,
                "{context}: survivors keep their relative order"
            );
            assert_eq!(
                cfg.block(new).instructions(),
                before.block(old).instructions(),
                "{context}"
            );
            assert_eq!(
                cfg.block(new).label(),
                before.block(old).label(),
                "{context}"
            );
        }
        for old in (0..before.block_bound()).map(BlockId::from_index) {
            assert_eq!(
                renumbering.node(old).is_some(),
                before.contains_block(old),
                "{context}"
            );
        }
        assert_eq!(
            renumbering.node(before.entry()),
            Some(cfg.entry()),
            "{context}"
        );
        for old in before.edge_ids() {
            let new = renumbering.edge(old).expect("a live edge survives");
            let (was, is) = (before.edge(old), cfg.edge(new));
            assert_eq!(
                renumbering.node(was.source()),
                Some(is.source()),
                "{context}"
            );
            assert_eq!(
                renumbering.node(was.target()),
                Some(is.target()),
                "{context}"
            );
            assert_eq!(was.kind(), is.kind(), "{context}");
            assert_eq!(was.payload(), is.payload(), "{context}");
        }
        assert!(
            cfg.compact().is_identity(),
            "{context}: a compact graph compacts to itself"
        );
    }
}

#[test]
fn composing_two_renumberings_equals_following_them_one_after_the_other() {
    for seed in 0..SEEDS {
        let mut cfg = random_program::<()>(seed, true);
        remove_unreachable_mapped(&mut cfg);
        let first = cfg.compact();
        remove_empty_blocks_mapped(&mut cfg);
        merge_blocks_mapped(&mut cfg);
        let second = cfg.compact();
        let both = first.compose(&second);

        assert_eq!(both.node_bound(), first.node_bound(), "seed {seed}");
        assert_eq!(both.edge_bound(), first.edge_bound(), "seed {seed}");
        for index in 0..first.node_bound() {
            let old = BlockId::from_index(index);
            assert_eq!(
                both.node(old),
                first.node(old).and_then(|mid| second.node(mid)),
                "seed {seed}: {old}"
            );
        }
        for index in 0..first.edge_bound() {
            let old = EdgeId::from_index(index);
            assert_eq!(
                both.edge(old),
                first.edge(old).and_then(|mid| second.edge(mid)),
                "seed {seed}: {old}"
            );
        }
    }
}

#[test]
fn the_entry_follows_its_block_when_earlier_slots_are_compacted_away() {
    let mut builder = Builder::<()>::new();
    let first = builder.entry();
    let dropped = builder.block([Inst::print(0)]);
    let entry = builder.block([Inst::print(2), Inst::ret()]);
    let mut cfg = builder.finish();
    cfg.set_entry(entry);
    cfg.remove_block(first);
    cfg.remove_block(dropped);

    let renumbering = cfg.compact();

    assert_eq!(cfg.entry().index(), 0);
    assert_eq!(renumbering.node(entry), Some(cfg.entry()));
    assert_eq!(renumbering.node(first), None);
    assert_eq!(
        cfg.block(cfg.entry()).instructions(),
        [Inst::print(2), Inst::ret()]
    );
    assert_eq!(cfg.block_count(), 1);
}
