//! Purity verdicts against hand-labelled effects.

use std::collections::BTreeSet;

use cfglib::{BlockId, Cfg, EdgeKind, Purity, block_purities, block_purity, cfg_purity};

use crate::random::Rng;
use crate::stmt::{Effect, Route, Stmt, block, link};

struct Labelled {
    cfg: Cfg<Stmt, Route>,
    pure_entry: BlockId,
    stores: BlockId,
    network: BlockId,
    empty: BlockId,
}

/// Four blocks in a chain with known effect sets: none, {Memory, Output}
/// (Memory declared twice), {Output, Network}, and an empty block.
fn labelled() -> Labelled {
    let mut cfg = Cfg::<Stmt, Route>::with_edge_payload();
    let pure_entry = cfg.entry();
    for statement in [
        Stmt::add("x", "a", "b"),
        Stmt::multiply("y", "x", "x"),
        Stmt::constant("z", 3),
    ] {
        cfg.block_mut(pure_entry).push(statement);
    }
    let stores = block(
        &mut cfg,
        [
            Stmt::effect(&[Effect::Memory], &["x"]),
            Stmt::effect(&[Effect::Output], &["y"]),
            Stmt::effect(&[Effect::Memory], &["x"]),
        ],
    );
    let network = block(
        &mut cfg,
        [
            Stmt::call("send", &["r"], &[Effect::Network, Effect::Output]),
            Stmt::add("w", "r", "r"),
        ],
    );
    let empty = block(&mut cfg, []);
    link(&mut cfg, pure_entry, stores, EdgeKind::Fallthrough);
    link(&mut cfg, stores, network, EdgeKind::Fallthrough);
    link(&mut cfg, network, empty, EdgeKind::Fallthrough);
    Labelled {
        cfg,
        pure_entry,
        stores,
        network,
        empty,
    }
}

#[test]
fn a_block_is_pure_exactly_when_nothing_in_it_declares_an_effect() {
    let Labelled {
        cfg,
        pure_entry,
        stores,
        network,
        empty,
    } = labelled();

    assert_eq!(block_purity(&cfg, pure_entry), Purity::Pure);
    assert_eq!(
        block_purity(&cfg, empty),
        Purity::Pure,
        "an empty block does nothing"
    );
    assert!(block_purity(&cfg, pure_entry).is_pure());
    assert!(!block_purity(&cfg, pure_entry).is_impure());
    assert!(block_purity(&cfg, stores).is_impure());
    assert!(!block_purity(&cfg, stores).is_pure());
    let _ = network;
}

#[test]
fn impure_blocks_report_each_effect_once_in_the_vocabulary_order() {
    let Labelled {
        cfg,
        stores,
        network,
        ..
    } = labelled();

    assert_eq!(
        block_purity(&cfg, stores),
        Purity::Impure(vec![Effect::Memory, Effect::Output]),
        "Memory is declared twice and reported once"
    );
    assert_eq!(
        block_purity(&cfg, network),
        Purity::Impure(vec![Effect::Output, Effect::Network]),
        "declaration order (Network, Output) does not matter; the order is sorted"
    );
}

#[test]
fn the_graph_is_impure_with_the_union_of_every_blocks_effects() {
    let Labelled { cfg, .. } = labelled();

    assert_eq!(
        cfg_purity(&cfg),
        Purity::Impure(vec![Effect::Memory, Effect::Output, Effect::Network])
    );
}

#[test]
fn removing_the_impure_blocks_makes_the_graph_pure_again() {
    let Labelled {
        mut cfg,
        stores,
        network,
        ..
    } = labelled();

    cfg.remove_block(network);
    assert_eq!(
        cfg_purity(&cfg),
        Purity::Impure(vec![Effect::Memory, Effect::Output]),
        "a removed block's effects no longer count"
    );
    cfg.remove_block(stores);
    assert_eq!(cfg_purity(&cfg), Purity::Pure);
}

#[test]
fn per_block_verdicts_come_in_allocation_order_and_skip_removed_blocks() {
    let Labelled {
        mut cfg,
        pure_entry,
        stores,
        network,
        empty,
    } = labelled();

    assert_eq!(
        block_purities(&cfg),
        [
            (pure_entry, Purity::Pure),
            (stores, Purity::Impure(vec![Effect::Memory, Effect::Output])),
            (
                network,
                Purity::Impure(vec![Effect::Output, Effect::Network])
            ),
            (empty, Purity::Pure),
        ]
    );

    cfg.remove_block(stores);
    let remaining: Vec<BlockId> = block_purities(&cfg)
        .into_iter()
        .map(|(block, _)| block)
        .collect();
    assert_eq!(remaining, [pure_entry, network, empty]);
}

#[test]
fn the_graph_verdict_is_the_union_of_the_block_verdicts_on_random_assignments() {
    let vocabulary = [Effect::Memory, Effect::Output, Effect::Network];
    for seed in 0..60 {
        let mut rng = Rng::new(seed);
        let mut cfg = Cfg::<Stmt>::new();
        let mut expected_per_block: Vec<BTreeSet<Effect>> = Vec::new();
        let mut previous = cfg.entry();
        for index in 0..8 {
            let current = if index == 0 {
                previous
            } else {
                block(&mut cfg, [])
            };
            if index > 0 {
                cfg.add_edge(previous, current, EdgeKind::Fallthrough);
            }
            let mut declared = BTreeSet::new();
            for _ in 0..rng.below(4) {
                let effect = vocabulary[rng.below(3)];
                declared.insert(effect);
                cfg.block_mut(current).push(Stmt::effect(&[effect], &[]));
                cfg.block_mut(current).push(Stmt::add("t", "a", "b"));
            }
            expected_per_block.push(declared);
            previous = current;
        }

        let verdicts = block_purities(&cfg);
        for ((_, verdict), declared) in verdicts.iter().zip(&expected_per_block) {
            match verdict {
                Purity::Pure => assert!(declared.is_empty(), "seed {seed}"),
                Purity::Impure(effects) => {
                    assert_eq!(
                        effects.iter().copied().collect::<BTreeSet<_>>(),
                        *declared,
                        "seed {seed}"
                    );
                    assert!(
                        effects.windows(2).all(|pair| pair[0] < pair[1]),
                        "seed {seed}: sorted, no repeats"
                    );
                }
            }
        }
        let union: BTreeSet<Effect> = expected_per_block.iter().flatten().copied().collect();
        match cfg_purity(&cfg) {
            Purity::Pure => assert!(union.is_empty(), "seed {seed}"),
            Purity::Impure(effects) => {
                assert_eq!(
                    effects.into_iter().collect::<BTreeSet<_>>(),
                    union,
                    "seed {seed}"
                );
            }
        }
    }
}
