//! Edge probabilities and the block frequencies they imply.

use std::collections::BTreeMap;

use cfglib::{BlockId, Cfg, CfgProfile, EdgeId, EdgeKind, set_uniform_edge_weights};

use crate::random::Rng;
use crate::stmt::Stmt;

const EPSILON: f64 = 1e-6;

fn close(left: f64, right: f64) -> bool {
    (left - right).abs() < EPSILON
}

fn assert_weight(profile: &CfgProfile, block: BlockId, expected: f64) {
    let actual = profile
        .block_weights
        .get(&block)
        .unwrap_or_else(|| panic!("{block} has no frequency"));
    assert!(close(*actual, expected), "{block}: {actual} != {expected}");
}

struct Diamond {
    cfg: Cfg<Stmt>,
    entry: BlockId,
    arms: [BlockId; 2],
    merge: BlockId,
    exit: BlockId,
}

fn diamond() -> Diamond {
    let mut cfg = Cfg::<Stmt>::new();
    let entry = cfg.entry();
    let (left, right, merge, exit) = (
        cfg.new_block(),
        cfg.new_block(),
        cfg.new_block(),
        cfg.new_block(),
    );
    cfg.add_edge(entry, left, EdgeKind::ConditionalTrue);
    cfg.add_edge(entry, right, EdgeKind::ConditionalFalse);
    cfg.add_edge(left, merge, EdgeKind::Fallthrough);
    cfg.add_edge(right, merge, EdgeKind::Fallthrough);
    cfg.add_edge(merge, exit, EdgeKind::Fallthrough);
    Diamond {
        cfg,
        entry,
        arms: [left, right],
        merge,
        exit,
    }
}

#[test]
fn uniform_weights_split_each_blocks_probability_equally_among_its_exits() {
    let Diamond { mut cfg, entry, .. } = diamond();
    let parallel_target = cfg.new_block();
    cfg.add_edge(entry, parallel_target, EdgeKind::SwitchCase);
    cfg.add_edge(entry, parallel_target, EdgeKind::SwitchCase);
    set_uniform_edge_weights(&mut cfg);

    for block in cfg.block_ids() {
        let weights: Vec<f64> = cfg
            .outgoing(block)
            .map(|edge| {
                cfg.edge(edge)
                    .weight()
                    .expect("every edge of a non-exit block is weighted")
            })
            .collect();
        if weights.is_empty() {
            continue;
        }
        let share = 1.0 / f64::from(u32::try_from(weights.len()).unwrap());
        assert!(
            weights.iter().all(|&weight| close(weight, share)),
            "{block}: {weights:?}"
        );
        assert!(
            close(weights.iter().sum(), 1.0),
            "{block}: probabilities sum to one"
        );
    }
    assert_eq!(cfg.outgoing(entry).count(), 4);
}

#[test]
fn uniform_weights_leave_removed_blocks_and_exit_blocks_alone() {
    let Diamond {
        mut cfg,
        arms,
        exit,
        ..
    } = diamond();
    cfg.remove_block(arms[1]);
    set_uniform_edge_weights(&mut cfg);

    let entry_edges: Vec<EdgeId> = cfg.outgoing(cfg.entry()).collect();
    assert_eq!(entry_edges.len(), 1);
    assert!(
        close(cfg.edge(entry_edges[0]).weight().unwrap(), 1.0),
        "one exit takes all of it"
    );
    assert_eq!(cfg.outgoing(exit).count(), 0);
}

#[test]
fn uniform_probabilities_propagate_into_one_visit_per_entry_at_the_merge() {
    let Diamond {
        mut cfg,
        entry,
        arms,
        merge,
        exit,
    } = diamond();
    set_uniform_edge_weights(&mut cfg);

    let profile = CfgProfile::from_edge_weights(&cfg);

    assert_weight(&profile, entry, 1.0);
    assert_weight(&profile, arms[0], 0.5);
    assert_weight(&profile, arms[1], 0.5);
    assert_weight(&profile, merge, 1.0);
    assert_weight(&profile, exit, 1.0);
    assert_eq!(profile.edge_weights.len(), cfg.edge_count());
    let (hottest, weight) = profile.hottest_block().unwrap();
    assert!(close(weight, 1.0) && [entry, merge, exit].contains(&hottest));
}

struct Loop {
    cfg: Cfg<Stmt>,
    entry: BlockId,
    header: BlockId,
    body: BlockId,
    exit: BlockId,
}

/// A loop that continues with probability 0.9: it runs `1 / (1 - 0.9)` = 10
/// header tests per entry, 9 of them entering the body.
fn weighted_loop() -> Loop {
    let mut cfg = Cfg::<Stmt>::new();
    let entry = cfg.entry();
    let (header, body, exit) = (cfg.new_block(), cfg.new_block(), cfg.new_block());
    for (source, target, kind, probability) in [
        (entry, header, EdgeKind::Fallthrough, 1.0),
        (header, body, EdgeKind::ConditionalTrue, 0.9),
        (header, exit, EdgeKind::ConditionalFalse, 0.1),
        (body, header, EdgeKind::Back, 1.0),
    ] {
        let edge = cfg.add_edge(source, target, kind);
        cfg.edge_mut(edge).set_weight(Some(probability));
    }
    Loop {
        cfg,
        entry,
        header,
        body,
        exit,
    }
}

#[test]
fn a_loop_back_edge_multiplies_the_header_frequency_by_the_expected_trip_count() {
    let Loop {
        cfg,
        entry,
        header,
        body,
        exit,
    } = weighted_loop();

    let profile = CfgProfile::from_edge_weights(&cfg);

    assert_weight(&profile, entry, 1.0);
    assert_weight(&profile, header, 10.0);
    assert_weight(&profile, body, 9.0);
    assert_weight(&profile, exit, 1.0);
    assert_eq!(
        profile.hottest_block().map(|(block, _)| block),
        Some(header)
    );
}

#[test]
fn hot_blocks_are_those_at_or_above_the_threshold_in_identity_order() {
    let Loop {
        cfg, header, body, ..
    } = weighted_loop();
    let profile = CfgProfile::from_edge_weights(&cfg);

    assert_eq!(profile.hot_blocks(8.5), [header, body]);
    assert_eq!(profile.hot_blocks(9.5), [header]);
    assert_eq!(profile.hot_blocks(10.5), []);
    assert_eq!(profile.hot_blocks(0.0).len(), 4);
}

#[test]
fn the_coldest_block_is_the_least_visited_branch_arm() {
    let mut cfg = Cfg::<Stmt>::new();
    let entry = cfg.entry();
    let (main, cold, exit) = (cfg.new_block(), cfg.new_block(), cfg.new_block());
    for (source, target, kind, probability) in [
        (entry, main, EdgeKind::ConditionalTrue, 0.7),
        (entry, cold, EdgeKind::ConditionalFalse, 0.3),
        (main, exit, EdgeKind::Fallthrough, 1.0),
        (cold, exit, EdgeKind::Fallthrough, 1.0),
    ] {
        let edge = cfg.add_edge(source, target, kind);
        cfg.edge_mut(edge).set_weight(Some(probability));
    }

    let profile = CfgProfile::from_edge_weights(&cfg);
    let (coldest, weight) = profile.coldest_block().unwrap();

    assert_eq!(coldest, cold);
    assert!(close(weight, 0.3));
    assert_weight(&profile, main, 0.7);
    assert_weight(&profile, exit, 1.0);
}

#[test]
fn unweighted_edges_contribute_nothing_and_unreached_blocks_have_no_frequency() {
    let mut cfg = Cfg::<Stmt>::new();
    let entry = cfg.entry();
    let (weighted, unweighted, orphan) = (cfg.new_block(), cfg.new_block(), cfg.new_block());
    let counted = cfg.add_edge(entry, weighted, EdgeKind::ConditionalTrue);
    cfg.edge_mut(counted).set_weight(Some(0.6));
    let silent = cfg.add_edge(entry, unweighted, EdgeKind::ConditionalFalse);

    let profile = CfgProfile::from_edge_weights(&cfg);

    assert_eq!(
        profile.edge_weights.keys().copied().collect::<Vec<_>>(),
        [counted]
    );
    assert!(!profile.edge_weights.contains_key(&silent));
    assert_weight(&profile, entry, 1.0);
    assert_weight(&profile, weighted, 0.6);
    assert!(!profile.block_weights.contains_key(&unweighted));
    assert!(!profile.block_weights.contains_key(&orphan));
}

#[test]
fn a_lone_entry_block_is_visited_once() {
    let cfg = Cfg::<Stmt>::new();
    let profile = CfgProfile::from_edge_weights(&cfg);

    assert_weight(&profile, cfg.entry(), 1.0);
    assert_eq!(profile.hottest_block(), Some((cfg.entry(), 1.0)));
    assert_eq!(profile.coldest_block(), Some((cfg.entry(), 1.0)));
    assert!(profile.edge_weights.is_empty());
}

/// A random forward graph where block `i` always reaches `i + 1` and may also
/// reach later blocks, with random branch probabilities summing to one.
fn random_forward_graph(seed: u64) -> (Cfg<Stmt>, Vec<Vec<(usize, f64)>>) {
    let mut rng = Rng::new(seed);
    let size = 5 + rng.below(6);
    let mut cfg = Cfg::<Stmt>::new();
    let blocks: Vec<BlockId> = std::iter::once(cfg.entry())
        .chain((1..size).map(|_| cfg.new_block()))
        .collect();
    let mut outgoing = vec![Vec::new(); size];
    for source in 0..size - 1 {
        let mut targets = vec![source + 1];
        for _ in 0..rng.below(3) {
            let later = source + 2 + rng.below(size - source - 1);
            if later < size && !targets.contains(&later) {
                targets.push(later);
            }
        }
        let shares: Vec<usize> = targets.iter().map(|_| 1 + rng.below(4)).collect();
        let total: usize = shares.iter().sum();
        for (&target, &share) in targets.iter().zip(&shares) {
            let probability =
                f64::from(u32::try_from(share).unwrap()) / f64::from(u32::try_from(total).unwrap());
            let edge = cfg.add_edge(blocks[source], blocks[target], EdgeKind::ConditionalTrue);
            cfg.edge_mut(edge).set_weight(Some(probability));
            outgoing[source].push((target, probability));
        }
    }
    (cfg, outgoing)
}

#[test]
fn frequencies_obey_flow_conservation_and_send_the_whole_unit_out_through_the_exit() {
    for seed in 0..60 {
        let (cfg, outgoing) = random_forward_graph(seed);
        let size = outgoing.len();
        let mut expected = vec![0.0_f64; size];
        expected[0] = 1.0;
        for source in 0..size {
            for &(target, probability) in &outgoing[source] {
                expected[target] += expected[source] * probability;
            }
        }

        let profile = CfgProfile::from_edge_weights(&cfg);
        let by_index: BTreeMap<usize, f64> = profile
            .block_weights
            .iter()
            .map(|(block, &weight)| (block.index(), weight))
            .collect();
        for (index, &frequency) in expected.iter().enumerate() {
            let actual = by_index[&index];
            assert!(
                close(actual, frequency),
                "seed {seed}: bb{index} is {actual}, flow says {frequency}"
            );
        }
        assert!(
            close(by_index[&(size - 1)], 1.0),
            "seed {seed}: everything reaches the single exit once"
        );
    }
}
