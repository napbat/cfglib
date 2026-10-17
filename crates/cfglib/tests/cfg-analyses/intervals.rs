//! First-level Allen-Cocke intervals against the textbook construction.

use std::collections::BTreeSet;

use cfglib::{BlockId, Cfg, DominatorTree, EdgeKind, IntervalAnalysis, is_reducible};

use crate::random::Rng;
use crate::stmt::Stmt;

/// A graph with every block reachable and random extra edges in both directions.
fn random_graph(seed: u64) -> (Cfg<Stmt>, Vec<Vec<usize>>) {
    let mut rng = Rng::new(seed);
    let size = 4 + rng.below(6);
    let mut cfg = Cfg::<Stmt>::new();
    let blocks: Vec<BlockId> = std::iter::once(cfg.entry())
        .chain((1..size).map(|_| cfg.new_block()))
        .collect();
    let mut successors = vec![Vec::new(); size];
    for source in 0..size {
        let mut targets = Vec::new();
        if source + 1 < size {
            targets.push(source + 1);
        }
        for _ in 0..rng.below(3) {
            targets.push(rng.below(size));
        }
        for target in targets {
            cfg.add_edge(blocks[source], blocks[target], EdgeKind::Fallthrough);
            successors[source].push(target);
        }
    }
    (cfg, successors)
}

/// The textbook algorithm: grow an interval from a header by admitting any
/// unassigned node all of whose predecessors are already inside; every
/// unassigned successor of a finished interval heads the next one.
fn textbook(successors: &[Vec<usize>]) -> BTreeSet<(usize, BTreeSet<usize>)> {
    let size = successors.len();
    let predecessors_of = |node: usize| -> Vec<usize> {
        (0..size)
            .filter(|&other| successors[other].contains(&node))
            .collect()
    };
    let mut assigned: BTreeSet<usize> = BTreeSet::new();
    let mut headers = vec![0];
    let mut intervals = BTreeSet::new();
    while let Some(header) = headers.pop() {
        if assigned.contains(&header) {
            continue;
        }
        let mut members = BTreeSet::from([header]);
        loop {
            let admitted = (0..size).find(|&node| {
                !assigned.contains(&node)
                    && !members.contains(&node)
                    && predecessors_of(node)
                        .iter()
                        .all(|predecessor| members.contains(predecessor))
            });
            match admitted {
                Some(node) => {
                    members.insert(node);
                }
                None => break,
            }
        }
        assigned.extend(&members);
        for &member in &members {
            headers.extend(
                successors[member]
                    .iter()
                    .filter(|next| !assigned.contains(next)),
            );
        }
        intervals.insert((header, members));
    }
    intervals
}

fn reported(analysis: &IntervalAnalysis) -> BTreeSet<(usize, BTreeSet<usize>)> {
    analysis
        .intervals
        .iter()
        .map(|interval| {
            (
                interval.header.index(),
                interval.blocks.iter().map(|block| block.index()).collect(),
            )
        })
        .collect()
}

#[test]
fn the_first_level_partition_is_the_textbook_one() {
    for seed in 0..200 {
        let (cfg, successors) = random_graph(seed);
        let analysis = IntervalAnalysis::compute(&cfg);

        let intervals = reported(&analysis);
        assert_eq!(intervals, textbook(&successors), "seed {seed}");

        let covered: Vec<usize> = intervals
            .iter()
            .flat_map(|(_, members)| members.iter().copied())
            .collect();
        assert_eq!(
            covered.iter().collect::<BTreeSet<_>>().len(),
            successors.len(),
            "seed {seed}: every block is in an interval"
        );
        assert_eq!(
            covered.len(),
            successors.len(),
            "seed {seed}: and in only one"
        );
    }
}

#[test]
fn a_single_interval_is_enough_to_prove_reducibility() {
    let mut single = 0;
    for seed in 0..200 {
        let (cfg, _) = random_graph(seed);
        if IntervalAnalysis::compute(&cfg).intervals.len() == 1 {
            single += 1;
            assert!(
                is_reducible(&cfg, &DominatorTree::compute(&cfg)),
                "seed {seed}"
            );
        }
    }
    assert!(single > 5, "no random graph collapsed to one interval");
}

#[test]
fn a_reducible_loop_can_still_split_into_several_first_level_intervals() {
    let mut cfg = Cfg::<Stmt>::new();
    let entry = cfg.entry();
    let (header, body, exit) = (cfg.new_block(), cfg.new_block(), cfg.new_block());
    cfg.add_edge(entry, header, EdgeKind::Fallthrough);
    cfg.add_edge(header, body, EdgeKind::ConditionalTrue);
    cfg.add_edge(header, exit, EdgeKind::ConditionalFalse);
    cfg.add_edge(body, header, EdgeKind::Back);

    let analysis = IntervalAnalysis::compute(&cfg);

    assert_eq!(
        reported(&analysis),
        BTreeSet::from([
            (0, BTreeSet::from([0])),
            (
                header.index(),
                BTreeSet::from([header.index(), body.index(), exit.index()])
            ),
        ]),
        "the loop header has a predecessor outside the entry's interval"
    );
    assert!(
        analysis.intervals.len() > 1,
        "several intervals do not prove irreducibility"
    );
    assert!(
        is_reducible(&cfg, &DominatorTree::compute(&cfg)),
        "the full test knows better"
    );
}

#[test]
fn a_cycle_with_two_entries_is_not_collapsed() {
    let mut cfg = Cfg::<Stmt>::new();
    let entry = cfg.entry();
    let (first, second) = (cfg.new_block(), cfg.new_block());
    cfg.add_edge(entry, first, EdgeKind::ConditionalTrue);
    cfg.add_edge(entry, second, EdgeKind::ConditionalFalse);
    cfg.add_edge(first, second, EdgeKind::Fallthrough);
    cfg.add_edge(second, first, EdgeKind::Fallthrough);

    let analysis = IntervalAnalysis::compute(&cfg);

    assert_eq!(
        analysis.intervals.len(),
        3,
        "the entry and each cycle node head their own interval"
    );
    assert!(!is_reducible(&cfg, &DominatorTree::compute(&cfg)));
}
