//! Structural idioms: diamonds, chains, self-loops, and empty trampolines.

use std::collections::BTreeSet;

use cfglib::{
    BlockId, Cfg, CfgPattern, EdgeKind, Graph, NodeId, detect_cfg_patterns, detect_patterns,
};

use crate::random::Rng;
use crate::stmt::{Route, Stmt, block, link};

fn diamonds<N: Clone>(patterns: &[CfgPattern<N>]) -> Vec<CfgPattern<N>> {
    patterns
        .iter()
        .filter(|pattern| matches!(pattern, CfgPattern::Diamond { .. }))
        .cloned()
        .collect()
}

fn chains(patterns: &[CfgPattern]) -> Vec<Vec<BlockId>> {
    patterns
        .iter()
        .filter_map(|pattern| match pattern {
            CfgPattern::Chain { blocks } => Some(blocks.clone()),
            _ => None,
        })
        .collect()
}

#[test]
fn a_branch_whose_arms_rejoin_is_a_diamond_with_the_true_arm_first_when_oriented() {
    let mut cfg = Cfg::<Stmt, Route>::with_edge_payload();
    let entry = cfg.entry();
    let (else_arm, then_arm, merge) = (
        block(&mut cfg, [Stmt::plain()]),
        block(&mut cfg, [Stmt::plain()]),
        block(&mut cfg, [Stmt::plain()]),
    );
    link(&mut cfg, entry, else_arm, EdgeKind::ConditionalFalse);
    link(&mut cfg, entry, then_arm, EdgeKind::ConditionalTrue);
    link(&mut cfg, else_arm, merge, EdgeKind::Fallthrough);
    link(&mut cfg, then_arm, merge, EdgeKind::Fallthrough);

    assert_eq!(
        diamonds(&detect_patterns(&cfg)),
        [CfgPattern::Diamond {
            entry,
            arms: [else_arm, then_arm],
            merge
        }],
        "the topology-only scan keeps successor order"
    );
    assert_eq!(
        diamonds(&detect_cfg_patterns(&cfg)),
        [CfgPattern::Diamond {
            entry,
            arms: [then_arm, else_arm],
            merge
        }],
        "the CFG scan puts the ConditionalTrue side first"
    );
}

#[test]
fn two_parallel_edges_to_one_block_are_not_a_diamond() {
    let mut cfg = Cfg::<Stmt, Route>::with_edge_payload();
    let entry = cfg.entry();
    let (arm, merge) = (
        block(&mut cfg, [Stmt::plain()]),
        block(&mut cfg, [Stmt::plain()]),
    );
    link(&mut cfg, entry, arm, EdgeKind::ConditionalTrue);
    link(&mut cfg, entry, arm, EdgeKind::ConditionalFalse);
    link(&mut cfg, arm, merge, EdgeKind::Fallthrough);

    assert_eq!(diamonds(&detect_patterns(&cfg)), []);
    assert_eq!(diamonds(&detect_cfg_patterns(&cfg)), []);
}

#[test]
fn a_diamond_followed_by_a_tail_reports_the_diamond_and_the_chain_through_its_merge() {
    let mut cfg = Cfg::<Stmt, Route>::with_edge_payload();
    let entry = cfg.entry();
    let (left, right, merge, first, second) = (
        block(&mut cfg, [Stmt::plain()]),
        block(&mut cfg, [Stmt::plain()]),
        block(&mut cfg, [Stmt::plain()]),
        block(&mut cfg, [Stmt::plain()]),
        block(&mut cfg, [Stmt::plain()]),
    );
    link(&mut cfg, entry, left, EdgeKind::ConditionalTrue);
    link(&mut cfg, entry, right, EdgeKind::ConditionalFalse);
    link(&mut cfg, left, merge, EdgeKind::Fallthrough);
    link(&mut cfg, right, merge, EdgeKind::Fallthrough);
    link(&mut cfg, merge, first, EdgeKind::Fallthrough);
    link(&mut cfg, first, second, EdgeKind::Fallthrough);

    let patterns = detect_patterns(&cfg);

    assert_eq!(
        patterns,
        [
            CfgPattern::Diamond {
                entry,
                arms: [left, right],
                merge
            },
            CfgPattern::Chain {
                blocks: vec![merge, first, second]
            },
        ]
    );
}

#[test]
fn a_straight_line_of_single_links_is_one_chain_from_its_first_block_to_its_last() {
    let mut cfg = Cfg::<Stmt, Route>::with_edge_payload();
    let entry = cfg.entry();
    let mut line = vec![entry];
    for _ in 0..4 {
        let next = block(&mut cfg, [Stmt::plain()]);
        link(&mut cfg, *line.last().unwrap(), next, EdgeKind::Fallthrough);
        line.push(next);
    }

    assert_eq!(chains(&detect_patterns(&cfg)), [line]);
}

#[test]
fn a_block_with_an_edge_to_itself_is_a_self_loop_once_however_many_such_edges_it_has() {
    let mut cfg = Cfg::<Stmt, Route>::with_edge_payload();
    let entry = cfg.entry();
    let spin = block(&mut cfg, [Stmt::plain()]);
    link(&mut cfg, entry, spin, EdgeKind::Fallthrough);
    link(&mut cfg, spin, spin, EdgeKind::Back);
    link(&mut cfg, spin, spin, EdgeKind::Back);

    let loops: Vec<_> = detect_patterns(&cfg)
        .into_iter()
        .filter(|pattern| matches!(pattern, CfgPattern::SelfLoop { .. }))
        .collect();

    assert_eq!(loops, [CfgPattern::SelfLoop { block: spin }]);
}

#[test]
fn a_cycle_of_single_links_terminates_and_reports_each_block_at_most_once() {
    let mut cfg = Cfg::<Stmt, Route>::with_edge_payload();
    let entry = cfg.entry();
    let ring: Vec<BlockId> = std::iter::once(entry)
        .chain((0..3).map(|_| block(&mut cfg, [Stmt::plain()])))
        .collect();
    for index in 0..ring.len() {
        link(
            &mut cfg,
            ring[index],
            ring[(index + 1) % ring.len()],
            EdgeKind::Fallthrough,
        );
    }

    let found = chains(&detect_patterns(&cfg));
    let covered: Vec<BlockId> = found.iter().flatten().copied().collect();

    assert_eq!(
        covered.iter().collect::<BTreeSet<_>>().len(),
        covered.len(),
        "no block twice"
    );
    assert!(covered.iter().all(|block| ring.contains(block)));
}

#[test]
fn an_empty_block_with_one_way_out_is_a_trampoline_unless_it_is_the_entry() {
    let mut cfg = Cfg::<Stmt, Route>::with_edge_payload();
    let entry = cfg.entry();
    let trampoline = block(&mut cfg, []);
    let busy = block(&mut cfg, [Stmt::plain()]);
    let two_ways = block(&mut cfg, []);
    let (left, right, exit) = (
        block(&mut cfg, [Stmt::plain()]),
        block(&mut cfg, [Stmt::plain()]),
        block(&mut cfg, []),
    );
    link(&mut cfg, entry, trampoline, EdgeKind::Fallthrough);
    link(&mut cfg, trampoline, busy, EdgeKind::Fallthrough);
    link(&mut cfg, busy, two_ways, EdgeKind::Fallthrough);
    link(&mut cfg, two_ways, left, EdgeKind::ConditionalTrue);
    link(&mut cfg, two_ways, right, EdgeKind::ConditionalFalse);
    link(&mut cfg, left, exit, EdgeKind::Fallthrough);
    link(&mut cfg, right, exit, EdgeKind::Fallthrough);

    let trampolines: BTreeSet<BlockId> = detect_cfg_patterns(&cfg)
        .into_iter()
        .filter_map(|pattern| match pattern {
            CfgPattern::EmptyTrampoline { block } => Some(block),
            _ => None,
        })
        .collect();

    assert_eq!(trampolines, BTreeSet::from([trampoline]));
    assert!(
        detect_patterns(&cfg)
            .iter()
            .all(|pattern| !matches!(pattern, CfgPattern::EmptyTrampoline { .. })),
        "emptiness is an instruction-level property the topology scan cannot see"
    );
}

#[test]
fn the_topology_scan_serves_any_graph_and_skips_removed_nodes() {
    let mut graph = Graph::<(), ()>::new();
    let nodes: Vec<NodeId> = (0..5).map(|_| graph.add_node(())).collect();
    for (source, target) in [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)] {
        graph.add_edge(nodes[source], nodes[target], ());
    }
    assert_eq!(
        diamonds(&detect_patterns(&graph)),
        [CfgPattern::Diamond {
            entry: nodes[0],
            arms: [nodes[1], nodes[2]],
            merge: nodes[3]
        }]
    );

    graph.remove_node(nodes[2]);
    assert_eq!(diamonds(&detect_patterns(&graph)), []);
}

fn random_forward_cfg(seed: u64) -> (Cfg<Stmt>, Vec<Vec<usize>>) {
    let mut rng = Rng::new(seed);
    let size = 6 + rng.below(5);
    let mut cfg = Cfg::<Stmt>::new();
    let blocks: Vec<BlockId> = std::iter::once(cfg.entry())
        .chain((1..size).map(|_| cfg.new_block()))
        .collect();
    let mut successors = vec![Vec::new(); size];
    for source in 0..size {
        for target in source + 1..size {
            if rng.chance(30) {
                let kind = if rng.chance(50) {
                    EdgeKind::ConditionalTrue
                } else {
                    EdgeKind::ConditionalFalse
                };
                cfg.add_edge(blocks[source], blocks[target], kind);
                successors[source].push(target);
            }
        }
        if rng.chance(70) {
            cfg.block_mut(blocks[source]).push(Stmt::plain());
        }
    }
    (cfg, successors)
}

#[test]
fn diamonds_are_exactly_two_way_branches_whose_arms_each_lead_only_to_one_shared_block() {
    for seed in 0..120 {
        let (cfg, successors) = random_forward_cfg(seed);
        let mut expected = BTreeSet::new();
        for (entry, outgoing) in successors.iter().enumerate() {
            let [left, right] = outgoing[..] else {
                continue;
            };
            if let ([merge], [other]) = (&successors[left][..], &successors[right][..])
                && merge == other
            {
                expected.insert((entry, [left, right], *merge));
            }
        }

        let found: BTreeSet<(usize, [usize; 2], usize)> = diamonds(&detect_patterns(&cfg))
            .into_iter()
            .map(|pattern| match pattern {
                CfgPattern::Diamond { entry, arms, merge } => {
                    (entry.index(), arms.map(BlockId::index), merge.index())
                }
                _ => unreachable!(),
            })
            .collect();
        assert_eq!(found, expected, "seed {seed}");

        let oriented: BTreeSet<(usize, BTreeSet<usize>, usize)> =
            diamonds(&detect_cfg_patterns(&cfg))
                .into_iter()
                .map(|pattern| match pattern {
                    CfgPattern::Diamond { entry, arms, merge } => {
                        let true_target = cfg
                            .outgoing(entry)
                            .find(|&edge| cfg.edge(edge).kind() == EdgeKind::ConditionalTrue)
                            .map(|edge| cfg.edge(edge).target());
                        if let Some(target) = true_target {
                            assert_eq!(arms[0], target, "seed {seed}: the true arm leads");
                        }
                        (
                            entry.index(),
                            arms.iter().map(|arm| arm.index()).collect(),
                            merge.index(),
                        )
                    }
                    _ => unreachable!(),
                })
                .collect();
        let unordered: BTreeSet<_> = expected
            .iter()
            .map(|&(entry, arms, merge)| (entry, arms.into_iter().collect::<BTreeSet<_>>(), merge))
            .collect();
        assert_eq!(
            oriented, unordered,
            "seed {seed}: orientation reorders arms, never changes them"
        );
    }
}

#[test]
fn chains_are_node_disjoint_maximal_runs_of_single_links() {
    let mut reported = 0;
    for seed in 0..120 {
        let (cfg, successors) = random_forward_cfg(seed);
        let size = successors.len();
        let mut predecessors = vec![Vec::new(); size];
        for (source, targets) in successors.iter().enumerate() {
            for &target in targets {
                predecessors[target].push(source);
            }
        }
        // A link joins `from` to `to` when each is the other's only neighbor on that side.
        let linked =
            |from: usize, to: usize| successors[from] == [to] && predecessors[to] == [from];

        let found = chains(&detect_patterns(&cfg));
        let covered: Vec<usize> = found.iter().flatten().map(|block| block.index()).collect();
        assert_eq!(
            covered.iter().collect::<BTreeSet<_>>().len(),
            covered.len(),
            "seed {seed}: chains share a block"
        );
        for chain in &found {
            let indexes: Vec<usize> = chain.iter().map(|block| block.index()).collect();
            assert!(indexes.len() >= 2, "seed {seed}");
            assert!(
                indexes.windows(2).all(|pair| linked(pair[0], pair[1])),
                "seed {seed}: {indexes:?}"
            );
            let (first, last) = (indexes[0], *indexes.last().unwrap());
            assert!(
                !(0..size).any(|other| linked(other, first)),
                "seed {seed}: the chain could extend backwards"
            );
            assert!(
                !(0..size).any(|other| linked(last, other)),
                "seed {seed}: the chain could extend forwards"
            );
            reported += 1;
        }
    }
    assert!(
        reported > 20,
        "the random graphs produced only {reported} chains"
    );
}

#[test]
fn trampolines_are_exactly_the_empty_non_entry_blocks_with_one_way_out_on_random_graphs() {
    for seed in 0..120 {
        let (cfg, successors) = random_forward_cfg(seed);
        let expected: BTreeSet<usize> = (1..successors.len())
            .filter(|&index| {
                cfg.block(BlockId::from_index(index)).is_empty() && successors[index].len() == 1
            })
            .collect();
        let found: BTreeSet<usize> = detect_cfg_patterns(&cfg)
            .into_iter()
            .filter_map(|pattern| match pattern {
                CfgPattern::EmptyTrampoline { block } => Some(block.index()),
                _ => None,
            })
            .collect();
        assert_eq!(found, expected, "seed {seed}");
    }
}
