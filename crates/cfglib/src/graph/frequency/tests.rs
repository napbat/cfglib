//! Frequencies against hand-computed loops and branches, the flow equations
//! on random reducible graphs, and invariance under renumbering.

use super::*;
use crate::cfg::Cfg;
use crate::edge::EdgeKind;
use crate::graph::dominator::DominatorTree;
use crate::graph::store::Graph;
use crate::graph::structure::is_reducible;
use crate::graph::view::Rooted;
use crate::test_util::rng::Rng;

const ONCE: u64 = 1 << FREQUENCY_SHIFT;

/// A CFG over `count` blocks, block 0 the entry, with one probability per edge
/// in edge order.
struct Fixture {
    cfg: Cfg<u32>,
    probability: Vec<Option<Probability>>,
}

impl Fixture {
    fn new(count: usize) -> Self {
        let mut cfg = Cfg::<u32>::new();
        for _ in 1..count {
            cfg.new_block();
        }
        Self {
            cfg,
            probability: Vec::new(),
        }
    }

    fn edge(&mut self, from: usize, to: usize, probability: Option<f64>) -> EdgeId {
        let id = self.cfg.add_edge(
            BlockId::from_index(from),
            BlockId::from_index(to),
            EdgeKind::Jump,
        );
        self.probability
            .push(probability.map(|p| Probability::from_f64(p).expect("a probability")));
        id
    }

    fn frequencies(&self, config: FrequencyConfig) -> BlockFrequencies {
        let dominators = DominatorTree::compute(&self.cfg);
        let loops = LoopForest::compute(&self.cfg, &dominators);
        BlockFrequencies::compute(
            &self.cfg,
            &loops,
            |edge| self.probability[edge.index()],
            config,
        )
    }

    fn block(frequencies: &BlockFrequencies, block: usize) -> u64 {
        frequencies.block(BlockId::from_index(block))
    }
}

/// `fraction` of one execution of the entry.
fn share(fraction: f64) -> f64 {
    fraction * f64::from(1_u32 << FREQUENCY_SHIFT)
}

#[track_caller]
fn assert_close(actual: u64, expected: f64, ulps: f64) {
    let actual = crate::usize_to_f64(usize::try_from(actual).expect("a test frequency fits usize"));
    assert!(
        (actual - expected).abs() <= ulps,
        "{actual} is not within {ulps} of {expected}"
    );
}

#[test]
fn a_diamond_splits_by_its_branch_probabilities() {
    let mut fixture = Fixture::new(4);
    let left = fixture.edge(0, 1, Some(0.9));
    let right = fixture.edge(0, 2, Some(0.1));
    fixture.edge(1, 3, None);
    fixture.edge(2, 3, None);
    let frequencies = fixture.frequencies(FrequencyConfig::new());

    assert_eq!(Fixture::block(&frequencies, 0), ONCE);
    assert_close(Fixture::block(&frequencies, 1), share(0.9), 1.0);
    assert_close(Fixture::block(&frequencies, 2), share(0.1), 1.0);
    assert_close(Fixture::block(&frequencies, 3), share(1.0), 1.0);
    assert_eq!(frequencies.edge(left), Fixture::block(&frequencies, 1));
    assert_eq!(frequencies.edge(right), Fixture::block(&frequencies, 2));
}

#[test]
fn a_loop_that_returns_nine_times_in_ten_runs_its_header_ten_times() {
    // entry -> header -> latch; latch -> header (0.9) or exit (0.1).
    let mut fixture = Fixture::new(4);
    fixture.edge(0, 1, None);
    fixture.edge(1, 2, None);
    fixture.edge(2, 1, Some(0.9));
    fixture.edge(2, 3, Some(0.1));
    let frequencies = fixture.frequencies(FrequencyConfig::new());

    // 0.9 rounds to the 31-bit probability 0.8999999999, whose scale is 10
    // less 1.6 units of the last place.
    assert_close(Fixture::block(&frequencies, 1), share(10.0), 2.0);
    assert_close(Fixture::block(&frequencies, 2), share(10.0), 2.0);
    assert_close(Fixture::block(&frequencies, 3), share(1.0), 1.0);
}

#[test]
fn nested_loops_multiply_their_scales() {
    // entry -> outer -> inner -> inner_latch; inner_latch -> inner (3/4) or
    // outer_latch (1/4); outer_latch -> outer (1/2) or exit (1/2).
    let mut fixture = Fixture::new(6);
    fixture.edge(0, 1, None);
    fixture.edge(1, 2, None);
    fixture.edge(2, 3, None);
    fixture.edge(3, 2, Some(0.75));
    fixture.edge(3, 4, Some(0.25));
    fixture.edge(4, 1, Some(0.5));
    fixture.edge(4, 5, Some(0.5));
    let frequencies = fixture.frequencies(FrequencyConfig::new());

    let runs: Vec<u64> = (0..6)
        .map(|block| Fixture::block(&frequencies, block))
        .collect();
    assert_eq!(
        runs,
        [1, 2, 8, 8, 2, 1].map(|scale| scale * ONCE),
        "the outer loop runs twice and the inner one four times per entry"
    );
}

#[test]
fn a_loop_that_never_exits_runs_exactly_the_capped_scale() {
    // entry -> header; header -> header (1.0) or exit (0.0).
    let mut fixture = Fixture::new(3);
    fixture.edge(0, 1, None);
    let back = fixture.edge(1, 1, Some(1.0));
    let leave = fixture.edge(1, 2, Some(0.0));

    let default = fixture.frequencies(FrequencyConfig::new());
    assert_eq!(
        Fixture::block(&default, 1),
        u64::from(FrequencyConfig::DEFAULT_MAX_LOOP_SCALE) * ONCE
    );
    assert_eq!(default.edge(back), Fixture::block(&default, 1));
    assert_eq!(default.edge(leave), 0);
    assert_eq!(Fixture::block(&default, 2), 0);

    let capped = fixture.frequencies(FrequencyConfig::new().with_max_loop_scale(100));
    assert_eq!(Fixture::block(&capped, 1), 100 * ONCE);
    let floor = fixture.frequencies(FrequencyConfig::new().with_max_loop_scale(0));
    assert_eq!(
        Fixture::block(&floor, 1),
        ONCE,
        "a cap below one counts as one"
    );
}

#[test]
fn deep_nests_of_long_loops_saturate_instead_of_overflowing() {
    // Three nested loops that each return with probability 1 - 2^-31, so each
    // multiplies by 2^31 under a cap that does not bind: 2^93 in all.
    const DEPTH: usize = 3;
    let almost = Probability::from_numerator(Probability::DENOMINATOR - 1).expect("below one");
    let mut cfg = Cfg::<u32>::new();
    let headers: Vec<BlockId> = (0..DEPTH).map(|_| cfg.new_block()).collect();
    let latches: Vec<BlockId> = (0..DEPTH).map(|_| cfg.new_block()).collect();
    let exit = cfg.new_block();
    let mut probability = Vec::new();
    let mut add = |cfg: &mut Cfg<u32>, from, to, p: Option<Probability>| {
        cfg.add_edge(from, to, EdgeKind::Jump);
        probability.push(p);
    };
    let entry = cfg.entry();
    add(&mut cfg, entry, headers[0], None);
    for depth in 0..DEPTH {
        let next = headers.get(depth + 1).copied().unwrap_or(latches[depth]);
        add(&mut cfg, headers[depth], next, None);
        add(&mut cfg, latches[depth], headers[depth], Some(almost));
        let out = if depth == 0 { exit } else { latches[depth - 1] };
        add(&mut cfg, latches[depth], out, None);
    }
    let dominators = DominatorTree::compute(&cfg);
    let loops = LoopForest::compute(&cfg, &dominators);
    assert_eq!(loops.len(), DEPTH);
    let frequencies = BlockFrequencies::compute(
        &cfg,
        &loops,
        |edge| probability[edge.index()],
        FrequencyConfig::new().with_max_loop_scale(u32::MAX),
    );

    assert_eq!(frequencies.block(cfg.entry()), ONCE);
    assert_eq!(frequencies.block(headers[0]), (1 << 31) * ONCE);
    assert_eq!(frequencies.block(headers[2]), u64::MAX);
    assert_eq!(frequencies.block(latches[2]), u64::MAX);
    assert_eq!(frequencies.block(exit), ONCE);
}

#[test]
fn missing_probabilities_share_what_the_stated_ones_leave() {
    let mut fixture = Fixture::new(4);
    let stated = fixture.edge(0, 1, Some(0.5));
    let first = fixture.edge(0, 2, None);
    let second = fixture.edge(0, 3, None);
    let frequencies = fixture.frequencies(FrequencyConfig::new());
    assert_eq!(frequencies.edge(stated), ONCE / 2);
    assert_eq!(frequencies.edge(first), ONCE / 4);
    assert_eq!(frequencies.edge(second), ONCE / 4);

    let mut fixture = Fixture::new(4);
    let edges = [1, 2, 3].map(|to| fixture.edge(0, to, None));
    let frequencies = fixture.frequencies(FrequencyConfig::new());
    for edge in edges {
        assert_close(frequencies.edge(edge), share(1.0 / 3.0), 1.0);
    }

    let mut fixture = Fixture::new(3);
    let stated = fixture.edge(0, 1, Some(1.0));
    let missing = fixture.edge(0, 2, None);
    let frequencies = fixture.frequencies(FrequencyConfig::new());
    assert_eq!(frequencies.edge(stated), ONCE);
    assert_eq!(frequencies.edge(missing), 0, "nothing is left to share");
}

#[test]
fn probabilities_that_miss_one_are_renormalized_per_source() {
    let mut fixture = Fixture::new(3);
    let heavy = fixture.edge(0, 1, Some(0.75));
    let light = fixture.edge(0, 2, Some(0.75));
    let frequencies = fixture.frequencies(FrequencyConfig::new());
    assert_eq!(frequencies.edge(heavy), ONCE / 2);
    assert_eq!(frequencies.edge(light), ONCE / 2);

    let mut fixture = Fixture::new(3);
    let small = fixture.edge(0, 1, Some(0.125));
    let large = fixture.edge(0, 2, Some(0.375));
    let frequencies = fixture.frequencies(FrequencyConfig::new());
    assert_eq!(frequencies.edge(small), ONCE / 4);
    assert_eq!(frequencies.edge(large), 3 * ONCE / 4);

    let mut fixture = Fixture::new(3);
    let zeros = [1, 2].map(|to| fixture.edge(0, to, Some(0.0)));
    let frequencies = fixture.frequencies(FrequencyConfig::new());
    for edge in zeros {
        assert_eq!(frequencies.edge(edge), ONCE / 2, "all zero splits evenly");
    }
}

#[test]
fn unreachable_removed_and_unknown_blocks_never_run() {
    let mut fixture = Fixture::new(4);
    fixture.edge(0, 1, None);
    let dead = fixture.edge(2, 1, None);
    fixture.edge(1, 3, None);
    fixture.cfg.remove_block(BlockId::from_index(3));
    let frequencies = fixture.frequencies(FrequencyConfig::new());

    assert_eq!(Fixture::block(&frequencies, 1), ONCE);
    assert_eq!(Fixture::block(&frequencies, 2), 0);
    assert_eq!(Fixture::block(&frequencies, 3), 0);
    assert_eq!(frequencies.edge(dead), 0);
    assert_eq!(Fixture::block(&frequencies, 99), 0);
    assert_eq!(frequencies.edge(EdgeId::from_index(99)), 0);
}

#[test]
fn a_lone_entry_runs_once_and_an_entry_loop_runs_its_scale() {
    let fixture = Fixture::new(1);
    let frequencies = fixture.frequencies(FrequencyConfig::new());
    assert_eq!(Fixture::block(&frequencies, 0), ONCE);

    // The entry heads a loop: it runs once per entry times the loop's scale.
    let mut fixture = Fixture::new(2);
    fixture.edge(0, 0, Some(0.5));
    fixture.edge(0, 1, Some(0.5));
    let frequencies = fixture.frequencies(FrequencyConfig::new());
    assert_eq!(Fixture::block(&frequencies, 0), 2 * ONCE);
    assert_eq!(Fixture::block(&frequencies, 1), ONCE);
}

/// entry branches into both `a` and `b`, which jump to each other; `b` exits.
fn irreducible_pair(tagged: bool) -> (Cfg<u32>, [BlockId; 4]) {
    let mut cfg = Cfg::<u32>::new();
    let [a, b, exit] = [cfg.new_block(), cfg.new_block(), cfg.new_block()];
    cfg.add_edge(cfg.entry(), a, EdgeKind::ConditionalTrue);
    cfg.add_edge(cfg.entry(), b, EdgeKind::ConditionalFalse);
    cfg.add_edge(a, b, EdgeKind::Fallthrough);
    let kind = if tagged {
        EdgeKind::Back
    } else {
        EdgeKind::ConditionalTrue
    };
    cfg.add_edge(b, a, kind);
    cfg.add_edge(b, exit, EdgeKind::ConditionalFalse);
    let entry = cfg.entry();
    (cfg, [entry, a, b, exit])
}

#[test]
fn an_irreducible_cycle_is_approximated_as_documented() {
    // Every branch splits evenly. The exact answer is a = 1.5, b = 2, exit = 1.
    let (cfg, [entry, a, b, exit]) = irreducible_pair(false);
    let dominators = DominatorTree::compute(&cfg);
    let even = |_| None;

    // Without a loop, the edge `b -> a` closes a cycle no loop names: it is
    // taken, but adds nothing to `a`.
    let loops = LoopForest::compute(&cfg, &dominators);
    assert!(loops.is_empty());
    let plain = BlockFrequencies::compute(&cfg, &loops, even, FrequencyConfig::new());
    assert_eq!(
        [entry, a, b, exit].map(|block| plain.block(block)),
        [ONCE, ONCE / 2, ONCE, ONCE / 2]
    );

    // With the tag, the cycle is a loop headed by `a`, and the mass entering
    // at `b` is propagated as if it entered at `a`.
    let (cfg, [entry, a, b, exit]) = irreducible_pair(true);
    let dominators = DominatorTree::compute(&cfg);
    let loops = LoopForest::compute_tagged(&cfg, &dominators);
    assert_eq!(loops.len(), 1);
    let tagged = BlockFrequencies::compute(&cfg, &loops, even, FrequencyConfig::new());
    assert_eq!(
        [entry, a, b, exit].map(|block| tagged.block(block)),
        [ONCE, 2 * ONCE, 2 * ONCE, ONCE]
    );
}

#[test]
fn probabilities_round_to_the_nearest_fixed_point_value() {
    assert_eq!(Probability::from_f64(0.0), Some(Probability::ZERO));
    assert_eq!(Probability::from_f64(-0.0), Some(Probability::ZERO));
    assert_eq!(Probability::from_f64(1.0), Some(Probability::ONE));
    assert_eq!(
        Probability::from_f64(0.9).map(Probability::numerator),
        Some(1_932_735_283)
    );
    let half_unit = 0.5 / f64::from(Probability::DENOMINATOR);
    assert_eq!(
        Probability::from_f64(half_unit).map(Probability::numerator),
        Some(1),
        "a tie rounds upward"
    );
    for outside in [-0.1, 1.000_001, f64::NAN, f64::INFINITY] {
        assert_eq!(Probability::from_f64(outside), None, "{outside}");
    }
    assert_eq!(
        Probability::from_numerator(Probability::DENOMINATOR),
        Some(Probability::ONE)
    );
    assert_eq!(
        Probability::from_numerator(Probability::DENOMINATOR + 1),
        None
    );
}

/// `count` probabilities that add up to exactly one.
fn random_split(rng: &mut Rng, count: usize) -> Vec<Probability> {
    let denominator = usize::try_from(Probability::DENOMINATOR).expect("2^31 fits usize");
    let mut cuts: Vec<usize> = (1..count).map(|_| rng.below(denominator + 1)).collect();
    cuts.push(0);
    cuts.push(denominator);
    cuts.sort_unstable();
    cuts.windows(2)
        .map(|pair| {
            let part = u32::try_from(pair[1] - pair[0]).expect("a part of 2^31 fits u32");
            Probability::from_numerator(part).expect("a part of one")
        })
        .collect()
}

/// A random graph of `count` nodes: each node has one to three successors,
/// with probabilities that add up to one, as `(from, to, probability)`.
fn random_edges(rng: &mut Rng, count: usize) -> Vec<(usize, usize, Probability)> {
    let mut edges = Vec::new();
    for from in 0..count {
        let successors = 1 + rng.below(3);
        let split = random_split(rng, successors);
        for probability in split {
            edges.push((from, rng.below(count), probability));
        }
    }
    edges
}

#[test]
fn random_reducible_graphs_obey_the_flow_equations() {
    let mut rng = Rng::new(42);
    let mut checked = 0;
    for _ in 0..2_000 {
        let count = 2 + rng.below(10);
        let edges = random_edges(&mut rng, count);
        let mut cfg = Cfg::<u32>::new();
        for _ in 1..count {
            cfg.new_block();
        }
        let mut probability = Vec::new();
        for &(from, to, p) in &edges {
            cfg.add_edge(
                BlockId::from_index(from),
                BlockId::from_index(to),
                EdgeKind::Jump,
            );
            probability.push(p);
        }
        let dominators = DominatorTree::compute(&cfg);
        if !is_reducible(&cfg, &dominators) {
            continue;
        }
        checked += 1;
        let loops = LoopForest::compute(&cfg, &dominators);
        let frequencies = BlockFrequencies::compute(
            &cfg,
            &loops,
            |edge| Some(probability[edge.index()]),
            FrequencyConfig::new(),
        );

        for edge in cfg.edge_ids() {
            let source = frequencies.block(cfg.edge(edge).source());
            let p = u128::from(probability[edge.index()].numerator());
            let expected = (u128::from(source) * p + (1 << 30)) >> 31;
            assert_eq!(u128::from(frequencies.edge(edge)), expected);
        }
        for block in cfg.block_ids() {
            if block == cfg.entry() || loops.loop_of_header(block).is_some() {
                continue;
            }
            let incoming: Vec<EdgeId> = cfg.incoming(block).collect();
            let sum: u64 = incoming.iter().map(|&edge| frequencies.edge(edge)).sum();
            let slack = u64::try_from(incoming.len()).expect("fits") + 1;
            assert!(
                frequencies.block(block).abs_diff(sum) <= slack,
                "block {block:?} runs {} but its edges bring {sum}",
                frequencies.block(block)
            );
        }
    }
    assert!(checked > 500, "only {checked} reducible graphs");
}

/// The frequencies of a random graph whose node `i` is `slot[i]`, read back
/// in the order of `i`, with edges in the same order.
fn renumbered_frequencies(
    count: usize,
    edges: &[(usize, usize, Probability)],
    slot: &[usize],
) -> (Vec<u64>, Vec<u64>) {
    let mut graph = Graph::<(), ()>::new();
    let nodes: Vec<_> = (0..count).map(|_| graph.add_node(())).collect();
    let ids: Vec<_> = edges
        .iter()
        .map(|&(from, to, _)| graph.add_edge(nodes[slot[from]], nodes[slot[to]], ()))
        .collect();
    let rooted = Rooted::new(&graph, nodes[slot[0]]);
    let dominators = DominatorTree::compute(&rooted);
    let loops = LoopForest::compute(&rooted, &dominators);
    let probability: Vec<Probability> = edges.iter().map(|&(_, _, p)| p).collect();
    let frequencies = BlockFrequencies::compute(
        &rooted,
        &loops,
        |edge| Some(probability[edge.index()]),
        FrequencyConfig::new(),
    );
    (
        (0..count)
            .map(|node| frequencies.block(nodes[slot[node]]))
            .collect(),
        ids.into_iter().map(|edge| frequencies.edge(edge)).collect(),
    )
}

#[test]
fn renumbering_the_nodes_does_not_change_the_frequencies() {
    let mut rng = Rng::new(7);
    for _ in 0..500 {
        let count = 1 + rng.below(10);
        let edges = random_edges(&mut rng, count);
        let identity: Vec<usize> = (0..count).collect();
        let mut permuted = identity.clone();
        rng.shuffle(&mut permuted);
        assert_eq!(
            renumbered_frequencies(count, &edges, &identity),
            renumbered_frequencies(count, &edges, &permuted),
            "{edges:?} under {permuted:?}"
        );
    }
}

#[test]
fn a_plain_graph_rooted_away_from_slot_zero_runs_its_root_once() {
    // A view without edge kinds or weights, rooted at its last node.
    let mut graph = Graph::<(), ()>::new();
    let nodes: Vec<_> = (0..3).map(|_| graph.add_node(())).collect();
    graph.add_edge(nodes[2], nodes[0], ());
    graph.add_edge(nodes[2], nodes[1], ());
    let rooted = Rooted::new(&graph, nodes[2]);
    let dominators = DominatorTree::compute(&rooted);
    let loops = LoopForest::compute(&rooted, &dominators);
    let frequencies =
        BlockFrequencies::compute(&rooted, &loops, |_| None, FrequencyConfig::default());
    assert_eq!(frequencies.block(nodes[2]), ONCE);
    assert_eq!(frequencies.block(nodes[0]), ONCE / 2);
    assert_eq!(frequencies.block(nodes[1]), ONCE / 2);
}
