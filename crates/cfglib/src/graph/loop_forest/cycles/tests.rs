//! The cycle forest against Havlak's definition, computed by brute force: the
//! strongly connected components of the reached blocks, each headed by the
//! block a recursive walk visits first, and the same again inside each
//! without its header.

use super::*;
use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::edge::EdgeKind;
use crate::graph::dominator::DominatorTree;
use crate::graph::structure::is_reducible;
use crate::graph::view::GraphView;
use crate::test_util::rng::Rng;
use alloc::collections::BTreeSet;
use alloc::format;

/// A digraph of at most eight blocks as successor sets, one bit per block,
/// rooted at block 0 and read in ascending order.
struct Bits {
    successors: [u8; 8],
    count: usize,
}

impl GraphView for Bits {
    type NodeId = usize;

    fn node_bound(&self) -> usize {
        self.count
    }

    fn node_ids(&self) -> impl Iterator<Item = usize> + '_ {
        0..self.count
    }

    fn successors(&self, node: usize) -> impl Iterator<Item = usize> + '_ {
        (0..self.count).filter(move |&to| self.successors[node] >> to & 1 == 1)
    }

    fn predecessors(&self, node: usize) -> impl Iterator<Item = usize> + '_ {
        (0..self.count).filter(move |&from| self.successors[from] >> node & 1 == 1)
    }
}

impl RootedView for Bits {
    fn root(&self) -> usize {
        0
    }
}

/// One loop: what the definition says, or what a forest answers.
#[derive(Debug, PartialEq, Eq)]
struct Cycle<N> {
    header: N,
    body: BTreeSet<N>,
    latches: Vec<N>,
    entries: Vec<N>,
    parent: Option<N>,
    depth: usize,
}

fn cycles_of<N: DenseId>(forest: &LoopForest<N>) -> Vec<Cycle<N>> {
    forest
        .loops()
        .map(|lp| Cycle {
            header: forest.header(lp),
            body: forest.body(lp).collect(),
            latches: forest.latches(lp).to_vec(),
            entries: forest.entries(lp),
            parent: forest.parent(lp).map(|parent| forest.header(parent)),
            depth: forest.depth(lp),
        })
        .collect()
}

fn members(set: u64) -> impl Iterator<Item = usize> + Clone {
    (0..64).filter(move |&index| set >> index & 1 == 1)
}

/// The blocks of `within` that `start` reaches inside it along `adjacent`.
fn closure(adjacent: &[u64], start: usize, within: u64) -> u64 {
    let mut seen = 1 << start;
    loop {
        let grown = members(seen).fold(seen, |set, index| set | adjacent[index] & within);
        if grown == seen {
            return seen;
        }
        seen = grown;
    }
}

/// The graph's adjacency as bit sets, and its blocks in the preorder of
/// a recursive walk from the root, `usize::MAX` for a block it misses.
struct Oracle {
    successors: Vec<u64>,
    predecessors: Vec<u64>,
    preorder: Vec<usize>,
    reached: u64,
}

impl Oracle {
    fn visit<G: GraphView>(&mut self, graph: &G, block: G::NodeId, next: &mut usize) {
        self.preorder[block.index()] = *next;
        self.reached |= 1 << block.index();
        *next += 1;
        for successor in graph.successors(block) {
            if self.preorder[successor.index()] == usize::MAX {
                self.visit(graph, successor, next);
            }
        }
    }

    /// The cycles among `within`: its strongly connected components with
    /// an edge inside, each headed by the block the walk visits first, and
    /// within each the cycles of the rest without that header.
    fn split(
        &self,
        within: u64,
        parent: Option<usize>,
        depth: usize,
        found: &mut Vec<Cycle<usize>>,
    ) {
        let mut left = within;
        for start in members(within) {
            if left >> start & 1 == 0 {
                continue;
            }
            let component = closure(&self.successors, start, within)
                & closure(&self.predecessors, start, within);
            left &= !component;
            if component == 1 << start && self.successors[start] >> start & 1 == 0 {
                continue;
            }
            let header = members(component)
                .min_by_key(|&index| self.preorder[index])
                .expect("a component is not empty");
            let outside = self.reached & !component;
            found.push(Cycle {
                header,
                body: members(component).collect(),
                latches: members(component)
                    .filter(|&index| self.successors[index] >> header & 1 == 1)
                    .collect(),
                entries: core::iter::once(header)
                    .chain(members(component).filter(|&index| {
                        index != header && self.predecessors[index] & outside != 0
                    }))
                    .collect(),
                parent,
                depth,
            });
            self.split(component & !(1 << header), Some(header), depth + 1, found);
        }
    }
}

/// Havlak's forest by its definition, on a graph of at most 64 blocks.
fn brute_force<G: RootedView>(graph: &G) -> Vec<Cycle<G::NodeId>> {
    let bound = graph.node_bound();
    assert!(bound <= 64, "the oracle holds a set of blocks in one word");
    let mut oracle = Oracle {
        successors: vec![0; bound],
        predecessors: vec![0; bound],
        preorder: vec![usize::MAX; bound],
        reached: 0,
    };
    for from in graph.node_ids() {
        for to in graph.successors(from) {
            oracle.successors[from.index()] |= 1 << to.index();
            oracle.predecessors[to.index()] |= 1 << from.index();
        }
    }
    oracle.visit(graph, graph.root(), &mut 0);
    let mut found = Vec::new();
    oracle.split(oracle.reached, None, 0, &mut found);
    found.sort_by_key(|cycle| cycle.header);
    let block = |index: usize| G::NodeId::from_index(index);
    found
        .into_iter()
        .map(|cycle| Cycle {
            header: block(cycle.header),
            body: cycle.body.into_iter().map(block).collect(),
            latches: cycle.latches.into_iter().map(block).collect(),
            entries: cycle.entries.into_iter().map(block).collect(),
            parent: cycle.parent.map(block),
            depth: cycle.depth,
        })
        .collect()
}

/// Every question the forest answers, against the loops it lists.
fn assert_answers<N: DenseId>(forest: &LoopForest<N>, bound: usize, label: &str) {
    let cycles = cycles_of(forest);
    for (lp, cycle) in forest.loops().zip(&cycles) {
        assert_eq!(forest.loop_of_header(cycle.header), Some(lp), "{label}");
        assert!(forest.is_nested(lp), "{label}");
        assert_eq!(forest.enclosing(lp), forest.parent(lp), "{label}");
        assert_eq!(forest.body_len(lp), cycle.body.len(), "{label}");
        assert_eq!(forest.body(lp).count(), cycle.body.len(), "{label}");
        assert_eq!(cycle.entries.first(), Some(&cycle.header), "{label}");
        assert_eq!(forest.entry_count(lp), cycle.entries.len(), "{label}");
        assert_eq!(
            forest.is_irreducible(lp),
            cycle.entries.len() > 1,
            "{label}"
        );
        for block in (0..bound + 2).map(N::from_index) {
            assert_eq!(
                forest.contains(lp, block),
                cycle.body.contains(&block),
                "{label}: {block:?} in the loop of {:?}",
                cycle.header
            );
        }
        for (inner, inside) in forest.loops().zip(&cycles) {
            assert_eq!(
                forest.encloses(lp, inner),
                inside.body.is_subset(&cycle.body),
                "{label}"
            );
        }
    }
    let depths = forest.node_depths();
    assert_eq!(depths.len(), bound, "{label}");
    for (index, &depth) in depths.iter().enumerate() {
        let block = N::from_index(index);
        let around = || {
            forest
                .loops()
                .zip(&cycles)
                .filter(move |(_, cycle)| cycle.body.contains(&block))
        };
        assert_eq!(depth, around().count(), "{label}: depth of {block:?}");
        assert_eq!(
            forest.innermost(block),
            around()
                .min_by_key(|(_, cycle)| cycle.body.len())
                .map(|(lp, _)| lp),
            "{label}: innermost loop of {block:?}"
        );
    }
    assert_eq!(
        forest.max_nesting_depth(),
        cycles
            .iter()
            .map(|cycle| cycle.depth + 1)
            .max()
            .unwrap_or(0),
        "{label}"
    );
}

fn graph_of(count: usize, edges: &[(usize, usize, EdgeKind)]) -> (Cfg<u32>, Vec<BlockId>) {
    let mut cfg = Cfg::<u32>::new();
    let mut blocks = vec![cfg.entry()];
    blocks.extend((1..count).map(|_| cfg.new_block()));
    for &(from, to, kind) in edges {
        cfg.add_edge(blocks[from], blocks[to], kind);
    }
    (cfg, blocks)
}

#[test]
fn the_irreducible_core_is_one_loop_with_two_entries() {
    let (cfg, blocks) = graph_of(
        3,
        &[
            (0, 1, EdgeKind::ConditionalTrue),
            (0, 2, EdgeKind::ConditionalFalse),
            (1, 2, EdgeKind::Jump),
            (2, 1, EdgeKind::Jump),
        ],
    );
    let (a, b) = (blocks[1], blocks[2]);

    let forest = LoopForest::compute_cycles(&cfg);

    assert_eq!(
        cycles_of(&forest),
        [Cycle {
            header: a,
            body: BTreeSet::from([a, b]),
            latches: vec![b],
            entries: vec![a, b],
            parent: None,
            depth: 0,
        }]
    );
    let cycle = forest.loop_of_header(a).expect("a heads the cycle");
    assert!(forest.is_irreducible(cycle));
    assert_answers(&forest, cfg.node_bound(), "the irreducible core");
    // Dominance finds no loop there, and only a tag makes one.
    let dominators = DominatorTree::compute(&cfg);
    assert!(LoopForest::compute(&cfg, &dominators).is_empty());
    assert!(LoopForest::compute_tagged(&cfg, &dominators).is_empty());
}

#[test]
fn tags_play_no_part() {
    let edges = |closing| {
        [
            (0, 1, EdgeKind::ConditionalTrue),
            (0, 2, EdgeKind::ConditionalFalse),
            (1, 2, EdgeKind::Jump),
            (2, 1, closing),
            (2, 3, EdgeKind::Back),
        ]
    };
    let (untagged, _) = graph_of(4, &edges(EdgeKind::Jump));
    let (tagged, _) = graph_of(4, &edges(EdgeKind::Back));
    assert_eq!(
        LoopForest::compute_cycles(&untagged),
        LoopForest::compute_cycles(&tagged)
    );
    assert_eq!(LoopForest::compute_cycles(&tagged).len(), 1);
}

#[test]
fn every_digraph_of_up_to_five_blocks_has_the_forest_of_the_definition() {
    // Every digraph of up to four blocks, and every digraph of five blocks
    // up to its self-loops, with one of the 32 sets of self-loops picked
    // by a hash of its other edges: all 2^25 would take minutes.
    let mut irreducible = 0_usize;
    for count in 1..=5_usize {
        let pairs: Vec<(usize, usize)> = (0..count)
            .flat_map(|from| (0..count).map(move |to| (from, to)))
            .filter(|&(from, to)| count < 5 || from != to)
            .collect();
        for mask in 0..1_u32 << pairs.len() {
            let mut graph = Bits {
                successors: [0; 8],
                count,
            };
            for (bit, &(from, to)) in pairs.iter().enumerate() {
                graph.successors[from] |= u8::from(mask >> bit & 1 == 1) << to;
            }
            if count == 5 {
                let loops = mask.wrapping_mul(0x9E37_79B9) >> 27;
                for (block, successors) in graph.successors[..count].iter_mut().enumerate() {
                    *successors |= u8::from(loops >> block & 1 == 1) << block;
                }
            }
            let label = || format!("{count} blocks, successors {:?}", graph.successors);
            let forest = LoopForest::compute_cycles(&graph);
            let cycles = cycles_of(&forest);
            assert_eq!(cycles, brute_force(&graph), "{}", label());
            for (lp, cycle) in forest.loops().zip(&cycles) {
                assert_eq!(forest.entry_count(lp), cycle.entries.len(), "{}", label());
            }
            irreducible += usize::from(forest.loops().any(|lp| forest.is_irreducible(lp)));
            if mask % 61 == 0 {
                assert_answers(&forest, count, &label());
            }
        }
    }
    assert!(irreducible > 100_000, "{irreducible} irreducible graphs");
}

/// A graph of up to sixteen blocks with random edges of random kinds: self
/// loops, parallel edges, blocks the root misses and, in a quarter of
/// them, a removed block.
fn random_graph(seed: u64) -> Cfg<u32> {
    let kinds = [
        EdgeKind::Fallthrough,
        EdgeKind::Back,
        EdgeKind::ConditionalTrue,
        EdgeKind::Jump,
    ];
    let mut rng = Rng::new(seed);
    let count = 4 + rng.below(13);
    let edges: Vec<_> = (0..count + rng.below(count * 2))
        .map(|_| {
            (
                rng.below(count),
                rng.below(count),
                kinds[rng.below(kinds.len())],
            )
        })
        .collect();
    let (mut cfg, blocks) = graph_of(count, &edges);
    if rng.one_in(4) {
        cfg.remove_block(blocks[1 + rng.below(count - 1)]);
    }
    cfg
}

#[test]
fn random_graphs_have_the_forest_of_the_definition() {
    let (mut irreducible, mut reducible, mut deep) = (0, 0, 0);
    for seed in 0..20_000_u64 {
        let cfg = random_graph(seed);
        let label = format!("seed {seed}");

        let forest = LoopForest::compute_cycles(&cfg);

        assert_eq!(cycles_of(&forest), brute_force(&cfg), "{label}");
        assert_answers(&forest, cfg.node_bound(), &label);
        let dominators = DominatorTree::compute(&cfg);
        if is_reducible(&cfg, &dominators) {
            reducible += 1;
            assert_eq!(
                cycles_of(&forest),
                cycles_of(&LoopForest::compute(&cfg, &dominators)),
                "{label}: the natural loops of a reducible graph"
            );
        }
        irreducible += usize::from(forest.loops().any(|lp| forest.is_irreducible(lp)));
        deep += usize::from(forest.max_nesting_depth() >= 3);
    }
    // The generator has to reach the shapes the comparison is for.
    assert!(irreducible > 2_000, "{irreducible} irreducible graphs");
    assert!(reducible > 2_000, "{reducible} reducible graphs");
    assert!(deep > 1_000, "{deep} graphs three loops deep");
}

#[test]
fn on_every_small_reducible_digraph_the_cycles_are_the_natural_loops() {
    let mut reducible = 0_usize;
    for count in 1..=4_usize {
        for mask in 0..1_u32 << (count * count) {
            let mut graph = Bits {
                successors: [0; 8],
                count,
            };
            for bit in (0..count * count).filter(|&bit| mask >> bit & 1 == 1) {
                graph.successors[bit / count] |= 1 << (bit % count);
            }
            let dominators = DominatorTree::compute(&graph);
            if is_reducible(&graph, &dominators) {
                reducible += 1;
                assert_eq!(
                    cycles_of(&LoopForest::compute_cycles(&graph)),
                    cycles_of(&LoopForest::compute(&graph, &dominators)),
                    "{count} blocks, successors {:?}",
                    graph.successors
                );
            }
        }
    }
    assert!(reducible > 10_000, "{reducible} reducible digraphs");
}

#[test]
fn the_dominance_forests_list_the_entries_of_their_loops() {
    let mut irreducible = 0_usize;
    for seed in 0..20_000_u64 {
        let cfg = random_graph(seed);
        let dominators = DominatorTree::compute(&cfg);
        for forest in [
            LoopForest::compute(&cfg, &dominators),
            LoopForest::compute_tagged(&cfg, &dominators),
        ] {
            for lp in forest.loops() {
                let header = forest.header(lp);
                let body: BTreeSet<BlockId> = forest.body(lp).collect();
                let entered = body.iter().copied().filter(|&block| {
                    block != header
                        && (block == cfg.entry()
                            || cfg.predecessors(block).any(|before| {
                                dominators.is_reachable(before) && !body.contains(&before)
                            }))
                });
                let entries: Vec<BlockId> = core::iter::once(header).chain(entered).collect();
                assert_eq!(forest.entry_count(lp), entries.len(), "seed {seed}");
                assert_eq!(forest.entries(lp), entries, "seed {seed}");
                assert_eq!(
                    forest.is_irreducible(lp),
                    !forest.is_nested(lp),
                    "seed {seed}"
                );
                irreducible += usize::from(forest.is_irreducible(lp));
            }
        }
    }
    assert!(irreducible > 1_000, "{irreducible} irreducible loops");
}

/// Loops nested `depth` deep, whose walk is twice as deep. Header `i`
/// enters header `i + 1` and its latch `i`, and latch `i + 1` returns to
/// latch `i`, so latch `i` reaches header `i` through every loop inside.
/// With `irreducible`, header `i` also enters latch `i + 1`, a second
/// entry of every loop but the outermost.
fn deep_nest(depth: usize, irreducible: bool) -> (Cfg<u32>, Vec<BlockId>, Vec<BlockId>) {
    let mut cfg = Cfg::<u32>::new();
    let headers: Vec<BlockId> = core::iter::once(cfg.entry())
        .chain((1..depth).map(|_| cfg.new_block()))
        .collect();
    let latches: Vec<BlockId> = (0..depth).map(|_| cfg.new_block()).collect();
    let exit = cfg.new_block();
    for index in 0..depth {
        if index + 1 < depth {
            cfg.add_edge(
                headers[index],
                headers[index + 1],
                EdgeKind::ConditionalTrue,
            );
            if irreducible {
                cfg.add_edge(headers[index], latches[index + 1], EdgeKind::Jump);
            }
            cfg.add_edge(latches[index + 1], latches[index], EdgeKind::Fallthrough);
        }
        cfg.add_edge(headers[index], latches[index], EdgeKind::ConditionalFalse);
        cfg.add_edge(latches[index], headers[index], EdgeKind::Jump);
    }
    cfg.add_edge(latches[0], exit, EdgeKind::ConditionalFalse);
    (cfg, headers, latches)
}

#[test]
fn deep_nests_need_no_recursion_and_cost_the_forest_only_its_size() {
    let depth = 100_000_usize;
    for irreducible in [false, true] {
        let (cfg, headers, latches) = deep_nest(depth, irreducible);

        let forest = LoopForest::compute_cycles(&cfg);

        assert_eq!(forest.len(), depth);
        assert_eq!(forest.max_nesting_depth(), depth);
        let outermost = forest
            .loop_of_header(headers[0])
            .expect("the entry heads a loop");
        let innermost = forest
            .loop_of_header(headers[depth - 1])
            .expect("the last header heads a loop");
        assert_eq!(forest.parent(outermost), None);
        assert_eq!(forest.depth(innermost), depth - 1);
        assert_eq!(
            forest.parent(innermost),
            forest.loop_of_header(headers[depth - 2])
        );
        assert_eq!(forest.body_len(outermost), 2 * depth);
        assert_eq!(forest.body_len(innermost), 2);
        assert_eq!(forest.latches(innermost), [latches[depth - 1]]);
        assert!(forest.contains(outermost, latches[depth - 1]));
        assert!(!forest.contains(innermost, headers[depth - 2]));
        assert_eq!(forest.entries(outermost), [headers[0]]);
        let second_entry = if irreducible {
            vec![headers[depth - 1], latches[depth - 1]]
        } else {
            vec![headers[depth - 1]]
        };
        assert_eq!(forest.entries(innermost), second_entry);
        assert_eq!(
            forest
                .loops()
                .filter(|&lp| forest.is_irreducible(lp))
                .count(),
            if irreducible { depth - 1 } else { 0 }
        );
    }
}
