//! The dispatch transformation against its contract: a reducible result, the
//! blocks it reports and no others, the builder calls in the documented order,
//! and the traces of the original graph when selectors and hubs are run.

use alloc::collections::BTreeMap;
use alloc::format;

use super::*;
use crate::graph::dominator::DominatorTree;
use crate::graph::structure::is_reducible;
use crate::test_util::rng::Rng;

/// A toy instruction: a block of the original graph, by number, or the code
/// the builder writes, which names the hub whose selector it sets or reads.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Inst {
    Original(usize),
    Select { hub: BlockId, index: usize },
    Dispatch { hub: BlockId },
}

/// A toy edge payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum Payload {
    #[default]
    Original,
    Case(usize),
    Jump,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Call {
    Dispatch(BlockId, Vec<BlockId>),
    Case(BlockId, usize),
    Select(BlockId, usize, BlockId),
    Jump(BlockId, BlockId),
}

/// A builder that records its calls.
#[derive(Default)]
struct Recorder {
    calls: Vec<Call>,
}

impl DispatchBuilder<Inst, Payload> for Recorder {
    fn select(&mut self, hub: BlockId, index: usize, entry: BlockId) -> Vec<Inst> {
        self.calls.push(Call::Select(hub, index, entry));
        vec![Inst::Select { hub, index }]
    }

    fn dispatch(&mut self, hub: BlockId, entries: &[BlockId]) -> Vec<Inst> {
        self.calls.push(Call::Dispatch(hub, entries.to_vec()));
        vec![Inst::Dispatch { hub }]
    }

    fn case(&mut self, hub: BlockId, index: usize) -> Payload {
        self.calls.push(Call::Case(hub, index));
        Payload::Case(index)
    }

    fn jump(&mut self, selector: BlockId, hub: BlockId) -> Payload {
        self.calls.push(Call::Jump(selector, hub));
        Payload::Jump
    }
}

/// Block `i` holds `Original(i)`, and the edges come in the order given.
fn graph_of(count: usize, edges: &[(usize, usize)]) -> Cfg<Inst, Payload> {
    let mut cfg = Cfg::with_edge_payload();
    let mut blocks = vec![cfg.entry()];
    blocks.extend((1..count).map(|_| cfg.new_block()));
    for (index, &block) in blocks.iter().enumerate() {
        cfg.block_mut(block)
            .instructions_mut()
            .push(Inst::Original(index));
    }
    for &(from, to) in edges {
        cfg.add_edge(blocks[from], blocks[to], EdgeKind::Jump);
    }
    cfg
}

/// The greatest number of irreducible cycles any block is in.
fn irreducible_depth(forest: &LoopForest) -> usize {
    forest
        .loops()
        .map(|lp| {
            core::iter::successors(Some(lp), |&inner| forest.parent(inner))
                .filter(|&around| forest.is_irreducible(around))
                .count()
        })
        .max()
        .unwrap_or(0)
}

/// Walk both graphs along the same random choices. At a block of the
/// original graph both take the same edge; in the dispatched graph a selector
/// sets the selector of its hub and a hub follows its own. The blocks of the
/// original graph the walks reach must be the same.
fn assert_same_traces(
    original: &Cfg<Inst, Payload>,
    dispatched: &Cfg<Inst, Payload>,
    rng: &mut Rng,
    label: &str,
) {
    for _ in 0..8 {
        let mut here = original.entry();
        let mut selectors: BTreeMap<BlockId, usize> = BTreeMap::new();
        for _ in 0..40 {
            let edges: Vec<EdgeId> = original.outgoing(here).collect();
            assert_eq!(
                dispatched.outgoing(here).collect::<Vec<_>>(),
                edges,
                "{label}: the edges out of {here:?} keep their identity and order"
            );
            let Some(&edge) = edges.get(rng.below(edges.len().max(1))) else {
                break;
            };
            let mut there = dispatched.edge(edge).target();
            here = original.edge(edge).target();
            loop {
                match *dispatched.block(there).instructions() {
                    [Inst::Original(_)] => break,
                    [Inst::Select { hub, index }] => {
                        selectors.insert(hub, index);
                        let [next] = dispatched.successors(there).collect::<Vec<_>>()[..] else {
                            panic!("{label}: a selector has one successor");
                        };
                        there = next;
                    }
                    [Inst::Dispatch { hub }] => {
                        let index = selectors[&hub];
                        there = dispatched
                            .outgoing(there)
                            .map(|edge| dispatched.edge(edge))
                            .find(|edge| *edge.payload() == Payload::Case(index))
                            .expect("the hub has a case for every index its selectors set")
                            .target();
                    }
                    ref other => panic!("{label}: unexpected code {other:?}"),
                }
            }
            assert_eq!(there, here, "{label}: the walks part");
        }
    }
}

/// Dispatch one digraph and check the result without reading the
/// transformation. Returns whether the digraph was irreducible.
fn check_digraph(count: usize, edges: &[(usize, usize)], seed: u64) -> bool {
    let original = graph_of(count, edges);
    let label = format!("{count} blocks, edges {edges:?}");
    let forest = LoopForest::compute_cycles(&original);
    let irreducible = forest
        .loops()
        .filter(|&lp| forest.is_irreducible(lp))
        .count();
    let reached_edges = {
        let reached = reachable(&original, [original.entry()], Direction::Forward);
        edges.iter().filter(|&&(from, _)| reached[from]).count()
    };

    let mut cfg = original.clone();
    let mut recorder = Recorder::default();
    let report = make_reducible_by_dispatch(&mut cfg, &mut recorder);

    assert!(is_reducible(&cfg, &DominatorTree::compute(&cfg)), "{label}");
    assert_eq!(report.hubs.len(), irreducible, "{label}");
    assert!(
        report.selectors.len() <= reached_edges * irreducible_depth(&forest),
        "{label}: {} selectors",
        report.selectors.len()
    );
    let added: Vec<BlockId> = cfg
        .block_ids()
        .filter(|block| block.index() >= original.block_bound())
        .collect();
    let mut reported: Vec<BlockId> = report
        .hubs
        .iter()
        .chain(&report.selectors)
        .copied()
        .collect();
    reported.sort_unstable();
    assert_eq!(
        added, reported,
        "{label}: the added blocks are the reported ones"
    );

    let mut calls = recorder.calls.iter().peekable();
    let mut selectors = report.selectors.iter();
    for &hub in &report.hubs {
        let Some(Call::Dispatch(at, entries)) = calls.next() else {
            panic!("{label}: a cycle starts with its dispatch");
        };
        assert_eq!(*at, hub, "{label}");
        assert_eq!(cfg.block(hub).instructions(), [Inst::Dispatch { hub }]);
        assert!(
            entries.len() > 1,
            "{label}: a hub dispatches several entries"
        );
        assert!(entries[1..].is_sorted(), "{label}");
        assert_eq!(
            cfg.successors(hub).collect::<Vec<_>>(),
            *entries,
            "{label}: the hub's cases"
        );
        for index in 0..entries.len() {
            assert_eq!(calls.next(), Some(&Call::Case(hub, index)), "{label}");
        }
        let mut previous = 0;
        while let Some(&&Call::Select(at, index, entry)) = calls.peek()
            && at == hub
        {
            calls.next();
            assert!(previous <= index, "{label}: selectors in entry order");
            previous = index;
            assert_eq!(entry, entries[index], "{label}");
            let &selector = selectors.next().expect("every selector is reported");
            assert_eq!(calls.next(), Some(&Call::Jump(selector, hub)), "{label}");
            assert_eq!(
                cfg.block(selector).instructions(),
                [Inst::Select { hub, index }]
            );
            let [out] = cfg.outgoing(selector).collect::<Vec<_>>()[..] else {
                panic!("{label}: a selector has one edge out");
            };
            assert_eq!(cfg.edge(out).kind(), EdgeKind::Jump, "{label}");
            assert_eq!(*cfg.edge(out).payload(), Payload::Jump, "{label}");
        }
    }
    assert_eq!(calls.next(), None, "{label}: no call outside a cycle");
    assert_eq!(selectors.next(), None, "{label}");

    assert_same_traces(&original, &cfg, &mut Rng::new(seed), &label);
    irreducible > 0
}

/// Every digraph of up to four blocks, then random ones of five to nine, and
/// random ones of up to sixteen with parallel edges.
fn digraphs() -> impl Iterator<Item = (usize, Vec<(usize, usize)>)> {
    let exhaustive = (1..=4_usize).flat_map(|count| {
        let pairs: Vec<(usize, usize)> = (0..count)
            .flat_map(|from| (0..count).map(move |to| (from, to)))
            .collect();
        (0..1_u32 << pairs.len()).map(move |mask| {
            let edges = pairs
                .iter()
                .enumerate()
                .filter(|&(bit, _)| mask >> bit & 1 == 1)
                .map(|(_, &pair)| pair)
                .collect();
            (count, edges)
        })
    });
    let random = (0..5_000_u64).map(|seed| {
        let mut rng = Rng::new(seed);
        let count = 5 + rng.below(5);
        let percent = 150 / count + rng.below(300 / count);
        let edges = (0..count)
            .flat_map(|from| (0..count).map(move |to| (from, to)))
            .filter(|_| rng.chance(percent))
            .collect();
        (count, edges)
    });
    let parallel = (0..2_000_u64).map(|seed| {
        let mut rng = Rng::new(!seed);
        let count = 4 + rng.below(13);
        let edges = (0..count + rng.below(count * 2))
            .map(|_| (rng.below(count), rng.below(count)))
            .collect();
        (count, edges)
    });
    exhaustive.chain(random).chain(parallel)
}

#[test]
fn a_removed_block_is_no_block_of_the_result() {
    // The irreducible core, and a block removed before the dispatch.
    let mut cfg = graph_of(4, &[(0, 1), (0, 2), (1, 2), (2, 1), (2, 3)]);
    cfg.remove_block(BlockId::from_index(3));
    let bound = cfg.block_bound();

    let report = make_reducible_by_dispatch(&mut cfg, &mut Recorder::default());

    assert!(is_reducible(&cfg, &DominatorTree::compute(&cfg)));
    assert_eq!(report.hubs.len(), 1);
    assert_eq!(report.selectors.len(), 4);
    assert!(!cfg.contains_block(BlockId::from_index(3)));
    assert!(
        report
            .hubs
            .iter()
            .chain(&report.selectors)
            .all(|block| block.index() >= bound)
    );
}

#[test]
fn every_small_digraph_becomes_reducible_with_the_same_traces() {
    let mut irreducible = 0_usize;
    for (seed, (count, edges)) in (0_u64..).zip(digraphs()) {
        irreducible += usize::from(check_digraph(count, &edges, seed));
    }
    assert!(
        irreducible > 5_000,
        "only {irreducible} irreducible digraphs"
    );
}

#[test]
fn a_reducible_graph_is_left_alone() {
    // A loop with a branch inside, and a block the root does not reach that
    // jumps into the loop.
    let original = graph_of(5, &[(0, 1), (1, 2), (1, 3), (2, 1), (3, 1), (4, 2)]);
    let mut cfg = original.clone();
    let mut recorder = Recorder::default();

    let report = make_reducible_by_dispatch(&mut cfg, &mut recorder);

    assert_eq!(report, DispatchReport::default());
    assert_eq!(recorder.calls, []);
    assert_eq!(cfg.block_count(), original.block_count());
    assert_eq!(cfg.edge_count(), original.edge_count());
}

#[test]
fn a_loop_an_entry_heads_inside_the_cycle_stays_a_loop() {
    // `a` heads the cycle; `b` is entered from the root too, and heads the
    // loop of `b` and `c` inside it, which `c` closes and leaves for `a`.
    let original = graph_of(4, &[(0, 1), (0, 2), (1, 2), (2, 3), (3, 2), (3, 1)]);
    let (b, c) = (BlockId::from_index(2), BlockId::from_index(3));
    let mut cfg = original.clone();

    let report = make_reducible_by_dispatch(&mut cfg, &mut Recorder::default());

    assert!(is_reducible(&cfg, &DominatorTree::compute(&cfg)));
    assert_eq!(report.hubs.len(), 1);
    // The latch `c` still jumps straight back to `b`.
    let dominators = DominatorTree::compute(&cfg);
    let loops = LoopForest::compute(&cfg, &dominators);
    let inner = loops.loop_of_header(b).expect("b still heads a loop");
    assert_eq!(loops.latches(inner), [c]);
    assert_eq!(loops.parent(inner), loops.loop_of_header(report.hubs[0]));
}

#[test]
fn an_edge_from_unreached_code_is_left_alone() {
    // The irreducible core, and a block the root misses that jumps to `b`.
    let original = graph_of(4, &[(0, 1), (0, 2), (1, 2), (2, 1), (3, 2)]);
    let (b, dead) = (BlockId::from_index(2), BlockId::from_index(3));
    let mut cfg = original.clone();

    let report = make_reducible_by_dispatch(&mut cfg, &mut Recorder::default());

    assert_eq!(report.selectors.len(), 4);
    assert_eq!(cfg.successors(dead).collect::<Vec<_>>(), [b]);
}

/// Every block jumps to every other, and the root to each of them: a cycle
/// at every level of the forest, every one irreducible. This is the family on
/// which node splitting must grow exponentially.
fn complete(blocks: usize) -> Vec<(usize, usize)> {
    (0..=blocks)
        .flat_map(|from| (1..=blocks).map(move |to| (from, to)))
        .filter(|&(from, to)| from != to)
        .collect()
}

#[test]
fn dispatch_grows_with_the_edges_per_level_where_splitting_grows_exponentially() {
    for blocks in 3..=7_usize {
        let edges = complete(blocks);
        let original = graph_of(blocks + 1, &edges);
        let levels = blocks - 1;

        let mut dispatched = original.clone();
        let report = make_reducible_by_dispatch(&mut dispatched, &mut Recorder::default());
        let mut split = original.clone();
        let copies = super::super::make_reducible(&mut split).len();

        assert!(is_reducible(
            &dispatched,
            &DominatorTree::compute(&dispatched)
        ));
        assert_eq!(report.hubs.len(), levels);
        // An edge between two blocks enters the cycles of every level between
        // them, and gets one selector on each.
        assert!(
            report.selectors.len() <= edges.len() * levels,
            "{blocks} blocks: {} selectors for {} edges",
            report.selectors.len(),
            edges.len()
        );
        assert!(
            copies >= 1 << levels,
            "{blocks} blocks: splitting copied only {copies}"
        );
    }
}

#[test]
fn a_deep_irreducible_nest_grows_linearly() {
    // Header `i` enters header `i + 1`, its latch `i` and latch `i + 1`; the
    // latches return to their headers and to the latch around them.
    let depth = 2_000_usize;
    let header = |index: usize| 1 + index;
    let latch = |index: usize| 1 + depth + index;
    let mut edges = vec![(0, header(0))];
    for index in 0..depth {
        if index + 1 < depth {
            edges.push((header(index), header(index + 1)));
            edges.push((header(index), latch(index + 1)));
            edges.push((latch(index + 1), latch(index)));
        }
        edges.push((header(index), latch(index)));
        edges.push((latch(index), header(index)));
    }
    let mut cfg = graph_of(1 + 2 * depth, &edges);

    let report = make_reducible_by_dispatch(&mut cfg, &mut Recorder::default());

    assert!(is_reducible(&cfg, &DominatorTree::compute(&cfg)));
    assert_eq!(report.hubs.len(), depth - 1);
    assert!(
        report.selectors.len() <= 3 * edges.len(),
        "{} selectors for {} edges",
        report.selectors.len(),
        edges.len()
    );
}
