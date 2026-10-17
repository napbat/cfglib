//! Loop nesting depth per block, judged by brute-force dominance.

use std::collections::BTreeSet;

use cfglib::{
    BlockId, Cfg, CfgMetrics, EdgeKind, GraphMetrics, block_nesting_depths,
    cfg_block_nesting_depths,
};

use crate::random::Rng;
use crate::stmt::Stmt;

struct Shape {
    cfg: Cfg<Stmt>,
    successors: Vec<Vec<usize>>,
    /// Edges the builder tagged as back edges.
    tagged: BTreeSet<(usize, usize)>,
}

/// A graph of `size` blocks reachable through the chain `0 -> 1 -> ...`, with
/// random extra edges in either direction, some backward ones tagged `Back`.
fn random_shape(seed: u64) -> Shape {
    let mut rng = Rng::new(seed);
    let size = 5 + rng.below(5);
    let mut cfg = Cfg::<Stmt>::new();
    let blocks: Vec<BlockId> = std::iter::once(cfg.entry())
        .chain((1..size).map(|_| cfg.new_block()))
        .collect();
    let mut successors = vec![Vec::new(); size];
    let mut tagged = BTreeSet::new();
    for source in 0..size {
        if source + 1 < size {
            cfg.add_edge(blocks[source], blocks[source + 1], EdgeKind::Fallthrough);
            successors[source].push(source + 1);
        }
        for _ in 0..rng.below(3) {
            let target = rng.below(size);
            let backward = target <= source;
            let kind = if backward && rng.chance(50) {
                EdgeKind::Back
            } else {
                EdgeKind::ConditionalTrue
            };
            cfg.add_edge(blocks[source], blocks[target], kind);
            successors[source].push(target);
            if kind == EdgeKind::Back {
                tagged.insert((source, target));
            }
        }
    }
    Shape {
        cfg,
        successors,
        tagged,
    }
}

/// Nodes reachable from `from` without entering `avoid`.
fn reach(successors: &[Vec<usize>], from: usize, avoid: Option<usize>) -> BTreeSet<usize> {
    let mut seen = BTreeSet::from([from]);
    let mut pending = vec![from];
    while let Some(current) = pending.pop() {
        for &next in &successors[current] {
            if Some(next) != avoid && seen.insert(next) {
                pending.push(next);
            }
        }
    }
    seen
}

/// `candidate` dominates `node` when removing it cuts `node` off from the entry
/// (every block here is reachable, and the entry dominates all of them).
fn dominates(successors: &[Vec<usize>], candidate: usize, node: usize) -> bool {
    candidate == node || candidate == 0 || !reach(successors, 0, Some(candidate)).contains(&node)
}

/// How many natural loops contain each block, by the textbook definition: a
/// back edge `tail -> header` is one whose header dominates its tail (or one
/// the builder tagged); a loop is the header plus everything that reaches a
/// tail without passing the header; loops with one header merge.
fn loop_counts(shape: &Shape, honor_tags: bool) -> Vec<usize> {
    let size = shape.successors.len();
    let mut counts = vec![0; size];
    for header in 0..size {
        let tails: BTreeSet<usize> = (0..size)
            .filter(|&tail| {
                shape.successors[tail].contains(&header)
                    && (dominates(&shape.successors, header, tail)
                        || (honor_tags && shape.tagged.contains(&(tail, header))))
            })
            .collect();
        if tails.is_empty() {
            continue;
        }
        let mut body = BTreeSet::from([header]);
        let mut pending: Vec<usize> = tails
            .iter()
            .copied()
            .filter(|&tail| body.insert(tail))
            .collect();
        while let Some(current) = pending.pop() {
            for predecessor in 0..size {
                if shape.successors[predecessor].contains(&current) && body.insert(predecessor) {
                    pending.push(predecessor);
                }
            }
        }
        for member in body {
            counts[member] += 1;
        }
    }
    counts
}

#[test]
fn block_depths_count_the_natural_loops_containing_each_block() {
    let mut looping = 0;
    for seed in 0..150 {
        let shape = random_shape(seed);
        let expected = loop_counts(&shape, false);
        assert_eq!(block_nesting_depths(&shape.cfg), expected, "seed {seed}");
        looping += usize::from(expected.iter().any(|&depth| depth > 0));
    }
    assert!(looping > 40, "only {looping} of the graphs had a loop");
}

/// Whether every back edge the builder tagged closes onto a header that
/// dominates its tail: the only case with a textbook natural loop.
fn tags_are_natural(shape: &Shape) -> bool {
    shape
        .tagged
        .iter()
        .all(|&(tail, header)| dominates(&shape.successors, header, tail))
}

#[test]
fn tags_on_natural_back_edges_add_nothing_to_the_dominance_loops() {
    let mut natural = 0;
    for seed in 0..150 {
        let shape = random_shape(seed);
        if !tags_are_natural(&shape) {
            continue;
        }
        natural += 1;
        let tagged = cfg_block_nesting_depths(&shape.cfg);
        assert_eq!(tagged, loop_counts(&shape, true), "seed {seed}");
        assert_eq!(
            tagged,
            block_nesting_depths(&shape.cfg),
            "seed {seed}: dominance already finds these loops"
        );
    }
    assert!(natural > 20, "only {natural} graphs had natural tags only");
}

#[test]
fn a_tagged_edge_into_a_non_dominating_header_still_puts_both_ends_inside_a_loop() {
    let mut irreducible = 0;
    for seed in 0..150 {
        let shape = random_shape(seed);
        let tagged = cfg_block_nesting_depths(&shape.cfg);
        for &(tail, header) in &shape.tagged {
            assert!(tagged[header] >= 1, "seed {seed}: header bb{header}");
            assert!(tagged[tail] >= 1, "seed {seed}: latch bb{tail}");
        }
        irreducible += usize::from(!tags_are_natural(&shape));
    }
    assert!(
        irreducible > 5,
        "no random graph had an irreducible tagged edge: {irreducible}"
    );
}

#[test]
fn under_dominance_the_entry_block_is_inside_only_the_loop_it_heads() {
    for seed in 0..150 {
        let shape = random_shape(seed);
        let size = shape.successors.len();
        let entry_heads_a_loop = (0..size).any(|tail| {
            shape.successors[tail].contains(&0) && dominates(&shape.successors, 0, tail)
        });
        // Loops with one header merge, so the entry heads at most one.
        let allowed = usize::from(entry_heads_a_loop);

        assert!(
            block_nesting_depths(&shape.cfg)[0] <= allowed,
            "seed {seed}: the entry dominates every block, so no other header can enclose it"
        );
    }
}

#[test]
fn nested_loops_stack_their_depths_and_blocks_outside_every_loop_are_zero() {
    let mut cfg = Cfg::<Stmt>::new();
    let entry = cfg.entry();
    let (outer, inner, inner_body, outer_latch, exit) = (
        cfg.new_block(),
        cfg.new_block(),
        cfg.new_block(),
        cfg.new_block(),
        cfg.new_block(),
    );
    cfg.add_edge(entry, outer, EdgeKind::Fallthrough);
    cfg.add_edge(outer, inner, EdgeKind::ConditionalTrue);
    cfg.add_edge(outer, exit, EdgeKind::ConditionalFalse);
    cfg.add_edge(inner, inner_body, EdgeKind::ConditionalTrue);
    cfg.add_edge(inner, outer_latch, EdgeKind::ConditionalFalse);
    cfg.add_edge(inner_body, inner, EdgeKind::Back);
    cfg.add_edge(outer_latch, outer, EdgeKind::Back);

    let depths = block_nesting_depths(&cfg);

    assert_eq!(depths[entry.index()], 0);
    assert_eq!(depths[outer.index()], 1);
    assert_eq!(depths[outer_latch.index()], 1);
    assert_eq!(depths[inner.index()], 2);
    assert_eq!(depths[inner_body.index()], 2);
    assert_eq!(depths[exit.index()], 0);
    assert_eq!(
        cfg_block_nesting_depths(&cfg),
        depths,
        "no irreducible tags here"
    );
}

#[test]
fn the_depth_vector_covers_the_node_bound_and_removed_blocks_are_outside_every_loop() {
    let mut cfg = Cfg::<Stmt>::new();
    let entry = cfg.entry();
    let (dropped, header, exit) = (cfg.new_block(), cfg.new_block(), cfg.new_block());
    cfg.add_edge(entry, header, EdgeKind::Fallthrough);
    cfg.add_edge(header, header, EdgeKind::Back);
    cfg.add_edge(header, exit, EdgeKind::ConditionalFalse);
    cfg.add_edge(dropped, header, EdgeKind::Fallthrough);
    cfg.remove_block(dropped);

    let depths = block_nesting_depths(&cfg);

    assert_eq!(depths.len(), cfg.block_bound());
    assert_eq!(
        depths,
        [0, 0, 1, 0],
        "the self-loop is a one-block loop; the removed slot is 0"
    );
    assert_eq!(cfg_block_nesting_depths(&cfg), depths);
}

#[test]
fn metrics_report_the_deepest_nest_the_block_depths_show() {
    for seed in 0..150 {
        let shape = random_shape(seed);
        let deepest_tagged = cfg_block_nesting_depths(&shape.cfg)
            .into_iter()
            .max()
            .unwrap();
        let deepest_untagged = block_nesting_depths(&shape.cfg).into_iter().max().unwrap();

        assert_eq!(
            CfgMetrics::compute(&shape.cfg).graph.max_nesting_depth,
            deepest_tagged,
            "seed {seed}: 0 means no loops, so one loop is depth 1"
        );
        assert_eq!(
            GraphMetrics::compute(&shape.cfg).max_nesting_depth,
            deepest_untagged,
            "seed {seed}"
        );
    }
}
