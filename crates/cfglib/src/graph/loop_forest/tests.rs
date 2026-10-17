//! The loop forest against the loop builder it replaced, which survives here as
//! the oracle: a set of blocks per loop, found by a reverse walk from the
//! latches, nested by looking each body block up among the headers.

use super::*;
use crate::cfg::Cfg;
use crate::edge::EdgeKind;
use crate::graph::loop_nest::nesting;
use crate::test_util::rng::Rng;
use alloc::vec;

/// Compute the merged natural loop body for every back-edge to one header.
///
/// The body is the set of reachable nodes that can reach any latch without
/// going through `header`, plus `header` itself. A single multi-source
/// reverse walk computes the same union without revisiting a shared body for
/// every latch.
///
/// When the header dominates every latch, that walk cannot leave the region
/// the header dominates. A tagged latch the header does not dominate (an
/// irreducible cycle) lets the walk run on to blocks that merely lead to the
/// cycle, such as the entry, so the body is cut down to the walked blocks the
/// header reaches: the blocks on a path from the header to a latch.
fn oracle_body_for<G: GraphView>(
    graph: &G,
    dom: &DominatorTree<G::NodeId>,
    header: G::NodeId,
    latches: &[G::NodeId],
) -> BTreeSet<G::NodeId> {
    let mut body = BTreeSet::new();
    body.insert(header);
    let mut stack = Vec::new();
    for &latch in latches {
        if body.insert(latch) {
            stack.push(latch);
        }
    }
    while let Some(n) = stack.pop() {
        for p in graph.predecessors(n).filter(|&p| dom.is_reachable(p)) {
            if !body.contains(&p) {
                body.insert(p);
                stack.push(p);
            }
        }
    }

    if latches.iter().all(|&latch| dom.dominates(header, latch)) {
        return body;
    }
    let mut cycle = BTreeSet::from([header]);
    cycle.extend(latches.iter().copied());
    stack.push(header);
    while let Some(n) = stack.pop() {
        for s in graph.successors(n) {
            if body.contains(&s) && cycle.insert(s) {
                stack.push(s);
            }
        }
    }
    cycle
}

/// Build merged, depth-annotated loops from a set of back-edges.
fn oracle_loops_from_backs<G: GraphView>(
    graph: &G,
    dom: &DominatorTree<G::NodeId>,
    backs: &[BackEdge<G::NodeId>],
) -> Vec<NaturalLoop<G::NodeId>> {
    if backs.is_empty() {
        return Vec::new();
    }

    // Group back-edges by header.
    let mut header_map: alloc::collections::BTreeMap<G::NodeId, Vec<G::NodeId>> =
        alloc::collections::BTreeMap::new();
    for be in backs {
        header_map.entry(be.header).or_default().push(be.tail);
    }

    let mut loops: Vec<NaturalLoop<G::NodeId>> = Vec::new();
    for (header, latches) in &header_map {
        let body = oracle_body_for(graph, dom, *header, latches);
        loops.push(NaturalLoop {
            header: *header,
            body,
            latches: latches.iter().copied().collect(),
            depth: 0, // filled in below
        });
    }

    // The nesting tree states the depth, so the flat list cannot disagree
    // with it.
    let depths = nesting(&loops).depths;
    for (lp, depth) in loops.iter_mut().zip(depths) {
        lp.depth = depth;
    }

    // Sort by depth (outermost first), then by header.
    loops.sort_by(|a, b| a.depth.cmp(&b.depth).then(a.header.cmp(&b.header)));
    loops
}

/// What `detect_loops` returned before the forest.
fn oracle_loops(cfg: &Cfg<u32>, dom: &DominatorTree) -> Vec<NaturalLoop> {
    oracle_loops_from_backs(cfg, dom, &find_back_edges(cfg, dom))
}

/// What `detect_loops_tagged` returned before the forest.
fn oracle_loops_tagged(cfg: &Cfg<u32>, dom: &DominatorTree) -> Vec<NaturalLoop> {
    oracle_loops_from_backs(cfg, dom, &find_back_edges_tagged(cfg, dom))
}

/// What `detect_all_loops_tagged` returned before the forest: the tagged loops,
/// then those of the dominator forest whose header the entry misses.
fn oracle_all_loops_tagged(cfg: &Cfg<u32>, dom: &DominatorTree) -> Vec<NaturalLoop> {
    let mut loops = oracle_loops_tagged(cfg, dom);
    let mut roots = Vec::new();
    let Some(forest) =
        complete_dominator_forest(&mut DominatorScratch::new(), &mut roots, cfg, dom)
    else {
        return loops;
    };
    loops.extend(
        oracle_loops_tagged(cfg, &forest)
            .into_iter()
            .filter(|dead| !dom.is_reachable(dead.header)),
    );
    loops.sort_by(|a, b| a.depth.cmp(&b.depth).then(a.header.cmp(&b.header)));
    loops
}

/// What `block_nesting_depths` computed from the flat loops.
fn oracle_node_depths(bound: usize, loops: &[NaturalLoop]) -> Vec<usize> {
    let mut depths = vec![0_usize; bound];
    for lp in loops {
        for &node in &lp.body {
            if node.index() < bound {
                depths[node.index()] = depths[node.index()].max(lp.depth + 1);
            }
        }
    }
    depths
}

/// Every question the forest answers, against the flat loops the oracle found.
fn assert_forest_matches(cfg: &Cfg<u32>, forest: &LoopForest, oracle: &[NaturalLoop], label: &str) {
    assert_eq!(forest.natural_loops(), oracle, "{label}: natural loops");
    assert_eq!(forest.len(), oracle.len(), "{label}");
    assert_eq!(forest.is_empty(), oracle.is_empty(), "{label}");

    let mut by_header = oracle.to_vec();
    by_header.sort_by_key(|lp| lp.header);
    let parents = nesting(&by_header).parents;
    let bound = cfg.node_bound();
    for (index, (lp, parent)) in by_header.iter().zip(parents).enumerate() {
        let id = LoopId(word(index));
        assert_eq!(forest.loops().nth(index), Some(id), "{label}");
        assert_eq!(forest.loop_of_header(lp.header), Some(id), "{label}");
        assert_eq!(forest.header(id), lp.header, "{label}");
        assert_eq!(
            forest.latches(id),
            lp.latches.iter().copied().collect::<Vec<_>>(),
            "{label}: latches of {:?}",
            lp.header
        );
        assert_eq!(
            forest.depth(id),
            lp.depth,
            "{label}: depth of {:?}",
            lp.header
        );
        assert_eq!(
            forest.parent(id),
            parent.map(|parent| LoopId(word(parent))),
            "{label}: parent of {:?}",
            lp.header
        );
        assert_eq!(forest.body_len(id), lp.body.len(), "{label}");
        assert_eq!(
            forest.body(id).collect::<BTreeSet<_>>(),
            lp.body,
            "{label}: body of {:?}",
            lp.header
        );
        assert_eq!(
            forest.body(id).count(),
            lp.body.len(),
            "{label}: no repeats"
        );
        for block in 0..bound + 2 {
            let block = BlockId::from_index(block);
            assert_eq!(
                forest.contains(id, block),
                lp.body.contains(&block),
                "{label}: {block:?} in the loop of {:?}",
                lp.header
            );
        }
    }
    for block in 0..bound {
        let block = BlockId::from_index(block);
        if !oracle.iter().any(|lp| lp.header == block) {
            assert_eq!(forest.loop_of_header(block), None, "{label}");
        }
    }
    assert_eq!(
        forest.node_depths(),
        oracle_node_depths(bound, oracle),
        "{label}: node depths"
    );
    assert_eq!(
        forest.max_nesting_depth(),
        oracle.iter().map(|lp| lp.depth + 1).max().unwrap_or(0),
        "{label}"
    );
}

fn check_all_three(cfg: &Cfg<u32>, label: &str) {
    let dom = DominatorTree::compute(cfg);
    assert_forest_matches(
        cfg,
        &LoopForest::compute(cfg, &dom),
        &oracle_loops(cfg, &dom),
        label,
    );
    assert_forest_matches(
        cfg,
        &LoopForest::compute_tagged(cfg, &dom),
        &oracle_loops_tagged(cfg, &dom),
        label,
    );
    assert_forest_matches(
        cfg,
        &LoopForest::compute_all_tagged(cfg, &dom),
        &oracle_all_loops_tagged(cfg, &dom),
        label,
    );
}

fn graph_of(count: usize, edges: &[(usize, usize, EdgeKind)]) -> Cfg<u32> {
    let mut cfg = Cfg::<u32>::new();
    let mut blocks = vec![cfg.entry()];
    blocks.extend((1..count).map(|_| cfg.new_block()));
    for &(from, to, kind) in edges {
        cfg.add_edge(blocks[from], blocks[to], kind);
    }
    cfg
}

#[test]
fn every_digraph_of_up_to_four_blocks_has_the_loops_of_the_oracle() {
    for count in 1..=4_usize {
        let pairs: Vec<(usize, usize)> = (0..count)
            .flat_map(|from| (0..count).map(move |to| (from, to)))
            .collect();
        for mask in 0..1_u32 << pairs.len() {
            let edges: Vec<_> = pairs
                .iter()
                .enumerate()
                .filter(|&(bit, _)| mask >> bit & 1 == 1)
                .map(|(_, &(from, to))| (from, to, EdgeKind::Fallthrough))
                .collect();
            check_all_three(
                &graph_of(count, &edges),
                &alloc::format!("{count} blocks, edges {edges:?}"),
            );
        }
    }
}

#[test]
fn every_tagged_digraph_of_three_blocks_has_the_loops_of_the_oracle() {
    let pairs: Vec<(usize, usize)> = (0..3)
        .flat_map(|from| (0..3).map(move |to| (from, to)))
        .collect();
    for code in 0..3_u32.pow(9) {
        let mut rest = code;
        let mut edges = Vec::new();
        for &(from, to) in &pairs {
            match rest % 3 {
                0 => {}
                1 => edges.push((from, to, EdgeKind::Fallthrough)),
                _ => edges.push((from, to, EdgeKind::Back)),
            }
            rest /= 3;
        }
        check_all_three(&graph_of(3, &edges), &alloc::format!("edges {edges:?}"));
    }
}

#[test]
fn random_graphs_with_dead_code_and_tags_have_the_loops_of_the_oracle() {
    let kinds = [
        EdgeKind::Fallthrough,
        EdgeKind::Back,
        EdgeKind::ConditionalTrue,
        EdgeKind::Jump,
        EdgeKind::Back,
    ];
    let (mut irreducible, mut dead_loops, mut deep) = (0, 0, 0);
    for seed in 0..20_000_u64 {
        let mut rng = Rng::new(seed);
        let count = 4 + rng.below(12);
        let edges: Vec<_> = (0..count + rng.below(count * 2))
            .map(|_| {
                (
                    rng.below(count),
                    rng.below(count),
                    kinds[rng.below(kinds.len())],
                )
            })
            .collect();
        let cfg = graph_of(count, &edges);
        let label = alloc::format!("seed {seed}");
        check_all_three(&cfg, &label);

        let dom = DominatorTree::compute(&cfg);
        let forest = LoopForest::compute_all_tagged(&cfg, &dom);
        irreducible += usize::from(forest.loops().any(|lp| forest.is_explicit(lp)));
        dead_loops += usize::from(
            forest
                .loops()
                .any(|lp| !dom.is_reachable(forest.header(lp))),
        );
        deep += usize::from(forest.max_nesting_depth() >= 3);
    }
    // The generator has to reach the shapes the comparison is for.
    assert!(
        irreducible > 500,
        "{irreducible} graphs with an irreducible loop"
    );
    assert!(dead_loops > 1_000, "{dead_loops} graphs with a dead loop");
    assert!(deep > 500, "{deep} graphs three loops deep");
}

#[test]
fn a_graph_without_loops_has_an_empty_forest() {
    let cfg = graph_of(
        3,
        &[(0, 1, EdgeKind::Fallthrough), (1, 2, EdgeKind::Fallthrough)],
    );
    let forest = LoopForest::compute_tagged(&cfg, &DominatorTree::compute(&cfg));
    assert!(forest.is_empty());
    assert_eq!(forest.loops().count(), 0);
    assert_eq!(forest.max_nesting_depth(), 0);
    assert_eq!(forest.node_depths(), vec![0; 3]);
    assert_eq!(forest.natural_loops(), []);
    assert_eq!(forest.loop_of_header(cfg.entry()), None);
}

/// Loops nested `depth` deep, and the answers that need no flat body: the
/// depth of the innermost, what each loop holds, which loop is the parent.
/// `loops` flattened would hold `depth * (depth + 1) / 2` blocks, which is
/// what the forest is for, so nothing here asks for them.
#[test]
fn deeply_nested_loops_cost_the_forest_only_its_size() {
    let depth = 20_000_usize;
    let mut cfg = Cfg::<u32>::new();
    let headers: Vec<BlockId> = core::iter::once(cfg.entry())
        .chain((1..depth).map(|_| cfg.new_block()))
        .collect();
    let latches: Vec<BlockId> = (0..depth).map(|_| cfg.new_block()).collect();
    let exit = cfg.new_block();
    for index in 0..depth {
        let inward = headers.get(index + 1).copied().unwrap_or(latches[index]);
        cfg.add_edge(headers[index], inward, EdgeKind::ConditionalTrue);
        cfg.add_edge(headers[index], exit, EdgeKind::ConditionalFalse);
        if index + 1 < depth {
            cfg.add_edge(latches[index + 1], latches[index], EdgeKind::Fallthrough);
        }
        cfg.add_edge(latches[index], headers[index], EdgeKind::Back);
    }
    let dom = DominatorTree::compute(&cfg);

    let forest = LoopForest::compute_tagged(&cfg, &dom);

    assert_eq!(forest.len(), depth);
    assert_eq!(forest.max_nesting_depth(), depth);
    let outermost = forest
        .loop_of_header(headers[0])
        .expect("the entry is a header");
    let innermost = forest
        .loop_of_header(headers[depth - 1])
        .expect("the last header");
    assert_eq!(forest.parent(outermost), None);
    assert_eq!(forest.depth(innermost), depth - 1);
    assert_eq!(
        forest.parent(innermost),
        forest.loop_of_header(headers[depth - 2])
    );
    assert_eq!(forest.body_len(outermost), 2 * depth);
    assert_eq!(forest.body_len(innermost), 2);
    assert!(forest.contains(outermost, latches[depth - 1]));
    assert!(!forest.contains(innermost, headers[depth - 2]));
    assert!(!forest.contains(outermost, exit));
    let depths = forest.node_depths();
    assert_eq!(depths[headers[0].index()], 1);
    assert_eq!(depths[headers[depth - 1].index()], depth);
    assert_eq!(depths[exit.index()], 0);
}

#[test]
fn an_irreducible_loop_is_answered_from_its_own_blocks() {
    // entry branches into both `a` and `b`, which jump to each other: one cycle,
    // two entries, and a back edge only a tag can name.
    let cfg = graph_of(
        4,
        &[
            (0, 1, EdgeKind::ConditionalTrue),
            (0, 2, EdgeKind::ConditionalFalse),
            (1, 2, EdgeKind::Fallthrough),
            (2, 1, EdgeKind::Back),
            (2, 3, EdgeKind::Fallthrough),
        ],
    );
    let dom = DominatorTree::compute(&cfg);

    let forest = LoopForest::compute_tagged(&cfg, &dom);

    let [a, b] = [BlockId::from_index(1), BlockId::from_index(2)];
    let cycle = forest
        .loop_of_header(a)
        .expect("the tag makes `a` a header");
    assert!(forest.is_explicit(cycle));
    assert_eq!(
        forest.body(cycle).collect::<BTreeSet<_>>(),
        BTreeSet::from([a, b])
    );
    assert!(forest.contains(cycle, b));
    assert!(!forest.contains(cycle, cfg.entry()));
    assert_eq!(forest.node_depths(), vec![0, 1, 1, 0]);
    assert_forest_matches(
        &cfg,
        &forest,
        &oracle_loops_tagged(&cfg, &dom),
        "irreducible pair",
    );
}

/// A properly nested loop around a block can be deeper than the loop the block
/// is innermost in: the nesting rule makes a loop's parent the smallest larger
/// loop that holds its header, and when that is an irreducible cycle the parent
/// chain skips a properly nested loop that holds the block too. A block's depth
/// is the deepest loop that holds it, so it takes the whole chain of enclosing
/// loops. Found by search over random graphs, 20 in 2,000,000; here is the
/// smallest.
#[test]
fn a_block_is_as_deep_as_the_deepest_loop_around_it_not_its_innermost() {
    use EdgeKind::{Back, ConditionalTrue, Fallthrough, Jump};
    let cfg = graph_of(
        8,
        &[
            (3, 5, Back),
            (6, 3, ConditionalTrue),
            (4, 6, Back),
            (2, 0, Back),
            (3, 2, ConditionalTrue),
            (1, 2, Fallthrough),
            (0, 1, Back),
            (1, 1, ConditionalTrue),
            (3, 2, Back),
            (5, 5, ConditionalTrue),
            (1, 6, Jump),
            (4, 3, Back),
            (3, 4, Back),
            (3, 7, Fallthrough),
            (5, 0, ConditionalTrue),
        ],
    );
    check_all_three(&cfg, "chain maximum");

    let dom = DominatorTree::compute(&cfg);
    let forest = LoopForest::compute_tagged(&cfg, &dom);
    let outer_is_deeper = forest.loops().any(|lp| {
        let record = forest.record(lp);
        record.is_nested() && record.deepest > record.depth
    });
    assert!(
        outer_is_deeper,
        "the graph no longer has the shape under test"
    );
}
