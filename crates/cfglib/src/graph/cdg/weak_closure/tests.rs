use super::*;
use crate::graph::dominator::{DominatorScratch, PostDominatorTree};
use crate::graph::store::{Graph, NodeId};
use crate::test_util::digraphs::{for_each_multigraph, graph_of};
use crate::test_util::rng::Rng;
use crate::{BlockId, Cfg, EdgeKind};
use alloc::vec;

/// The successor lists of `count` nodes with `edges`.
fn successors_of(count: usize, edges: &[(usize, usize)]) -> Vec<Vec<usize>> {
    let mut successors = vec![Vec::new(); count];
    for &(from, to) in edges {
        successors[from].push(to);
    }
    successors
}

/// Whether each vertex is reachable from a seed, the seeds included.
fn reachable(successors: &[Vec<usize>], seeds: &[bool]) -> Vec<bool> {
    let mut reached = seeds.to_vec();
    let mut stack: Vec<usize> = (0..seeds.len()).filter(|&vertex| seeds[vertex]).collect();
    while let Some(vertex) = stack.pop() {
        for &next in &successors[vertex] {
            if !reached[next] {
                reached[next] = true;
                stack.push(next);
            }
        }
    }
    reached
}

/// The closure from its definition: a vertex decides when two of the simple
/// paths that leave it for a seed, passing no other seed, share no vertex
/// but it. Each path is the bit set of its vertices after the first.
fn closure_by_paths(successors: &[Vec<usize>], seeds: &[bool]) -> Vec<usize> {
    /// Collect every simple seed-path that extends the one ending at
    /// `vertex` and visiting `visited`, the start included.
    fn extend(
        successors: &[Vec<usize>],
        seeds: &[bool],
        vertex: usize,
        visited: u64,
        start: usize,
        paths: &mut Vec<u64>,
    ) {
        for &next in &successors[vertex] {
            let bit = 1 << next;
            if visited & bit != 0 {
                continue;
            }
            if seeds[next] {
                paths.push((visited | bit) & !(1 << start));
            } else {
                extend(successors, seeds, next, visited | bit, start, paths);
            }
        }
    }

    assert!(successors.len() <= 64, "a path is a 64-bit set");
    let reached = reachable(successors, seeds);
    let mut paths = Vec::new();
    (0..successors.len())
        .filter(|&vertex| {
            if seeds[vertex] {
                return true;
            }
            if !reached[vertex] {
                return false;
            }
            paths.clear();
            extend(successors, seeds, vertex, 1 << vertex, vertex, &mut paths);
            paths
                .iter()
                .enumerate()
                .any(|(at, &first)| paths[at + 1..].iter().any(|&second| first & second == 0))
        })
        .collect()
}

/// The closure by Danicic's algorithm, adding at each round the origin of
/// every critical edge: an edge from a reachable vertex that observes two
/// vertices of the set to one that observes one.
fn closure_by_critical_edges(successors: &[Vec<usize>], seeds: &[bool]) -> Vec<usize> {
    let count = successors.len();
    let mut predecessors = vec![Vec::new(); count];
    for (from, targets) in successors.iter().enumerate() {
        for &to in targets {
            predecessors[to].push(from);
        }
    }
    let reached = reachable(successors, seeds);
    let mut set = seeds.to_vec();
    loop {
        // Up to two of the members of the set each vertex observes: those
        // its paths reach before any other member.
        let mut observed: Vec<Vec<usize>> = vec![Vec::new(); count];
        for member in (0..count).filter(|&vertex| set[vertex]) {
            observed[member].push(member);
            let mut seen = vec![false; count];
            let mut stack = vec![member];
            while let Some(vertex) = stack.pop() {
                for &before in &predecessors[vertex] {
                    if !set[before] && !seen[before] {
                        seen[before] = true;
                        if observed[before].len() < 2 {
                            observed[before].push(member);
                        }
                        stack.push(before);
                    }
                }
            }
        }
        let critical: Vec<usize> = (0..count)
            .filter(|&vertex| {
                reached[vertex]
                    && observed[vertex].len() == 2
                    && successors[vertex]
                        .iter()
                        .any(|&next| observed[next].len() == 1)
            })
            .collect();
        if critical.is_empty() {
            return (0..count).filter(|&vertex| set[vertex]).collect();
        }
        for vertex in critical {
            set[vertex] = true;
        }
    }
}

/// The closure `weak_control_closure` gives for `seeds`, by index.
fn closure(
    graph: &Graph<(), ()>,
    seeds: impl IntoIterator<Item = usize>,
    scratch: &mut ControlClosureScratch,
) -> Vec<usize> {
    let mut out = vec![NodeId::from_index(99)];
    weak_control_closure_in(
        scratch,
        graph,
        seeds.into_iter().map(NodeId::from_index),
        &mut out,
    );
    out.iter().map(|vertex| vertex.index()).collect()
}

/// Check the closure of every seed set of the graph against the definition.
fn check_every_seed_set(
    count: usize,
    edges: &[(usize, usize)],
    scratch: &mut ControlClosureScratch,
) {
    let successors = successors_of(count, edges);
    let graph = graph_of(count, edges);
    for mask in 0..1_usize << count {
        let seeds: Vec<bool> = (0..count).map(|vertex| mask >> vertex & 1 == 1).collect();
        let chosen = (0..count).filter(|&vertex| seeds[vertex]);
        assert_eq!(
            closure(&graph, chosen, scratch),
            closure_by_paths(&successors, &seeds),
            "closure of {seeds:?} in {edges:?}"
        );
    }
}

#[test]
fn every_digraph_of_up_to_four_nodes_matches_the_definition() {
    let mut scratch = ControlClosureScratch::default();
    for count in 1..=4 {
        for_each_multigraph(count, 1, |edges| {
            check_every_seed_set(count, edges, &mut scratch);
        });
    }
}

#[test]
fn every_multigraph_of_up_to_three_nodes_matches_the_definition() {
    let mut scratch = ControlClosureScratch::default();
    for count in 1..=3 {
        for_each_multigraph(count, 2, |edges| {
            check_every_seed_set(count, edges, &mut scratch);
        });
    }
}

#[test]
fn random_graphs_match_danicics_algorithm() {
    let mut rng = Rng::new(84);
    let mut scratch = ControlClosureScratch::default();
    let mut grown = 0;
    for round in 0..60 {
        let count = 2 + rng.below(if round < 40 { 30 } else { 300 });
        let edges: Vec<(usize, usize)> = (0..count + rng.below(2 * count))
            .map(|_| (rng.below(count), rng.below(count)))
            .collect();
        let graph = graph_of(count, &edges);
        let mut seeds = vec![false; count];
        for _ in 0..=rng.below(1 + count / 8) {
            seeds[rng.below(count)] = true;
        }
        let expected = closure_by_critical_edges(&successors_of(count, &edges), &seeds);

        let chosen: Vec<usize> = (0..count).filter(|&vertex| seeds[vertex]).collect();
        grown += usize::from(expected.len() > chosen.len());
        assert_eq!(
            closure(&graph, chosen.iter().copied(), &mut scratch),
            expected,
            "closure of {chosen:?} in {edges:?}"
        );
        // The closure is a set: neither the order of the seeds nor a
        // repeated seed changes it.
        let shuffled = chosen.iter().rev().chain(&chosen).copied();
        assert_eq!(closure(&graph, shuffled, &mut scratch), expected);
        // And it is closed.
        assert_eq!(
            closure(&graph, expected.iter().copied(), &mut scratch),
            expected
        );
    }
    // A third of the closures add a decision, so the comparison is not one
    // of seed sets alone.
    assert!(grown >= 20, "only {grown} closures grew");
}

/// The CFG over `count` blocks with `edges`, entered at block 0.
fn cfg_of(count: usize, edges: &[(usize, usize)]) -> Cfg<u32> {
    let mut cfg = Cfg::new();
    for _ in 1..count {
        cfg.new_block();
    }
    for &(from, to) in edges {
        cfg.add_edge(
            BlockId::from_index(from),
            BlockId::from_index(to),
            EdgeKind::Unconditional,
        );
    }
    cfg
}

#[test]
fn with_the_entry_and_every_exit_as_seeds_it_is_the_iterated_frontier() {
    // Where every reachable block reaches an exit, post-dominance and weak
    // control-closure agree once the closure can see both ends: the entry,
    // so that controllers before a seed are reached, and the exits, so that
    // a choice between a seed and leaving is a choice between two seeds.
    let mut scratch = ControlClosureScratch::default();
    let mut frontier = DominatorScratch::default();
    let mut found = Vec::new();
    let mut expected = Vec::new();
    for count in 1..=4 {
        for_each_multigraph(count, 1, |edges| {
            let cfg = cfg_of(count, edges);
            let post = PostDominatorTree::compute(&cfg);
            let blocks: Vec<BlockId> = (0..count)
                .map(BlockId::from_index)
                .filter(|&block| post.contains(block))
                .collect();
            if !blocks.iter().all(|&block| post.reaches_exit(block)) {
                return;
            }
            for mask in 0..1_usize << blocks.len() {
                let mut seeds: Vec<BlockId> = (0..blocks.len())
                    .filter(|&at| mask >> at & 1 == 1)
                    .map(|at| blocks[at])
                    .chain([cfg.entry()])
                    .chain(post.exits().iter().copied())
                    .collect();
                seeds.sort_unstable();
                seeds.dedup();
                weak_control_closure_in(&mut scratch, &cfg, seeds.iter().copied(), &mut found);
                post.iterated_frontier_in(
                    &mut frontier,
                    &cfg,
                    seeds.iter().copied(),
                    None,
                    &mut expected,
                );
                expected.extend(&seeds);
                expected.sort_unstable();
                expected.dedup();
                assert_eq!(found, expected, "closure of {seeds:?} in {edges:?}");
            }
        });
    }
}

#[test]
fn a_loop_exit_test_does_not_decide_for_the_body() {
    // 0 -> 1 -> 0 | 2: the test at 1 controls the body 0 under
    // post-dominance, but leaving leads to no seed, so 1 chooses nothing.
    let cfg = cfg_of(3, &[(0, 1), (1, 0), (1, 2)]);
    let body = cfg.entry();
    let test = BlockId::from_index(1);
    let post = PostDominatorTree::compute(&cfg);
    let mut sources = Vec::new();
    post.frontier(&cfg, body, &mut sources);
    assert_eq!(sources, [test]);

    let mut scratch = ControlClosureScratch::default();
    let mut closure = Vec::new();
    weak_control_closure_in(&mut scratch, &cfg, [body], &mut closure);
    assert_eq!(closure, [body]);
    // With the exit as a seed the test chooses between two seeds.
    let exit = BlockId::from_index(2);
    weak_control_closure_in(&mut scratch, &cfg, [body, exit], &mut closure);
    assert_eq!(closure, [body, test, exit]);
}

#[test]
fn a_decision_the_seeds_do_not_reach_stays_out() {
    // 0 -> 1 | 2: block 0 chooses between the seeds 1 and 2, but neither
    // reaches it.
    let graph = graph_of(3, &[(0, 1), (0, 2)]);
    let mut scratch = ControlClosureScratch::default();
    assert_eq!(closure(&graph, [1, 2], &mut scratch), [1, 2]);
    assert_eq!(closure(&graph, [0, 1, 2], &mut scratch), [0, 1, 2]);
    // From a seed before it, block 3 decides between the seeds 1 and 2.
    let graph = graph_of(4, &[(0, 3), (3, 1), (3, 2)]);
    assert_eq!(closure(&graph, [0, 1, 2], &mut scratch), [0, 1, 2, 3]);
}

#[test]
fn removed_slots_and_a_reused_scratch_change_nothing() {
    // 0 -> 1 -> 3 | 4, 3 -> 4, with the slot of 2 (between 1 and 4) empty.
    let mut graph = graph_of(5, &[(0, 1), (1, 2), (1, 3), (2, 4), (3, 4), (1, 4)]);
    assert!(graph.remove_node(NodeId::from_index(2)));
    let mut scratch = ControlClosureScratch::default();
    for _ in 0..2 {
        assert_eq!(closure(&graph, [0, 3, 4], &mut scratch), [0, 1, 3, 4]);
        assert_eq!(closure(&graph, [0, 4], &mut scratch), [0, 4]);
        assert_eq!(closure(&graph, [], &mut scratch), Vec::<usize>::new());
    }
}
