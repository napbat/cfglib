//! Sink classes against the sink post-dominance of Hecker et al., computed
//! as the greatest fixpoint of its rules: the gate on contracting each
//! exit-free sink into one class, which the paper does not state.

extern crate alloc;
use alloc::collections::BTreeSet;
use alloc::vec;
use alloc::vec::Vec;

use super::differential::small_graph;
use crate::graph::cdg::control_dependence_graph;
use crate::graph::dominator::tests::support::{
    Digraph, EdgeList, Family, Recipe, Rng, for_each_multigraph, node, random_graph,
};
use crate::graph::dominator::{
    DominanceFrontiers, DominatorScratch, ExitPolicy, PostDominatorTree, PostNode,
};
use crate::graph::view::GraphView;
use crate::test_util::postdominance::{
    Fixpoint, post_dominance, reachable_part, two_successor_dependences,
};

/// The successor lists of `edges`.
fn successors_of(edges: &EdgeList) -> Vec<Vec<usize>> {
    let mut successors = vec![Vec::new(); edges.node_count];
    for &(from, to) in &edges.edges {
        successors[from as usize].push(to as usize);
    }
    successors
}

/// The sets of blocks to close under the frontier: every set of `count`
/// blocks while there are few, and each block alone and each pair after.
fn subsets(count: usize) -> Vec<Vec<u32>> {
    if count > 6 {
        return (0..node(count))
            .flat_map(|first| (first..node(count)).map(move |second| vec![first, second]))
            .collect();
    }
    (0..1_u32 << count)
        .map(|mask| {
            (0..node(count))
                .filter(|&bit| (mask >> bit) & 1 == 1)
                .collect()
        })
        .collect()
}

/// Check the tree with sink classes of `edges` against the fixpoint.
fn check(edges: &EdgeList, scratch: &mut DominatorScratch) {
    let graph: Digraph = edges.graph();
    let post = PostDominatorTree::compute_in(scratch, &graph, ExitPolicy::SinkComponents);
    let (part, kept) = reachable_part(&successors_of(edges), 0);
    let relation = post_dominance(&part, Fixpoint::Sink);
    let id = |index: usize| node(kept[index]);

    check_relation(&post, &relation, &kept, edges);
    // Nontermination-insensitive control dependence: the two-successor
    // definition over the fixpoint.
    let expected: BTreeSet<(u32, u32)> = two_successor_dependences(&part, &relation)
        .into_iter()
        .map(|(controller, dependent)| (id(controller), id(dependent)))
        .collect();
    check_dependences(&post, &graph, &expected, scratch, edges);

    // Where every reachable block reaches an exit there is nothing to read
    // two ways, and the two policies agree.
    let cut = PostDominatorTree::compute(&graph);
    if (0..node(edges.node_count)).all(|block| !post.contains(block) || post.reaches_exit(block)) {
        assert_eq!(post, cut, "{edges:?}");
    }
}

/// Check post-dominance, the classes and the common post-dominators against
/// `relation`, over the reachable blocks `kept`.
fn check_relation(
    post: &PostDominatorTree<u32>,
    relation: &[Vec<bool>],
    kept: &[usize],
    edges: &EdgeList,
) {
    let id = |index: usize| node(kept[index]);
    let count = edges.node_count;
    for block in 0..node(count) {
        assert_eq!(
            post.contains(block),
            kept.contains(&(block as usize)),
            "{block} in {edges:?}"
        );
    }
    for (m, row) in relation.iter().enumerate() {
        for (n, &holds) in row.iter().enumerate() {
            assert_eq!(
                post.post_dominates(id(m), id(n)),
                holds,
                "{} post-dominates {} in {edges:?}",
                id(m),
                id(n)
            );
            // Two blocks of a class post-dominate each other, and so does
            // nothing else but a block and itself.
            let class = post.sink_class(id(m));
            assert_eq!(
                holds && relation[n][m],
                m == n || (class.is_some() && class == post.sink_class(id(n))),
                "{} and {} in {edges:?}",
                id(m),
                id(n)
            );
        }
        if let Some(class) = post.sink_class(id(m)) {
            assert!(post.sink_members(class).contains(&id(m)));
            assert_eq!(post.ipdom(id(m)), Some(PostNode::Exit));
        }
    }
    for first in 0..node(count) {
        for second in 0..node(count) {
            if let Some(PostNode::Node(meet)) = post.nearest_common_post_dominator(first, second) {
                assert!(post.post_dominates(meet, first) && post.post_dominates(meet, second));
                assert!(
                    post.sink_class(meet)
                        .is_none_or(|class| post.sink_members(class)[0] == meet)
                );
            }
        }
    }
}

/// Check the control dependence graph, the frontier and the iterated
/// frontier, per query and off a [`DominanceFrontiers`] table, against the
/// dependences `expected`.
fn check_dependences(
    post: &PostDominatorTree<u32>,
    graph: &Digraph,
    expected: &BTreeSet<(u32, u32)>,
    scratch: &mut DominatorScratch,
    edges: &EdgeList,
) {
    let dependences = control_dependence_graph(graph, post);
    let found: BTreeSet<(u32, u32)> = dependences
        .edges()
        .map(|edge| {
            (
                *dependences.node(edge.source()),
                *dependences.node(edge.target()),
            )
        })
        .collect();
    assert_eq!(&found, expected, "dependences of {edges:?}");

    let sources_of = |block: u32| {
        expected
            .iter()
            .filter(move |&&(_, dependent)| dependent == block)
            .map(|&(controller, _)| controller)
    };
    let table = DominanceFrontiers::compute(graph, post);
    let mut sources = Vec::new();
    for block in 0..node(edges.node_count) {
        post.frontier(graph, block, &mut sources);
        let expected: Vec<u32> = sources_of(block)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        assert_eq!(sources, expected, "sources of {block} in {edges:?}");
        assert_eq!(
            table.frontier(block),
            expected,
            "tabulated sources of {block} in {edges:?}"
        );
    }
    let mut from_table = Vec::new();
    for blocks in subsets(edges.node_count) {
        post.iterated_frontier_in(scratch, graph, blocks.iter().copied(), None, &mut sources);
        table.iterated_frontier_in(scratch, blocks.iter().copied(), None, &mut from_table);
        let mut closure: BTreeSet<u32> = BTreeSet::new();
        let mut frontier: Vec<u32> = blocks.clone();
        while let Some(block) = frontier.pop() {
            for controller in sources_of(block) {
                if closure.insert(controller) {
                    frontier.push(controller);
                }
            }
        }
        let closure: Vec<u32> = closure.into_iter().collect();
        assert_eq!(
            sources, closure,
            "iterated frontier of {blocks:?} in {edges:?}"
        );
        assert_eq!(
            from_table, closure,
            "tabulated iterated frontier of {blocks:?} in {edges:?}"
        );
    }
}

#[test]
fn every_digraph_of_up_to_four_nodes_matches_sink_post_dominance() {
    let mut scratch = DominatorScratch::default();
    for count in 1..=4 {
        for_each_multigraph(count, 1, |edges| check(edges, &mut scratch));
    }
}

#[test]
fn every_multigraph_of_up_to_three_nodes_matches_sink_post_dominance() {
    let mut scratch = DominatorScratch::default();
    for count in 1..=3 {
        for_each_multigraph(count, 2, |edges| check(edges, &mut scratch));
    }
}

#[test]
fn random_graphs_match_sink_post_dominance() {
    let mut scratch = DominatorScratch::default();
    for seed in 0..120 {
        let mut rng = Rng::new(seed);
        let (_, edges) = small_graph(seed, &mut rng);
        check(&edges, &mut scratch);
    }
}

#[test]
fn sinks_entered_at_several_blocks_match_sink_post_dominance() {
    // Graphs with no exit at all: every block that would end a path jumps
    // back somewhere, so the exit-free blocks form loops entered from many
    // places, and every one of them is in a sink or reaches one.
    let mut scratch = DominatorScratch::default();
    for seed in 0..120 {
        let mut rng = Rng::new(seed + 1_000);
        let recipe = Recipe {
            family: Family::ALL[rng.below(3)],
            node_count: 5 + rng.below(9),
            parallel: rng.chance(30),
            detached: 0,
        };
        let mut edges = random_graph(&mut rng, recipe);
        let count = edges.node_count;
        for block in 0..count {
            if !edges.edges.iter().any(|&(from, _)| from as usize == block) {
                edges.edges.push((node(block), node(rng.below(count))));
            }
        }
        check(&edges, &mut scratch);
    }
}

#[test]
fn a_server_loop_is_one_class_named_by_its_head() {
    // entry -> head, head -> ok | err, ok -> latch, err -> latch,
    // latch -> head.
    let edges = EdgeList::new(5, vec![(0, 1), (1, 2), (1, 3), (2, 4), (3, 4), (4, 1)]);
    let graph = edges.graph();
    let post = PostDominatorTree::compute_in(
        &mut DominatorScratch::default(),
        &graph,
        ExitPolicy::SinkComponents,
    );
    let class = post.sink_class(1).expect("the loop is a sink");
    assert_eq!(class.index(), 0);
    assert_eq!(post.sink_members(class), [1, 2, 3, 4]);
    assert_eq!(post.sink_class(0), None);
    assert_eq!(post.cut_roots(), Vec::<u32>::new());
    assert_eq!(post.ipdom(0), Some(PostNode::Node(1)));
    assert_eq!(post.ipdom(3), Some(PostNode::Exit));
    assert_eq!(post.preorder(), [1, 0]);
    assert_eq!(post.subtree(3), [1, 0]);
    assert_eq!(post.children(PostNode::Node(4)).collect::<Vec<_>>(), [0]);
    assert_eq!(
        post.nearest_common_post_dominator(2, 4),
        Some(PostNode::Node(1))
    );
    assert_eq!(
        post.nearest_common_post_dominator_among([0, 3]),
        Some(PostNode::Node(1))
    );
    assert_eq!(
        PostDominatorTree::compute_in(
            &mut DominatorScratch::default(),
            &graph,
            ExitPolicy::CutLoops
        ),
        PostDominatorTree::compute(&graph)
    );
    assert_eq!(graph.node_bound(), 5);
}
