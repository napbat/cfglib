//! `common_ancestors`, `nearest_common_ancestor`, `Reversed`, `reachable` and `reverse_postorder`.

use std::collections::BTreeSet;

use cfglib::{
    Graph, GraphView, NodeId, NodeView, Reversed, TraversalDirection, breadth_first,
    common_ancestors, depth_first_postorder, depth_first_preorder, nearest_common_ancestor,
    reachable, reverse_postorder, scan_predecessors,
};

use crate::Lcg;
use crate::digraph::Digraph;

const DIRECTIONS: [(TraversalDirection, bool); 2] = [
    (TraversalDirection::Outgoing, true),
    (TraversalDirection::Incoming, false),
];

fn indexes(nodes: Vec<NodeId>) -> Vec<usize> {
    nodes.into_iter().map(NodeId::index).collect()
}

#[test]
fn common_ancestors_match_explicit_ancestor_sets_in_the_discovery_order_of_b() {
    for seed in 1..=8 {
        let digraph = Digraph::random(&mut Lcg(seed), 16, 26);
        let graph = digraph.graph();
        for (direction, forward) in DIRECTIONS {
            for limit in [None, Some(0), Some(1), Some(2)] {
                for (a, b) in [(0, 5), (3, 3), (7, 12), (15, 1), (9, 2)] {
                    let from_a = digraph.distances(a, forward, limit);
                    let from_b = digraph.distances(b, forward, limit);
                    let expected: Vec<(usize, usize, usize)> = digraph
                        .discovery_order(b, forward, limit)
                        .into_iter()
                        .filter_map(|node| Some((node, from_a[node]?, from_b[node]?)))
                        .collect();

                    let actual: Vec<(usize, usize, usize)> = common_ancestors(
                        &graph,
                        NodeId::from_index(a),
                        NodeId::from_index(b),
                        direction,
                        limit,
                    )
                    .into_iter()
                    .map(|found| (found.node.index(), found.from_a, found.from_b))
                    .collect();
                    assert_eq!(
                        actual, expected,
                        "seed {seed}, {direction:?}, limit {limit:?}, n{a} and n{b}"
                    );
                }
            }
        }
    }
}

#[test]
fn the_nearest_common_ancestor_minimizes_combined_distance_then_node_id() {
    let mut ties = 0;
    for seed in 1..=8 {
        let digraph = Digraph::random(&mut Lcg(seed + 30), 16, 26);
        let graph = digraph.graph();
        for (direction, forward) in DIRECTIONS {
            for a in 0..16 {
                for b in 0..16 {
                    let from_a = digraph.distances(a, forward, None);
                    let from_b = digraph.distances(b, forward, None);
                    let mut ranked: Vec<(usize, usize)> = (0..16)
                        .filter_map(|node| Some((from_a[node]? + from_b[node]?, node)))
                        .collect();
                    ranked.sort_unstable();
                    if ranked.len() > 1 && ranked[0].0 == ranked[1].0 {
                        ties += 1;
                    }
                    let expected = ranked.first().map(|&(_, node)| node);

                    let (first, second) = (NodeId::from_index(a), NodeId::from_index(b));
                    let found = nearest_common_ancestor(&graph, first, second, direction)
                        .map(NodeId::index);
                    assert_eq!(found, expected, "seed {seed}, {direction:?}, n{a} and n{b}");
                    let swapped = nearest_common_ancestor(&graph, second, first, direction)
                        .map(NodeId::index);
                    assert_eq!(
                        swapped, expected,
                        "the answer never depends on the argument order"
                    );
                }
            }
        }
    }
    assert!(ties > 50, "{ties} pairs had a tie to break");
}

/// Edges run base to derived: `object -> {mixin, plain}`, `mixin -> {a, b}`, `plain -> b`.
fn class_hierarchy() -> (Graph<&'static str, ()>, [NodeId; 5]) {
    let mut graph = Graph::new();
    let object = graph.add_node("object");
    let mixin = graph.add_node("mixin");
    let plain = graph.add_node("plain");
    let a = graph.add_node("A");
    let b = graph.add_node("B");
    graph.add_edge(object, mixin, ());
    graph.add_edge(object, plain, ());
    graph.add_edge(mixin, a, ());
    graph.add_edge(mixin, b, ());
    graph.add_edge(plain, b, ());
    (graph, [object, mixin, plain, a, b])
}

#[test]
fn a_hand_derived_inheritance_query_lists_shared_bases_in_the_order_b_discovers_them() {
    let (graph, [object, mixin, _plain, a, b]) = class_hierarchy();
    let incoming = TraversalDirection::Incoming;

    let shared = common_ancestors(&graph, a, b, incoming, None);
    let summary: Vec<(&str, usize, usize)> = shared
        .iter()
        .map(|found| (*graph.node(found.node), found.from_a, found.from_b))
        .collect();
    // B reaches mixin first (its first predecessor), then plain, then object; A reaches only
    // mixin (one hop) and object (two). `plain` is not an ancestor of A.
    assert_eq!(summary, [("mixin", 1, 1), ("object", 2, 2)]);
    assert_eq!(nearest_common_ancestor(&graph, a, b, incoming), Some(mixin));

    let near = common_ancestors(&graph, a, b, incoming, Some(1));
    assert_eq!(
        near.iter().map(|found| found.node).collect::<Vec<_>>(),
        [mixin]
    );
    assert_eq!(near[0].combined(), 2);
    assert_eq!(
        common_ancestors(&graph, a, b, incoming, Some(0)).len(),
        0,
        "only a == b can meet at zero hops"
    );
    assert_eq!(
        nearest_common_ancestor(&graph, object, a, incoming),
        Some(object),
        "an ancestor of the other"
    );
}

#[test]
fn a_depth_bound_applies_to_each_endpoint_independently() {
    // origin -> first -> second -> deep is three hops long; origin -> shallow is one.
    let mut graph = Graph::<&str, ()>::new();
    let origin = graph.add_node("origin");
    let first = graph.add_node("first");
    let second = graph.add_node("second");
    let deep = graph.add_node("deep");
    let shallow = graph.add_node("shallow");
    graph.add_edge(origin, first, ());
    graph.add_edge(first, second, ());
    graph.add_edge(second, deep, ());
    graph.add_edge(origin, shallow, ());
    let incoming = TraversalDirection::Incoming;

    assert_eq!(
        common_ancestors(&graph, deep, shallow, incoming, Some(2)).len(),
        0,
        "the origin is three hops from `deep`"
    );
    let found = common_ancestors(&graph, deep, shallow, incoming, Some(3));
    assert_eq!(found.len(), 1);
    assert_eq!(
        (found[0].node, found[0].from_a, found[0].from_b),
        (origin, 3, 1)
    );
}

#[test]
fn the_nearest_meeting_point_follows_the_combined_distance_rule_not_the_second_endpoint() {
    // left -> meeting, left -> detour -> turn -> right, right -> meeting: `right` is reachable from
    // `left` in three hops, but meeting at `meeting` costs two.
    let mut graph = Graph::<&str, ()>::new();
    let left = graph.add_node("left");
    let meeting = graph.add_node("meeting");
    let detour = graph.add_node("detour");
    let turn = graph.add_node("turn");
    let right = graph.add_node("right");
    graph.add_edge(left, meeting, ());
    graph.add_edge(left, detour, ());
    graph.add_edge(detour, turn, ());
    graph.add_edge(turn, right, ());
    graph.add_edge(right, meeting, ());

    let outgoing = TraversalDirection::Outgoing;
    assert_eq!(
        nearest_common_ancestor(&graph, left, right, outgoing),
        Some(meeting),
        "1 + 1 beats 3 + 0"
    );
    let all: Vec<_> = common_ancestors(&graph, left, right, outgoing, None)
        .into_iter()
        .map(|found| (found.node, found.from_a, found.from_b))
        .collect();
    assert_eq!(
        all,
        [(right, 3, 0), (meeting, 1, 1)],
        "`right` discovers itself first, then `meeting`"
    );
}

#[test]
fn equal_endpoints_and_disconnected_endpoints_have_hand_derived_answers() {
    let (graph, [object, mixin, plain, a, b]) = class_hierarchy();
    let incoming = TraversalDirection::Incoming;
    assert_eq!(
        nearest_common_ancestor(&graph, a, a, incoming),
        Some(a),
        "a == b answers a"
    );
    let own: Vec<_> = common_ancestors(&graph, a, a, incoming, None)
        .into_iter()
        .map(|found| (found.node, found.from_a == found.from_b))
        .collect();
    assert_eq!(own, [(a, true), (mixin, true), (object, true)]);

    // `plain` and `mixin` are siblings under `object`, and have no common descendant but B.
    assert_eq!(
        nearest_common_ancestor(&graph, plain, mixin, incoming),
        Some(object)
    );
    assert_eq!(
        nearest_common_ancestor(&graph, plain, mixin, TraversalDirection::Outgoing),
        Some(b)
    );
    assert_eq!(
        nearest_common_ancestor(&graph, a, b, TraversalDirection::Outgoing),
        None
    );
    assert_eq!(
        common_ancestors(&graph, a, b, TraversalDirection::Outgoing, None).len(),
        0
    );
}

#[test]
#[should_panic(expected = "node `a` is out of range")]
fn a_first_endpoint_outside_the_graph_panics() {
    let (graph, _) = class_hierarchy();
    let _ = nearest_common_ancestor(
        &graph,
        NodeId::from_index(9),
        NodeId::from_index(0),
        TraversalDirection::Incoming,
    );
}

#[test]
#[should_panic(expected = "node `b` is out of range")]
fn a_second_endpoint_outside_the_graph_panics_in_the_ranked_query() {
    let (graph, _) = class_hierarchy();
    let _ = common_ancestors(
        &graph,
        NodeId::from_index(0),
        NodeId::from_index(9),
        TraversalDirection::Incoming,
        None,
    );
}

/// A view that stores only forward adjacency.
struct ForwardOnly {
    successors: Vec<Vec<usize>>,
}

impl ForwardOnly {
    fn from_digraph(digraph: &Digraph) -> Self {
        Self {
            successors: (0..digraph.nodes)
                .map(|node| digraph.neighbors(node, true).to_vec())
                .collect(),
        }
    }
}

impl GraphView for ForwardOnly {
    type NodeId = usize;

    fn node_bound(&self) -> usize {
        self.successors.len()
    }

    fn node_ids(&self) -> impl Iterator<Item = usize> + '_ {
        0..self.successors.len()
    }

    fn successors(&self, node: usize) -> impl Iterator<Item = usize> + '_ {
        self.successors[node].iter().copied()
    }

    fn predecessors(&self, node: usize) -> impl Iterator<Item = usize> + '_ {
        scan_predecessors(self, node)
    }
}

/// Distinct edges `i -> j` with `i < j`, so the graph is acyclic and has no parallel edges.
fn random_dag(rng: &mut Lcg, nodes: usize, wanted: usize) -> Digraph {
    let mut edges = BTreeSet::new();
    while edges.len() < wanted {
        let (first, second) = (rng.below(nodes), rng.below(nodes));
        if first != second {
            edges.insert((first.min(second), first.max(second)));
        }
    }
    Digraph::new(nodes, edges.into_iter().collect())
}

#[test]
fn reversing_swaps_the_two_adjacency_directions_and_nothing_else() {
    let digraph = Digraph::random(&mut Lcg(4), 14, 30);
    let graph = digraph.graph();
    let reversed = Reversed::new(&graph);

    assert_eq!(reversed.node_bound(), graph.node_bound());
    assert_eq!(
        reversed.node_ids().collect::<Vec<_>>(),
        graph.node_ids().collect::<Vec<_>>()
    );
    for node in graph.node_ids() {
        assert_eq!(
            reversed.successors(node).collect::<Vec<_>>(),
            graph.predecessors(node).collect::<Vec<_>>()
        );
        assert_eq!(
            reversed.predecessors(node).collect::<Vec<_>>(),
            graph.successors(node).collect::<Vec<_>>()
        );
    }
    let twice = Reversed::new(&reversed);
    for node in graph.node_ids() {
        assert_eq!(
            twice.successors(node).collect::<Vec<_>>(),
            graph.successors(node).collect::<Vec<_>>()
        );
    }
    assert_eq!(reversed.graph().edge_count(), graph.edge_count());
}

#[test]
fn a_reversed_view_forwards_node_payloads_and_leaves_removed_nodes_out() {
    let mut graph = Graph::<&str, ()>::new();
    let a = graph.add_node("a");
    let b = graph.add_node("b");
    let c = graph.add_node("c");
    graph.add_edge(a, b, ());
    graph.add_edge(b, c, ());
    graph.remove_node(b);

    let reversed = Reversed::new(&graph);
    assert_eq!(*reversed.node(a), "a");
    assert_eq!(
        *reversed.node(b),
        "b",
        "a removed payload is still readable through the view"
    );
    assert_eq!(reversed.node_ids().collect::<Vec<_>>(), [a, c]);
    assert_eq!(reversed.node_bound(), 3);
    assert_eq!(reversed.successors(c).count(), 0);
}

#[test]
fn walks_over_a_reversed_view_equal_the_incoming_walks_over_the_view_itself() {
    for seed in 1..=5 {
        let digraph = Digraph::random(&mut Lcg(seed + 50), 20, 40);
        let graph = digraph.graph();
        let reversed = Reversed::new(&graph);
        let outgoing = TraversalDirection::Outgoing;
        let incoming = TraversalDirection::Incoming;
        for start in [0, 9, 19] {
            let id = NodeId::from_index(start);
            assert_eq!(
                breadth_first(&reversed, id, outgoing),
                breadth_first(&graph, id, incoming)
            );
            assert_eq!(
                depth_first_preorder(&reversed, id, outgoing),
                depth_first_preorder(&graph, id, incoming)
            );
            assert_eq!(
                reachable(&reversed, [id], outgoing),
                reachable(&graph, [id], incoming)
            );
        }

        // A view with no reverse index still reverses: its predecessors are scanned.
        let simple_edges: BTreeSet<(usize, usize)> = digraph.edges.iter().copied().collect();
        let simple = Digraph::new(20, simple_edges.into_iter().collect());
        let view = ForwardOnly::from_digraph(&simple);
        let reversed_view = Reversed::new(&view);
        let stored = simple.graph();
        for start in [0_usize, 9, 19] {
            assert_eq!(
                breadth_first(&reversed_view, start, outgoing),
                indexes(breadth_first(&stored, NodeId::from_index(start), incoming)),
                "seed {seed}, n{start}"
            );
        }
    }
}

/// Recursive postorder from the definition: successors in order, a node after all of them.
fn reference_postorder(
    digraph: &Digraph,
    node: usize,
    forward: bool,
    seen: &mut [bool],
    out: &mut Vec<usize>,
) {
    seen[node] = true;
    for &next in digraph.neighbors(node, forward) {
        if !seen[next] {
            reference_postorder(digraph, next, forward, seen, out);
        }
    }
    out.push(node);
}

#[test]
fn reverse_postorder_is_the_reversed_recursive_postorder_and_postorder_is_its_mirror() {
    for seed in 1..=6 {
        let digraph = Digraph::random(&mut Lcg(seed + 70), 20, 36);
        let view = ForwardOnly::from_digraph(&digraph);
        let graph = digraph.graph();
        for start in [0, 8, 16] {
            let mut seen = vec![false; 20];
            let mut postorder = Vec::new();
            reference_postorder(&digraph, start, true, &mut seen, &mut postorder);
            let mut expected = postorder.clone();
            expected.reverse();

            assert_eq!(
                depth_first_postorder(&view, start, TraversalDirection::Outgoing),
                postorder
            );
            assert_eq!(
                reverse_postorder(&view, start, TraversalDirection::Outgoing),
                expected,
                "seed {seed}"
            );
            assert_eq!(
                indexes(reverse_postorder(
                    &graph,
                    NodeId::from_index(start),
                    TraversalDirection::Outgoing
                )),
                expected
            );
            assert_eq!(
                expected.len(),
                seen.iter().filter(|&&yes| yes).count(),
                "only reached nodes appear"
            );
        }
    }
}

#[test]
fn over_an_acyclic_view_reverse_postorder_puts_every_source_before_its_targets() {
    for seed in 1..=8 {
        let digraph = random_dag(&mut Lcg(seed + 90), 18, 30);
        let view = ForwardOnly::from_digraph(&digraph);
        for (direction, forward) in DIRECTIONS {
            let start = if forward { 0 } else { 17 };
            let order = reverse_postorder(&view, start, direction);
            let mut position = vec![None; 18];
            for (index, &node) in order.iter().enumerate() {
                position[node] = Some(index);
            }
            for &(source, target) in &digraph.edges {
                let (first, second) = if forward {
                    (source, target)
                } else {
                    (target, source)
                };
                if let (Some(before), Some(after)) = (position[first], position[second]) {
                    assert!(
                        before < after,
                        "seed {seed}, {direction:?}: n{first} must precede n{second}"
                    );
                }
            }
            let mut seen = vec![false; 18];
            let mut postorder = Vec::new();
            reference_postorder(&digraph, start, forward, &mut seen, &mut postorder);
            assert_eq!(order.len(), postorder.len());
        }
    }
}

#[test]
fn reverse_postorder_of_a_unit_graph_and_a_cycle_are_hand_derived() {
    let single = ForwardOnly {
        successors: vec![vec![]],
    };
    assert_eq!(
        reverse_postorder(&single, 0, TraversalDirection::Outgoing),
        [0]
    );

    // 0 -> 1 -> 2 -> 0 and 0 -> 3
    let cycle = ForwardOnly {
        successors: vec![vec![1, 3], vec![2], vec![0], vec![]],
    };
    assert_eq!(
        depth_first_postorder(&cycle, 0, TraversalDirection::Outgoing),
        [2, 1, 3, 0]
    );
    assert_eq!(
        reverse_postorder(&cycle, 0, TraversalDirection::Outgoing),
        [0, 3, 1, 2]
    );
}

#[test]
fn multi_source_reachability_is_the_union_of_the_single_source_sets() {
    for seed in 1..=6 {
        let digraph = Digraph::random(&mut Lcg(seed + 110), 24, 34);
        let graph = digraph.graph();
        let view = ForwardOnly::from_digraph(&digraph);
        for (direction, forward) in DIRECTIONS {
            for seeds in [vec![], vec![3], vec![3, 3, 17], vec![0, 5, 11, 23]] {
                let mut expected = vec![false; 24];
                for &start in &seeds {
                    for (node, distance) in digraph
                        .distances(start, forward, None)
                        .into_iter()
                        .enumerate()
                    {
                        expected[node] |= distance.is_some();
                    }
                }
                let ids: Vec<NodeId> = seeds.iter().map(|&node| NodeId::from_index(node)).collect();
                assert_eq!(
                    reachable(&graph, ids, direction),
                    expected,
                    "seed {seed}, {direction:?}, {seeds:?}"
                );
                if forward {
                    assert_eq!(reachable(&view, seeds.clone(), direction), expected);
                }
            }
        }
    }
}

#[test]
fn reachability_skips_removed_nodes_and_every_seed_is_reached_itself() {
    let digraph = Digraph::random(&mut Lcg(8), 12, 20);
    let mut graph = digraph.graph();
    graph.remove_node(NodeId::from_index(4));
    let live = reachable(
        &graph,
        [NodeId::from_index(0)],
        TraversalDirection::Outgoing,
    );
    assert!(live[0], "a seed is reached");
    assert!(!live[4], "a removed node is reached by nothing");
    assert_eq!(live.len(), graph.node_bound());
}

#[test]
#[should_panic(expected = "seed node is out of range")]
fn a_reachability_seed_outside_the_graph_panics() {
    let graph = Digraph::random(&mut Lcg(1), 5, 6).graph();
    let _ = reachable(
        &graph,
        [NodeId::from_index(5)],
        TraversalDirection::Outgoing,
    );
}
