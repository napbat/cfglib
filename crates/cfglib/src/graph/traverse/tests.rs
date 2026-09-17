use super::*;
use crate::edge::EdgeKind;
use crate::graph::store::{Graph, NodeId};
use crate::test_util::ff;
use alloc::vec;

#[test]
fn cfg_traversal_methods_delegate_to_generic_algorithms() {
    let mut cfg = Cfg::new();
    let middle = cfg.new_block();
    let last = cfg.new_block();
    cfg.block_mut(cfg.entry()).push(ff("entry"));
    cfg.block_mut(middle).push(ff("middle"));
    cfg.block_mut(last).push(ff("last"));
    cfg.add_edge(cfg.entry(), middle, EdgeKind::Fallthrough);
    cfg.add_edge(middle, last, EdgeKind::Fallthrough);

    assert_eq!(cfg.depth_first_preorder(), vec![cfg.entry(), middle, last]);
    assert_eq!(cfg.depth_first_postorder(), vec![last, middle, cfg.entry()]);
    assert_eq!(cfg.reverse_postorder(), vec![cfg.entry(), middle, last]);
    assert_eq!(cfg.breadth_first(), vec![cfg.entry(), middle, last]);
}

#[test]
fn directed_graph_can_be_walked_in_both_directions() {
    let mut graph = Graph::<&str, ()>::new();
    let first = graph.add_node("first");
    let second = graph.add_node("second");
    let third = graph.add_node("third");
    graph.add_edge(first, second, ());
    graph.add_edge(second, third, ());

    assert_eq!(
        breadth_first(&graph, first, TraversalDirection::Outgoing),
        vec![first, second, third]
    );
    assert_eq!(
        breadth_first(&graph, third, TraversalDirection::Incoming),
        vec![third, second, first]
    );
    assert_eq!(
        shortest_path(&graph, first, third, TraversalDirection::Outgoing),
        Some(vec![first, second, third])
    );
}

/// `a -> b -> c` with a `c -> b` back edge, plus a disconnected `d`.
fn reach_fixture() -> (Graph<(), ()>, [NodeId; 4]) {
    let mut graph = Graph::<(), ()>::new();
    let a = graph.add_node(());
    let b = graph.add_node(());
    let c = graph.add_node(());
    let d = graph.add_node(());
    graph.add_edge(a, b, ());
    graph.add_edge(b, c, ());
    graph.add_edge(c, b, ());
    (graph, [a, b, c, d])
}

#[test]
fn reachable_from_no_seeds_marks_nothing() {
    let (graph, _) = reach_fixture();
    assert_eq!(
        reachable(&graph, [], TraversalDirection::Outgoing),
        vec![false; 4]
    );

    // An empty graph yields an empty table rather than panicking.
    let empty = Graph::<(), ()>::new();
    assert!(reachable(&empty, [], TraversalDirection::Outgoing).is_empty());
}

#[test]
fn reachable_unions_multiple_sources_and_terminates_on_cycles() {
    let (graph, [a, _, c, d]) = reach_fixture();
    // The b <-> c cycle terminates; d stays unreached.
    assert_eq!(
        reachable(&graph, [a], TraversalDirection::Outgoing),
        vec![true, true, true, false]
    );
    // A second seed unions in, and duplicate seeds change nothing.
    assert_eq!(
        reachable(&graph, [a, d, a, d], TraversalDirection::Outgoing),
        vec![true; 4]
    );
    // Order-insensitive: the answer is a set.
    assert_eq!(
        reachable(&graph, [d, c], TraversalDirection::Outgoing),
        reachable(&graph, [c, d], TraversalDirection::Outgoing)
    );
    assert_eq!(
        reachable(&graph, [c], TraversalDirection::Outgoing),
        vec![false, true, true, false]
    );
}

#[test]
fn reachable_walks_predecessors_in_the_incoming_direction() {
    let (graph, [a, _, c, _]) = reach_fixture();
    assert_eq!(
        reachable(&graph, [c], TraversalDirection::Incoming),
        vec![true, true, true, false]
    );
    assert_eq!(
        reachable(&graph, [a], TraversalDirection::Incoming),
        vec![true, false, false, false]
    );
}

#[test]
fn reachable_handles_self_loops() {
    let mut graph = Graph::<(), ()>::new();
    let only = graph.add_node(());
    let other = graph.add_node(());
    graph.add_edge(only, only, ());
    assert_eq!(
        reachable(&graph, [only], TraversalDirection::Outgoing),
        vec![true, false]
    );
    // A self-loop is not a reason to be reachable from elsewhere.
    assert_eq!(
        reachable(&graph, [other], TraversalDirection::Outgoing),
        vec![false, true]
    );
}

/// `root -> mid`, `mid -> left`, `mid -> right`, both legs into `bottom`.
/// `root` has the smallest id but is the *farther* common ancestor.
fn diamond() -> (Graph<(), ()>, [NodeId; 5]) {
    let mut graph = Graph::<(), ()>::new();
    let root = graph.add_node(());
    let mid = graph.add_node(());
    let left = graph.add_node(());
    let right = graph.add_node(());
    let bottom = graph.add_node(());
    graph.add_edge(root, mid, ());
    graph.add_edge(mid, left, ());
    graph.add_edge(mid, right, ());
    graph.add_edge(left, bottom, ());
    graph.add_edge(right, bottom, ());
    (graph, [root, mid, left, right, bottom])
}

#[test]
fn nearest_common_ancestor_meets_at_the_closest_shared_node() {
    let (graph, [_root, mid, left, right, bottom]) = diamond();
    // `mid` (combined 2) beats `root` (combined 4) even though `root`
    // has the smaller id: distance ranks first.
    assert_eq!(
        nearest_common_ancestor(&graph, left, right, TraversalDirection::Incoming),
        Some(mid)
    );
    // Forward, the same two legs merge at the bottom.
    assert_eq!(
        nearest_common_ancestor(&graph, left, right, TraversalDirection::Outgoing),
        Some(bottom)
    );
    // The answer does not depend on which endpoint is passed first.
    assert_eq!(
        nearest_common_ancestor(&graph, right, left, TraversalDirection::Incoming),
        Some(mid)
    );
    assert_eq!(
        nearest_common_ancestor(&graph, bottom, mid, TraversalDirection::Incoming),
        Some(mid)
    );
}

#[test]
fn nearest_common_ancestor_treats_endpoints_as_distance_zero() {
    let (graph, [root, _, left, _, bottom]) = diamond();
    // A node is its own meet, in either direction.
    assert_eq!(
        nearest_common_ancestor(&graph, root, root, TraversalDirection::Outgoing),
        Some(root)
    );
    assert_eq!(
        nearest_common_ancestor(&graph, bottom, bottom, TraversalDirection::Incoming),
        Some(bottom)
    );
    // `left` is reachable from `root`, so the meet is `left` itself.
    assert_eq!(
        nearest_common_ancestor(&graph, root, left, TraversalDirection::Outgoing),
        Some(left)
    );
    assert_eq!(
        nearest_common_ancestor(&graph, left, root, TraversalDirection::Outgoing),
        Some(left)
    );
}

#[test]
fn nearest_common_ancestor_breaks_ties_by_smallest_node_id() {
    // Adjacency deliberately offers the higher-id sink first, so a
    // discovery-order answer would pick `second_sink`.
    let (graph, [first_sink, second_sink, left, right]) = twin_sinks();

    // Both sinks sit at combined distance 2; the smaller id wins.
    assert_eq!(
        nearest_common_ancestor(&graph, left, right, TraversalDirection::Outgoing),
        Some(first_sink)
    );
    assert_eq!(
        nearest_common_ancestor(&graph, right, left, TraversalDirection::Outgoing),
        Some(first_sink)
    );
    assert_eq!(
        nearest_common_ancestor(
            &graph,
            first_sink,
            second_sink,
            TraversalDirection::Incoming
        ),
        Some(left)
    );
}

#[test]
fn nearest_common_ancestor_without_a_shared_node_is_none() {
    let mut graph = Graph::<(), ()>::new();
    let start = graph.add_node(());
    let lonely = graph.add_node(());
    let end = graph.add_node(());
    graph.add_edge(start, end, ());

    assert_eq!(
        nearest_common_ancestor(&graph, start, lonely, TraversalDirection::Outgoing),
        None
    );
    assert_eq!(
        nearest_common_ancestor(&graph, start, lonely, TraversalDirection::Incoming),
        None
    );
    // Two nodes with no shared successor, both inside the connected part.
    assert_eq!(
        nearest_common_ancestor(&graph, end, lonely, TraversalDirection::Incoming),
        None
    );
}

#[test]
fn nearest_common_ancestor_terminates_on_cycles() {
    let (mut graph, [_, mid, left, right, bottom]) = diamond();
    graph.add_edge(bottom, mid, ());

    // Every node is now reachable from both legs; `bottom` (1 + 1) is
    // the closest forward meet and `mid` (1 + 1) the closest backward one.
    assert_eq!(
        nearest_common_ancestor(&graph, left, right, TraversalDirection::Outgoing),
        Some(bottom)
    );
    assert_eq!(
        nearest_common_ancestor(&graph, left, right, TraversalDirection::Incoming),
        Some(mid)
    );
}

/// Two shared sinks at the same distance, offered to the traversal in
/// descending id order — discovery order and id order disagree.
fn twin_sinks() -> (Graph<(), ()>, [NodeId; 4]) {
    let mut graph = Graph::<(), ()>::new();
    let first_sink = graph.add_node(());
    let second_sink = graph.add_node(());
    let left = graph.add_node(());
    let right = graph.add_node(());
    graph.add_edge(left, second_sink, ());
    graph.add_edge(left, first_sink, ());
    graph.add_edge(right, second_sink, ());
    graph.add_edge(right, first_sink, ());
    (graph, [first_sink, second_sink, left, right])
}

fn ancestor_nodes(found: &[CommonAncestor<NodeId>]) -> Vec<NodeId> {
    found.iter().map(|entry| entry.node).collect()
}

#[test]
fn common_ancestors_returns_every_shared_node_with_both_distances() {
    let (graph, [root, mid, left, right, bottom]) = diamond();
    let found = common_ancestors(&graph, left, right, TraversalDirection::Incoming, None);
    assert_eq!(ancestor_nodes(&found), vec![mid, root]);
    assert_eq!((found[0].from_a, found[0].from_b), (1, 1));
    assert_eq!((found[1].from_a, found[1].from_b), (2, 2));
    assert_eq!(found[1].combined(), 4);

    // Forward, the same two legs share only the merge point.
    let merged = common_ancestors(&graph, left, right, TraversalDirection::Outgoing, None);
    assert_eq!(ancestor_nodes(&merged), vec![bottom]);
}

#[test]
fn common_ancestors_are_in_b_discovery_order_not_id_order() {
    let (graph, [first_sink, second_sink, left, right]) = twin_sinks();
    // A walk from `right` reaches `second_sink` first because adjacency
    // offers it first, even though its id is larger.
    let found = common_ancestors(&graph, left, right, TraversalDirection::Outgoing, None);
    assert_eq!(ancestor_nodes(&found), vec![second_sink, first_sink]);
    // Both sit at the same combined distance, so a consumer's first-match
    // scan takes the discovery-order winner while the fixed rank of
    // `nearest_common_ancestor` takes the smaller id.
    let scanned = found.iter().min_by_key(|entry| entry.combined());
    assert_eq!(scanned.map(|entry| entry.node), Some(second_sink));
    assert_eq!(
        nearest_common_ancestor(&graph, left, right, TraversalDirection::Outgoing),
        Some(first_sink)
    );

    // Swapping the endpoints swaps the order, since it is `b`'s.
    let swapped = common_ancestors(&graph, right, left, TraversalDirection::Outgoing, None);
    assert_eq!(ancestor_nodes(&swapped), vec![second_sink, first_sink]);
}

#[test]
fn common_ancestors_bound_each_side_by_max_depth() {
    let (graph, [_root, mid, left, right, _bottom]) = diamond();
    // `root` is two hops from either leg, `mid` one.
    assert_eq!(
        ancestor_nodes(&common_ancestors(
            &graph,
            left,
            right,
            TraversalDirection::Incoming,
            Some(1)
        )),
        vec![mid]
    );
    assert!(
        common_ancestors(&graph, left, right, TraversalDirection::Incoming, Some(0)).is_empty()
    );
    // A bound at or beyond the eccentricity is the unbounded answer.
    assert_eq!(
        common_ancestors(&graph, left, right, TraversalDirection::Incoming, Some(2)),
        common_ancestors(&graph, left, right, TraversalDirection::Incoming, None)
    );
}

#[test]
fn common_ancestors_of_a_node_with_itself_is_its_reachable_set() {
    let (graph, [_root, _mid, left, _right, bottom]) = diamond();
    let found = common_ancestors(&graph, left, left, TraversalDirection::Outgoing, None);
    assert_eq!(ancestor_nodes(&found), vec![left, bottom]);
    assert!(
        found
            .iter()
            .all(|entry| entry.from_a == entry.from_b && entry.combined() % 2 == 0)
    );
    // Bounded to zero hops, only the node itself.
    assert_eq!(
        ancestor_nodes(&common_ancestors(
            &graph,
            left,
            left,
            TraversalDirection::Outgoing,
            Some(0)
        )),
        vec![left]
    );
}

#[test]
fn common_ancestors_without_a_shared_node_is_empty() {
    let mut graph = Graph::<(), ()>::new();
    let start = graph.add_node(());
    let lonely = graph.add_node(());
    let end = graph.add_node(());
    graph.add_edge(start, end, ());
    assert!(common_ancestors(&graph, start, lonely, TraversalDirection::Outgoing, None).is_empty());
    assert!(common_ancestors(&graph, end, lonely, TraversalDirection::Incoming, None).is_empty());
}

#[test]
fn common_ancestors_terminates_on_cycles() {
    let (mut graph, [root, mid, left, right, bottom]) = diamond();
    graph.add_edge(bottom, mid, ());

    // Every node now reaches both legs; the walk still terminates and
    // reports each node exactly once, in `right`'s discovery order.
    let found = common_ancestors(&graph, left, right, TraversalDirection::Incoming, None);
    assert_eq!(ancestor_nodes(&found), vec![right, mid, root, bottom, left]);
    assert_eq!(
        found
            .iter()
            .min_by_key(|entry| entry.combined())
            .map(|entry| entry.node),
        Some(mid)
    );
}

#[test]
fn common_ancestors_generalizes_nearest_common_ancestor() {
    let (plain, _) = diamond();
    let (twins, _) = twin_sinks();
    let (mut cyclic, [_, mid, _, _, bottom]) = diamond();
    cyclic.add_edge(bottom, mid, ());
    for graph in [&plain, &twins, &cyclic] {
        for a in graph.node_ids() {
            for b in graph.node_ids() {
                for direction in [TraversalDirection::Incoming, TraversalDirection::Outgoing] {
                    // The fixed rank is a `min` over `(combined, id)` of
                    // exactly this candidate set.
                    let ranked = common_ancestors(graph, a, b, direction, None)
                        .into_iter()
                        .min_by_key(|entry| (entry.combined(), entry.node))
                        .map(|entry| entry.node);
                    assert_eq!(ranked, nearest_common_ancestor(graph, a, b, direction));
                }
            }
        }
    }
}

#[test]
fn topological_sort_rejects_cycles() {
    let mut graph = Graph::<(), ()>::new();
    let left = graph.add_node(());
    let right = graph.add_node(());
    graph.add_edge(left, right, ());
    assert_eq!(topological_sort(&graph), Some(vec![left, right]));
    graph.add_edge(right, left, ());
    assert!(topological_sort(&graph).is_none());
}
