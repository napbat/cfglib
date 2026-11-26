//! Ordered compaction against its contract: the compacted store is the image
//! of the original under the returned renumbering, only nodes move, and every
//! order numbers the nodes the way its definition says, checked against an
//! oracle built from the crate's plain traversals.

extern crate alloc;
extern crate std;

use alloc::vec;
use alloc::vec::Vec;
use core::mem;

use crate::graph::dominator::DominatorTree;
use crate::graph::store::{EdgeId, Graph, NodeId, NodeOrder, Renumbering};
use crate::graph::structure::is_reducible;
use crate::graph::traverse::{breadth_first, reverse_postorder};
use crate::graph::view::{Direction, Rooted, VirtualRoot};
use crate::test_util::rng::Rng;

type Store = Graph<u64, u64>;

/// Payloads are unique across nodes and edges, so a payload names the entity
/// it was minted for wherever a compaction puts it.
struct Minter(u64);

impl Minter {
    fn next(&mut self) -> u64 {
        self.0 += 1;
        self.0
    }
}

/// A store in every state a compaction meets: a compacted base under a
/// delta, tombstones in both, redirected endpoints and moved adjacency runs,
/// self edges and parallel edges.
fn random_store(rng: &mut Rng) -> Store {
    let mut graph = Store::new();
    let mut minter = Minter(0);
    let rounds = 1 + rng.below(3);
    for round in 0..rounds {
        for _ in 0..rng.below(20) {
            graph.add_node(minter.next());
        }
        let live: Vec<NodeId> = graph.node_ids().collect();
        if !live.is_empty() {
            for _ in 0..rng.below(3 * live.len()) {
                let source = live[rng.below(live.len())];
                let target = live[rng.below(live.len())];
                graph.add_edge(source, target, minter.next());
            }
        }
        if round + 1 < rounds && rng.one_in(2) {
            graph.compact(NodeOrder::Insertion);
        }
    }

    let live: Vec<NodeId> = graph.node_ids().collect();
    let edges: Vec<EdgeId> = graph.edge_ids().collect();
    for edge in edges {
        match rng.below(10) {
            0 => {
                graph.remove_edge(edge);
            }
            1 => {
                graph.redirect_edge_source(edge, live[rng.below(live.len())]);
            }
            2 => {
                graph.redirect_edge_target(edge, live[rng.below(live.len())]);
            }
            _ => {}
        }
    }
    if live.len() >= 2 && rng.one_in(3) {
        let old = live[rng.below(live.len())];
        let new = live[rng.below(live.len())];
        if old != new {
            graph.move_adjacency(old, new, Direction::Forward);
        }
    }
    for node in live {
        if rng.one_in(6) {
            graph.remove_node(node);
        }
    }
    graph
}

/// Up to three roots drawn from the live nodes, repeats allowed.
fn random_roots(graph: &Store, rng: &mut Rng) -> Vec<NodeId> {
    let live: Vec<NodeId> = graph.node_ids().collect();
    if live.is_empty() {
        return Vec::new();
    }
    (0..rng.below(4))
        .map(|_| live[rng.below(live.len())])
        .collect()
}

/// The old identities in the order `order` numbers them, computed the plain
/// way: insertion by the slot order alone, breadth-first by one traversal
/// per root, reverse postorder by one traversal from a virtual root.
fn oracle(graph: &Store, order: NodeOrder<'_>) -> Vec<NodeId> {
    let mut sequence = Vec::new();
    let mut seen = vec![false; graph.node_bound()];
    let mut visit = |node: NodeId, sequence: &mut Vec<NodeId>| {
        if !mem::replace(&mut seen[node.index()], true) {
            sequence.push(node);
        }
    };
    match order {
        // The slot order every order ends with is the whole of this one.
        NodeOrder::Insertion => {}
        NodeOrder::BreadthFirst(roots) => {
            // A node reached through an earlier root's node was reached by
            // that root, so filtering each full traversal is the forest.
            for &root in roots {
                for node in breadth_first(graph, root, Direction::Forward) {
                    visit(node, &mut sequence);
                }
            }
        }
        NodeOrder::ReversePostorder(roots) => {
            let view = VirtualRoot::new(graph, roots);
            let ranked = reverse_postorder(&view, view.virtual_root(), Direction::Forward);
            for node in ranked.into_iter().skip(1) {
                visit(NodeId::from_index(node), &mut sequence);
            }
        }
        NodeOrder::Explicit(nodes) => {
            for &node in nodes {
                visit(node, &mut sequence);
            }
        }
    }
    for node in graph.node_ids() {
        visit(node, &mut sequence);
    }
    sequence
}

fn renumbered(renumbering: &Renumbering, node: NodeId) -> NodeId {
    renumbering
        .node(node)
        .expect("a live node survives compaction")
}

/// Check that `after` is `before` compacted under `renumbering`, with the
/// edge numbering and every adjacency order a plain compaction would give.
fn assert_image(before: &Store, after: &Store, renumbering: &Renumbering) {
    assert!(after.is_compact());
    assert_eq!(after.node_count(), before.node_count());
    assert_eq!(after.edge_count(), before.edge_count());

    let mut taken = vec![false; after.node_bound()];
    for slot in 0..before.node_bound() {
        let old = NodeId::from_index(slot);
        assert_eq!(renumbering.node(old).is_some(), before.contains_node(old));
        if let Some(new) = renumbering.node(old) {
            assert!(
                !mem::replace(&mut taken[new.index()], true),
                "{new} is taken twice"
            );
            assert_eq!(after.node(new), before.node(old), "{old} lost its payload");
        }
    }

    let plain = before.clone().compact(NodeOrder::Insertion);
    let mut previous = None;
    for slot in 0..before.edge_bound() {
        let old = EdgeId::from_index(slot);
        let new = renumbering.edge(old);
        assert_eq!(
            new,
            plain.edge(old),
            "{old} is numbered as a plain compaction would"
        );
        let Some(new) = new else { continue };
        assert!(previous < Some(new), "edge identities keep their order");
        previous = Some(new);
        let (was, is) = (before.edge(old), after.edge(new));
        assert_eq!(is.source(), renumbered(renumbering, was.source()));
        assert_eq!(is.target(), renumbered(renumbering, was.target()));
        assert_eq!(is.payload(), was.payload());
    }

    for old in before.node_ids() {
        let new = renumbered(renumbering, old);
        for direction in [Direction::Forward, Direction::Backward] {
            let expected: Vec<EdgeId> = before
                .adjacent(old, direction)
                .map(|edge| renumbering.edge(edge).expect("a live edge survives"))
                .collect();
            assert_eq!(
                after.adjacent(new, direction).collect::<Vec<_>>(),
                expected,
                "{direction:?} adjacency of {old} changed order"
            );
        }
    }
}

/// Mutate the compacted store, compact it again, and check the composed
/// renumbering against applying the two in turn.
fn assert_composes(before: &Store, mut after: Store, first: &Renumbering, rng: &mut Rng) {
    let live: Vec<NodeId> = after.node_ids().collect();
    if let Some(&node) = live.get(rng.below(live.len() + 1)) {
        after.remove_node(node);
    }
    let edges: Vec<EdgeId> = after.edge_ids().collect();
    if let Some(&edge) = edges.get(rng.below(edges.len() + 1)) {
        after.remove_edge(edge);
    }
    let second = after.compact(NodeOrder::Insertion);
    let composed = first.compose(&second);
    for slot in 0..before.node_bound() {
        let old = NodeId::from_index(slot);
        let direct = first.node(old).and_then(|middle| second.node(middle));
        assert_eq!(composed.node(old), direct);
        if let Some(new) = direct {
            assert_eq!(after.node(new), before.node(old));
        }
    }
    for slot in 0..before.edge_bound() {
        let old = EdgeId::from_index(slot);
        let direct = first.edge(old).and_then(|middle| second.edge(middle));
        assert_eq!(composed.edge(old), direct);
    }
}

#[test]
fn every_order_compacts_to_the_image_its_oracle_numbers() {
    let mut rng = Rng::new(0x86);
    for _ in 0..400 {
        let before = random_store(&mut rng);
        let roots = random_roots(&before, &mut rng);
        let mut shuffled: Vec<NodeId> = before.node_ids().collect();
        rng.shuffle(&mut shuffled);
        for order in [
            NodeOrder::Insertion,
            NodeOrder::BreadthFirst(&roots),
            NodeOrder::ReversePostorder(&roots),
            NodeOrder::Explicit(&shuffled),
        ] {
            let mut after = before.clone();
            let renumbering = after.compact(order);
            assert_image(&before, &after, &renumbering);
            for (position, old) in oracle(&before, order).into_iter().enumerate() {
                assert_eq!(
                    renumbered(&renumbering, old),
                    NodeId::from_index(position),
                    "{order:?} numbered {old} out of order"
                );
            }
            assert_composes(&before, after, &renumbering, &mut rng);
        }
    }
}

/// A reducible flow graph whose slot order is unrelated to its shape: a
/// random DAG over shuffled slots in which every node is reachable from the
/// root, plus back edges to dominators, which keep it reducible.
fn reducible(rng: &mut Rng, size: usize) -> (Store, NodeId) {
    let mut graph = Store::new();
    let mut minter = Minter(0);
    let mut nodes: Vec<NodeId> = (0..size).map(|_| graph.add_node(minter.next())).collect();
    rng.shuffle(&mut nodes);
    for position in 1..size {
        let earlier = nodes[rng.below(position)];
        graph.add_edge(earlier, nodes[position], minter.next());
        if rng.one_in(2) {
            let other = nodes[rng.below(position)];
            graph.add_edge(other, nodes[position], minter.next());
        }
    }
    let root = nodes[0];
    let dominators = DominatorTree::compute(&Rooted::new(&graph, root));
    for _ in 0..size / 2 {
        let source = nodes[rng.below(size)];
        let mut target = source;
        while rng.one_in(2) {
            match dominators.idom(target) {
                Some(parent) => target = parent,
                None => break,
            }
        }
        graph.add_edge(source, target, minter.next());
    }
    (graph, root)
}

#[test]
fn reverse_postorder_points_every_forward_edge_up_the_numbering() {
    let mut rng = Rng::new(0x8600);
    for size in 1..120 {
        let (mut graph, root) = reducible(&mut rng, size);
        let dominators = DominatorTree::compute(&Rooted::new(&graph, root));
        assert!(is_reducible(&Rooted::new(&graph, root), &dominators));

        let renumbering = graph.compact(NodeOrder::ReversePostorder(&[root]));
        let root = renumbered(&renumbering, root);
        assert_eq!(root.index(), 0, "a single root is numbered first");
        let dominators = DominatorTree::compute(&Rooted::new(&graph, root));
        for edge in graph.edges() {
            let (source, target) = (edge.source(), edge.target());
            assert!(
                dominators.dominates(target, source) || source < target,
                "the forward edge {source} -> {target} points down the numbering"
            );
        }
    }
}

/// `a -> b`, `d -> a`, and `c` with no edges, added in that order.
fn sparse() -> (Store, [NodeId; 4]) {
    let mut graph = Store::new();
    let nodes = [10, 11, 12, 13].map(|payload| graph.add_node(payload));
    let [a, b, _, d] = nodes;
    graph.add_edge(a, b, 0);
    graph.add_edge(d, a, 1);
    (graph, nodes)
}

fn payloads(graph: &Store) -> Vec<u64> {
    graph.nodes().copied().collect()
}

/// The payloads of a copy of `graph` compacted in `order`, in identity order.
fn ordered(graph: &Store, order: NodeOrder<'_>) -> Vec<u64> {
    let mut graph = graph.clone();
    graph.compact(order);
    payloads(&graph)
}

#[test]
fn nodes_an_order_does_not_reach_follow_in_slot_order() {
    let (graph, [a, _, _, d]) = sparse();
    assert_eq!(ordered(&graph, NodeOrder::Insertion), [10, 11, 12, 13]);
    assert_eq!(
        ordered(&graph, NodeOrder::BreadthFirst(&[a])),
        [10, 11, 12, 13]
    );
    assert_eq!(
        ordered(&graph, NodeOrder::BreadthFirst(&[d])),
        [13, 10, 11, 12]
    );
    assert_eq!(
        ordered(&graph, NodeOrder::ReversePostorder(&[a])),
        [10, 11, 12, 13]
    );
    assert_eq!(
        ordered(&graph, NodeOrder::BreadthFirst(&[])),
        [10, 11, 12, 13]
    );
}

#[test]
fn several_roots_search_in_turn_and_repeats_add_nothing() {
    let (graph, [a, b, c, d]) = sparse();
    assert_eq!(
        ordered(&graph, NodeOrder::BreadthFirst(&[c, a, c, d])),
        [12, 10, 11, 13]
    );
    // The search from the later root finishes last, so it comes first.
    assert_eq!(
        ordered(&graph, NodeOrder::ReversePostorder(&[a, d, b])),
        [13, 10, 11, 12]
    );
    assert_eq!(
        ordered(&graph, NodeOrder::ReversePostorder(&[c, d])),
        [13, 10, 11, 12]
    );
}

#[test]
fn an_explicit_order_numbers_each_node_by_its_position() {
    let (mut graph, [a, b, c, d]) = sparse();
    graph.remove_node(b);
    let renumbering = graph.compact(NodeOrder::Explicit(&[d, c, a]));
    assert_eq!(payloads(&graph), [13, 12, 10]);
    assert_eq!(renumbering.node(b), None);
    assert_eq!(renumbering.node(d), Some(NodeId::from_index(0)));
    assert_eq!(
        graph.successors(NodeId::from_index(0)).collect::<Vec<_>>(),
        [NodeId::from_index(2)]
    );
}

#[test]
fn an_empty_store_compacts_in_every_order() {
    for order in [
        NodeOrder::Insertion,
        NodeOrder::BreadthFirst(&[]),
        NodeOrder::ReversePostorder(&[]),
        NodeOrder::Explicit(&[]),
    ] {
        let mut graph = Store::new();
        let renumbering = graph.compact(order);
        assert!(graph.is_compact());
        assert!(graph.is_empty());
        assert!(renumbering.is_identity());
    }

    let (mut graph, nodes) = sparse();
    for node in nodes {
        graph.remove_node(node);
    }
    let renumbering = graph.compact(NodeOrder::BreadthFirst(&[]));
    assert!(graph.is_empty() && graph.is_compact());
    assert!(nodes.iter().all(|&node| renumbering.node(node).is_none()));
}

#[test]
fn a_node_added_after_an_ordered_compaction_follows_every_compacted_one() {
    let (mut graph, [a, _, _, d]) = sparse();
    let renumbering = graph.compact(NodeOrder::BreadthFirst(&[d]));
    let added = graph.add_node(14);
    assert_eq!(added.index(), 4);
    graph.add_edge(renumbered(&renumbering, a), added, 2);
    assert_eq!(payloads(&graph), [13, 10, 11, 12, 14]);
    assert_eq!(
        graph
            .successors(renumbered(&renumbering, a))
            .collect::<Vec<_>>(),
        [NodeId::from_index(2), added]
    );
}

#[test]
#[should_panic(expected = "root node has been removed")]
fn a_removed_root_panics() {
    let (mut graph, [a, ..]) = sparse();
    graph.remove_node(a);
    graph.compact(NodeOrder::BreadthFirst(&[a]));
}

#[test]
#[should_panic(expected = "root node is out of range")]
fn a_root_out_of_range_panics() {
    let (mut graph, _) = sparse();
    graph.compact(NodeOrder::ReversePostorder(&[NodeId::from_index(4)]));
}

#[test]
#[should_panic(expected = "an explicit order names node n0 twice")]
fn an_explicit_order_naming_a_node_twice_panics() {
    let (mut graph, [a, b, c, d]) = sparse();
    graph.compact(NodeOrder::Explicit(&[a, b, a, c, d]));
}

#[test]
#[should_panic(expected = "an explicit order must name every live node")]
fn an_explicit_order_leaving_a_node_out_panics() {
    let (mut graph, [a, b, c, _]) = sparse();
    graph.compact(NodeOrder::Explicit(&[a, b, c]));
}

#[test]
#[should_panic(expected = "explicitly ordered node has been removed")]
fn an_explicit_order_naming_a_removed_node_panics() {
    let (mut graph, [a, b, c, d]) = sparse();
    graph.remove_node(b);
    graph.compact(NodeOrder::Explicit(&[a, b, c, d]));
}

#[test]
fn a_rejected_order_leaves_the_store_unchanged() {
    let (mut graph, [a, b, c, _]) = sparse();
    graph.remove_edge(EdgeId::from_index(0));
    let before = graph.clone();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        graph.compact(NodeOrder::Explicit(&[a, b, c]))
    }));
    assert!(result.is_err());
    assert_eq!(graph, before);
}
