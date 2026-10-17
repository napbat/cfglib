//! The tombstone oracle: a seeded graph that went through appends, removals and redirections
//! answers every traversal question the same way before and after `compact`, once the answers
//! are carried through the `Renumbering`.
//!
//! Two independent checks run over the same scenarios. A plain reference model, written here from
//! the documented contract, predicts the store's adjacency order exactly. The store's own answers
//! before `compact` are then translated to the new identities and compared with its answers after.

use std::collections::BTreeMap;

use cfglib::{
    EdgeId, Graph, NodeId, Renumbering, TraversalDirection, breadth_first, depth_first_preorder,
    reachable, shortest_path,
};

use crate::reference_model::Reference;
use crate::{Relation, Symbol, relation, symbol};

/// A linear congruential generator, so every scenario is reproducible without a dependency.
struct Lcg(u64);

impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }

    fn below(&mut self, bound: usize) -> usize {
        let bound = u64::try_from(bound).expect("the bound fits in u64");
        usize::try_from(self.next_u64() % bound).expect("a value below the bound fits in usize")
    }
}

type Store = Graph<Symbol, Relation>;

fn add_node(graph: &mut Store, reference: &mut Reference) -> usize {
    let slot = reference.node_live.len();
    reference.add_node(slot);
    let id = graph.add_node(symbol(&format!("n{slot}")));
    assert_eq!(id.index(), slot);
    slot
}

fn add_edge(graph: &mut Store, reference: &mut Reference, source: usize, target: usize) {
    let slot = reference.edges.len();
    reference.add_edge(source, target, slot);
    let id = graph.add_edge(
        NodeId::from_index(source),
        NodeId::from_index(target),
        relation(&format!("e{slot}")),
    );
    assert_eq!(id.index(), slot);
}

/// Thirty nodes and ninety edges compacted into the base, then ten more nodes and forty more edges
/// in the delta, then removals and (optionally) redirections.
fn scenario(seed: u64, redirects: bool) -> (Store, Reference) {
    let mut rng = Lcg(seed);
    let mut graph = Store::new();
    let mut reference = Reference::new();

    for _ in 0..30 {
        add_node(&mut graph, &mut reference);
    }
    add_edge(&mut graph, &mut reference, 0, 0);
    add_edge(&mut graph, &mut reference, 1, 2);
    add_edge(&mut graph, &mut reference, 1, 2);
    for _ in 0..90 {
        let (source, target) = (rng.below(30), rng.below(30));
        add_edge(&mut graph, &mut reference, source, target);
    }
    assert!(graph.compact().is_identity(), "nothing was removed yet");

    for _ in 0..10 {
        add_node(&mut graph, &mut reference);
    }
    for _ in 0..40 {
        let (source, target) = (rng.below(40), rng.below(40));
        add_edge(&mut graph, &mut reference, source, target);
    }

    for _ in 0..6 {
        let node = rng.below(40);
        assert_eq!(
            graph.remove_node(NodeId::from_index(node)),
            reference.remove_node(node)
        );
    }
    for _ in 0..12 {
        let edge = rng.below(reference.edges.len());
        assert_eq!(
            graph.remove_edge(EdgeId::from_index(edge)),
            reference.remove_edge(edge)
        );
    }
    if redirects {
        let nodes = reference.live_nodes();
        for step in 0..14 {
            let edges = reference.live_edges();
            let edge = edges[rng.below(edges.len())];
            let endpoint = nodes[rng.below(nodes.len())];
            let source_side = step % 2 == 0;
            let (id, node) = (EdgeId::from_index(edge), NodeId::from_index(endpoint));
            if source_side {
                graph.redirect_edge_source(id, node);
            } else {
                graph.redirect_edge_target(id, node);
            }
            reference.redirect(edge, endpoint, source_side);
        }
    }
    (graph, reference)
}

fn indexes(nodes: Vec<NodeId>) -> Vec<usize> {
    nodes.into_iter().map(NodeId::index).collect()
}

fn same_sequence(exact_order: bool, mut actual: Vec<usize>, mut expected: Vec<usize>) {
    if !exact_order {
        actual.sort_unstable();
        expected.sort_unstable();
    }
    assert_eq!(actual, expected);
}

/// Every traversal answer the store gives at a node, in the store's own identities.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Observation {
    successors: Vec<usize>,
    predecessors: Vec<usize>,
    out_degree: usize,
    in_degree: usize,
    breadth_first: Vec<usize>,
    breadth_first_backward: Vec<usize>,
    preorder: Vec<usize>,
    reachable: Vec<bool>,
    distances: Vec<Option<usize>>,
}

fn observe(graph: &Store) -> BTreeMap<usize, Observation> {
    graph
        .node_ids()
        .map(|node| {
            let distances = (0..graph.node_bound())
                .map(|slot| {
                    let target = NodeId::from_index(slot);
                    if !graph.contains_node(target) {
                        return None;
                    }
                    shortest_path(graph, node, target, TraversalDirection::Outgoing)
                        .map(|path| path.len() - 1)
                })
                .collect();
            let observation = Observation {
                successors: indexes(graph.successors(node).collect()),
                predecessors: indexes(graph.predecessors(node).collect()),
                out_degree: graph.out_degree(node),
                in_degree: graph.in_degree(node),
                breadth_first: indexes(breadth_first(graph, node, TraversalDirection::Outgoing)),
                breadth_first_backward: indexes(breadth_first(
                    graph,
                    node,
                    TraversalDirection::Incoming,
                )),
                preorder: indexes(depth_first_preorder(
                    graph,
                    node,
                    TraversalDirection::Outgoing,
                )),
                reachable: reachable(graph, [node], TraversalDirection::Outgoing),
                distances,
            };
            (node.index(), observation)
        })
        .collect()
}

/// An observation made before a compaction, restated in the identities after it.
fn translate(
    observation: &Observation,
    renumbering: &Renumbering,
    new_bound: usize,
) -> Observation {
    let map = |old: usize| {
        renumbering
            .node(NodeId::from_index(old))
            .expect("every node an observation mentions is live")
            .index()
    };
    let map_all = |olds: &[usize]| olds.iter().map(|&old| map(old)).collect::<Vec<_>>();
    let mut reachable = vec![false; new_bound];
    let mut distances = vec![None; new_bound];
    for (old, &yes) in observation.reachable.iter().enumerate() {
        if yes {
            reachable[map(old)] = true;
        }
    }
    for (old, &distance) in observation.distances.iter().enumerate() {
        if distance.is_some() {
            distances[map(old)] = distance;
        }
    }
    Observation {
        successors: map_all(&observation.successors),
        predecessors: map_all(&observation.predecessors),
        out_degree: observation.out_degree,
        in_degree: observation.in_degree,
        breadth_first: map_all(&observation.breadth_first),
        breadth_first_backward: map_all(&observation.breadth_first_backward),
        preorder: map_all(&observation.preorder),
        reachable,
        distances,
    }
}

/// Sort the order-sensitive fields, for a comparison that must not depend on adjacency order.
fn unordered(mut observation: Observation) -> Observation {
    observation.successors.sort_unstable();
    observation.predecessors.sort_unstable();
    observation.breadth_first.sort_unstable();
    observation.breadth_first_backward.sort_unstable();
    observation.preorder.sort_unstable();
    observation
}

fn assert_matches_reference(graph: &Store, reference: &Reference, exact_order: bool) {
    let live = reference.live_nodes();
    assert_eq!(indexes(graph.node_ids().collect()), live);
    assert_eq!(
        graph.edge_ids().map(EdgeId::index).collect::<Vec<_>>(),
        reference.live_edges()
    );
    assert_eq!(graph.node_count(), live.len());
    assert_eq!(graph.node_bound(), reference.node_live.len());
    assert_eq!(graph.edge_count(), reference.live_edges().len());
    assert_eq!(graph.edge_bound(), reference.edges.len());

    for edge in reference.live_edges() {
        let (source, target) = reference.edges[edge].expect("live");
        let record = graph.edge(EdgeId::from_index(edge));
        assert_eq!(
            (record.source().index(), record.target().index()),
            (source, target)
        );
        assert_eq!(
            record.payload().label,
            format!("e{}", reference.edge_tag[edge])
        );
    }

    for &node in &live {
        let id = NodeId::from_index(node);
        assert_eq!(
            graph.node(id).name,
            format!("n{}", reference.node_tag[node])
        );
        same_sequence(
            exact_order,
            indexes(graph.successors(id).collect()),
            reference.neighbors(node, true),
        );
        same_sequence(
            exact_order,
            indexes(graph.predecessors(id).collect()),
            reference.neighbors(node, false),
        );
        assert_eq!(
            graph.out_degree(id),
            reference.outgoing[node].len(),
            "out degree of n{node}"
        );
        assert_eq!(
            graph.in_degree(id),
            reference.incoming[node].len(),
            "in degree of n{node}"
        );

        for forward in [true, false] {
            let direction = if forward {
                TraversalDirection::Outgoing
            } else {
                TraversalDirection::Incoming
            };
            let mut seen = vec![false; reference.node_live.len()];
            let mut preorder = Vec::new();
            reference.preorder(node, forward, &mut seen, &mut preorder);
            let reachable_set = reference.breadth_first(node, forward);
            same_sequence(
                exact_order,
                indexes(breadth_first(graph, id, direction)),
                reachable_set.clone(),
            );
            same_sequence(
                exact_order,
                indexes(depth_first_preorder(graph, id, direction)),
                preorder,
            );
            let mut expected = vec![false; reference.node_live.len()];
            for &member in &reachable_set {
                expected[member] = true;
            }
            assert_eq!(
                reachable(graph, [id], direction),
                expected,
                "reachable from n{node}"
            );
        }

        let distances = reference.distances(node);
        for &target in &live {
            let path = shortest_path(
                graph,
                id,
                NodeId::from_index(target),
                TraversalDirection::Outgoing,
            );
            let Some(expected) = distances[target] else {
                assert!(path.is_none(), "n{node} cannot reach n{target}");
                continue;
            };
            let path = indexes(path.expect("the reference reaches the target"));
            assert_eq!(
                path.len(),
                expected + 1,
                "length of the path n{node} to n{target}"
            );
            assert_eq!((path[0], path[path.len() - 1]), (node, target));
            for hop in path.windows(2) {
                assert!(
                    reference.neighbors(hop[0], true).contains(&hop[1]),
                    "n{} to n{} is not an edge",
                    hop[0],
                    hop[1]
                );
            }
        }
    }
}

const SEEDS: [u64; 4] = [1, 7, 0x00C0_FFEE, 20_251_017];

#[test]
fn the_scenarios_hold_the_shapes_the_oracle_is_meant_to_cover() {
    for seed in SEEDS {
        let (graph, reference) = scenario(seed, true);
        let live_edges = reference.live_edges();
        let pairs: Vec<_> = live_edges
            .iter()
            .map(|&edge| reference.edges[edge])
            .collect();
        let parallel = pairs
            .iter()
            .enumerate()
            .any(|(position, pair)| pairs[..position].contains(pair));
        assert!(parallel, "seed {seed} has no parallel edges");
        assert!(
            pairs
                .iter()
                .flatten()
                .any(|&(source, target)| source == target),
            "seed {seed} has no self loop"
        );
        assert!(
            graph.node_count() < graph.node_bound(),
            "seed {seed} removed no node"
        );
        assert!(
            graph.edge_count() < graph.edge_bound(),
            "seed {seed} removed no edge"
        );
        assert!(!graph.is_compact());

        let reaches: usize = observe(&graph)
            .values()
            .map(|observation| observation.breadth_first.len())
            .max()
            .expect("the scenario keeps nodes");
        assert!(
            reaches > 10,
            "seed {seed}: the largest reachable set has {reaches} nodes"
        );
        let deepest = observe(&graph)
            .values()
            .flat_map(|observation| observation.distances.iter().flatten().copied())
            .max()
            .expect("the scenario has paths");
        assert!(
            deepest >= 3,
            "seed {seed}: the longest shortest path has {deepest} hops"
        );
    }
}

#[test]
fn an_uncompacted_store_with_tombstones_and_redirects_matches_the_reference_exactly() {
    for seed in SEEDS {
        for redirects in [false, true] {
            let (graph, reference) = scenario(seed, redirects);
            assert_matches_reference(&graph, &reference, true);
        }
    }
}

#[test]
fn compaction_without_redirects_preserves_every_answer_exactly_and_matches_the_reference() {
    for seed in SEEDS {
        let (mut graph, reference) = scenario(seed, false);
        let before = observe(&graph);
        let renumbering = graph.compact();
        let after = observe(&graph);

        assert!(graph.is_compact());
        assert_eq!(after.len(), before.len());
        for (&old, observation) in &before {
            let new = renumbering
                .node(NodeId::from_index(old))
                .expect("a live node survives")
                .index();
            let expected = translate(observation, &renumbering, graph.node_bound());
            assert_eq!(after[&new], expected, "seed {seed}: n{old} became n{new}");
        }

        let (compacted, _, _) = reference.compacted();
        assert_matches_reference(&graph, &compacted, true);
    }
}

#[test]
fn compaction_after_redirects_preserves_every_answer_up_to_adjacency_order() {
    for seed in SEEDS {
        let (mut graph, reference) = scenario(seed, true);
        let before = observe(&graph);
        let renumbering = graph.compact();
        let after = observe(&graph);

        assert_eq!(after.len(), before.len());
        for (&old, observation) in &before {
            let new = renumbering
                .node(NodeId::from_index(old))
                .expect("a live node survives")
                .index();
            let expected = unordered(translate(observation, &renumbering, graph.node_bound()));
            assert_eq!(
                unordered(after[&new].clone()),
                expected,
                "seed {seed}: n{old} became n{new}"
            );
        }

        let (compacted, _, _) = reference.compacted();
        assert_matches_reference(&graph, &compacted, false);
    }
}

#[test]
fn the_renumbering_maps_exactly_the_survivors_the_reference_predicts() {
    for seed in SEEDS {
        for redirects in [false, true] {
            let (mut graph, reference) = scenario(seed, redirects);
            let (_, node_map, edge_map) = reference.compacted();
            let names: Vec<_> = (0..reference.node_live.len())
                .map(|slot| graph.node(NodeId::from_index(slot)).name.clone())
                .collect();
            let labels: Vec<_> = (0..reference.edges.len())
                .map(|slot| graph.edge(EdgeId::from_index(slot)).payload().label.clone())
                .collect();
            let renumbering = graph.compact();

            assert_eq!(renumbering.node_bound(), node_map.len());
            assert_eq!(renumbering.edge_bound(), edge_map.len());
            for (old, expected) in node_map.iter().enumerate() {
                let new = renumbering.node(NodeId::from_index(old));
                assert_eq!(new.map(NodeId::index), *expected, "node slot {old}");
                if let Some(new) = new {
                    assert_eq!(graph.node(new).name, names[old]);
                }
            }
            for (old, expected) in edge_map.iter().enumerate() {
                let new = renumbering.edge(EdgeId::from_index(old));
                assert_eq!(new.map(EdgeId::index), *expected, "edge slot {old}");
                if let Some(new) = new {
                    assert_eq!(graph.edge(new).payload().label, labels[old]);
                    let (source, target) = reference.edges[old].expect("a mapped edge is live");
                    let record = graph.edge(new);
                    assert_eq!(
                        record.source(),
                        renumbering.node(NodeId::from_index(source)).unwrap()
                    );
                    assert_eq!(
                        record.target(),
                        renumbering.node(NodeId::from_index(target)).unwrap()
                    );
                }
            }
        }
    }
}

#[test]
fn compacting_a_compacted_store_changes_no_identity_and_no_answer() {
    for seed in SEEDS {
        let (mut graph, _) = scenario(seed, true);
        graph.compact();
        let settled = observe(&graph);
        let again = graph.compact();
        assert!(again.is_identity());
        assert_eq!(observe(&graph), settled);
    }
}

#[test]
fn compaction_after_redirects_preserves_the_adjacency_order_redirection_produced() {
    for seed in SEEDS {
        let (mut graph, _) = scenario(seed, true);
        let before = observe(&graph);
        let renumbering = graph.compact();
        let after = observe(&graph);

        for (&old, observation) in &before {
            let new = renumbering
                .node(NodeId::from_index(old))
                .expect("a live node survives")
                .index();
            let expected = translate(observation, &renumbering, graph.node_bound());
            assert_eq!(after[&new], expected, "seed {seed}: n{old} became n{new}");
        }
    }
}
