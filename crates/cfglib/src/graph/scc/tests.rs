use super::*;
use crate::cfg::Cfg;
use crate::edge::EdgeKind;
use crate::graph::store::Graph;
use crate::test_util::ff;
use alloc::format;

/// The leaves-first decomposition of `graph`.
fn leaves_first<G: GraphView>(graph: &G) -> SccDecomposition<G::NodeId> {
    SccDecomposition::compute(graph, SccNumbering::LeavesFirst)
}

/// The sources-first decomposition of `graph`.
fn sources_first<G: GraphView>(graph: &G) -> SccDecomposition<G::NodeId> {
    SccDecomposition::compute(graph, SccNumbering::SourcesFirst)
}

#[test]
fn cfg_uses_generic_scc_algorithm() {
    let mut cfg = Cfg::new();
    let next = cfg.new_block();
    cfg.block_mut(cfg.entry()).push(ff("entry"));
    cfg.block_mut(next).push(ff("next"));
    cfg.add_edge(cfg.entry(), next, EdgeKind::Fallthrough);

    let result = leaves_first(&cfg);
    assert_eq!(result.component_count(), 2);
    assert!(result.is_dag(&cfg));
}

#[test]
fn condensation_collapses_cycles_to_a_dag() {
    // entry -> (a <-> b) -> exit
    let mut graph = Graph::<&str, ()>::new();
    let entry = graph.add_node("entry");
    let a = graph.add_node("a");
    let b = graph.add_node("b");
    let exit = graph.add_node("exit");
    graph.add_edge(entry, a, ());
    graph.add_edge(a, b, ());
    graph.add_edge(b, a, ());
    graph.add_edge(b, exit, ());

    let components = leaves_first(&graph);
    let condensed = components.condensation(&graph);
    assert_eq!(condensed.node_count(), 3);
    assert_eq!(condensed.edge_count(), 2, "cycle-internal edges collapse");
    assert!(leaves_first(&condensed).is_dag(&condensed));
    let cycle_component = condensed
        .node_ids()
        .find(|&node| components.components().nth(node.index()).map(<[_]>::len) == Some(2))
        .expect("the a/b component");
    assert_eq!(cycle_component.index(), components.component_index(a));
    assert_eq!(cycle_component.index(), components.component_index(b));
}

#[test]
fn directed_graph_cycle_forms_one_component() {
    let mut graph = Graph::<&str, ()>::new();
    let left = graph.add_node("left");
    let right = graph.add_node("right");
    graph.add_edge(left, right, ());
    graph.add_edge(right, left, ());

    let result = leaves_first(&graph);
    assert_eq!(result.component_count(), 1);
    assert!(result.component(left).contains(right));
    assert_eq!(result.component_index(left), result.component_index(right));
    assert!(!result.is_dag(&graph));
}

/// Shapes both algorithms must agree on: a chain, a diamond, a cycle with
/// a tail, two cycles in series, a self-loop beside a plain node, a
/// disconnected pair, and the empty graph.
fn fixtures() -> Vec<(&'static str, Graph<(), ()>)> {
    fn build(node_count: usize, edges: &[(usize, usize)]) -> Graph<(), ()> {
        let mut graph = Graph::<(), ()>::new();
        for _ in 0..node_count {
            graph.add_node(());
        }
        for &(from, to) in edges {
            graph.add_edge(NodeId::from_index(from), NodeId::from_index(to), ());
        }
        graph
    }

    vec![
        ("chain", build(3, &[(0, 1), (1, 2)])),
        ("diamond", build(4, &[(0, 1), (0, 2), (1, 3), (2, 3)])),
        (
            "cycle with a tail",
            build(4, &[(0, 1), (1, 2), (2, 1), (2, 3)]),
        ),
        (
            "two cycles in series",
            build(5, &[(0, 1), (1, 2), (2, 0), (1, 3), (3, 4), (4, 3)]),
        ),
        ("self loop", build(2, &[(0, 0)])),
        ("disconnected", build(4, &[(0, 1), (2, 3)])),
        ("empty", build(0, &[])),
    ]
}

fn partition(result: &SccDecomposition<NodeId>) -> BTreeSet<BTreeSet<NodeId>> {
    result
        .components()
        .map(|nodes| nodes.iter().copied().collect())
        .collect()
}

#[test]
fn kosaraju_and_tarjan_agree_on_the_partition() {
    for (name, graph) in fixtures() {
        let kosaraju = sources_first(&graph);
        let tarjan = leaves_first(&graph);
        assert_eq!(partition(&kosaraju), partition(&tarjan), "{name}");
        assert_eq!(
            kosaraju.component_count(),
            tarjan.component_count(),
            "{name}"
        );
        assert_eq!(kosaraju.is_dag(&graph), tarjan.is_dag(&graph), "{name}");
        for node in graph.node_ids() {
            assert_eq!(
                kosaraju.component(node).nodes(),
                tarjan.component(node).nodes(),
                "{name}"
            );
        }
    }
}

#[test]
fn kosaraju_numbers_components_topologically() {
    for (name, graph) in fixtures() {
        let kosaraju = sources_first(&graph);
        let tarjan = leaves_first(&graph);
        for edge in graph.edges() {
            let (from, to) = (edge.source(), edge.target());
            let (source, target) = (kosaraju.component_index(from), kosaraju.component_index(to));
            if kosaraju.component(from).contains(to) {
                assert_eq!(source, target, "{name}: an intra-component edge");
                continue;
            }
            // Sources first: an edge always runs from a lower component
            // index to a higher one.
            assert!(source < target, "{name}: {source} !< {target}");
            // Tarjan numbers the very same edge the other way round.
            assert!(
                tarjan.component_index(from) > tarjan.component_index(to),
                "{name}"
            );
        }
    }
}

/// `a <-> b <-> c` is a cycle, `b` enters the `d <-> e` cycle, `f` enters
/// `a`, and `g` is isolated. A branching frame (`b` has two successors)
/// makes this the shape that catches a depth-first walk which loses a
/// frame's remaining successors when a sibling subtree finishes.
fn nested() -> (Graph<&'static str, ()>, [NodeId; 7]) {
    let mut graph = Graph::<&str, ()>::new();
    let outer_a = graph.add_node("a");
    let outer_b = graph.add_node("b");
    let outer_c = graph.add_node("c");
    let inner_d = graph.add_node("d");
    let inner_e = graph.add_node("e");
    let entry_f = graph.add_node("f");
    let lone_g = graph.add_node("g");
    graph.add_edge(outer_a, outer_b, ());
    graph.add_edge(outer_b, outer_c, ());
    graph.add_edge(outer_c, outer_a, ());
    graph.add_edge(outer_b, inner_d, ());
    graph.add_edge(inner_d, inner_e, ());
    graph.add_edge(inner_e, inner_d, ());
    graph.add_edge(entry_f, outer_a, ());
    (
        graph,
        [outer_a, outer_b, outer_c, inner_d, inner_e, entry_f, lone_g],
    )
}

/// The component sequence, as payload names.
fn sequence(
    result: &SccDecomposition<NodeId>,
    graph: &Graph<&'static str, ()>,
) -> Vec<Vec<&'static str>> {
    result
        .components()
        .map(|nodes| nodes.iter().map(|&node| *graph.node(node)).collect())
        .collect()
}

#[test]
fn kosaraju_numbering_matches_the_hand_computed_two_pass_order() {
    let (graph, [_, outer_b, _, _, inner_e, entry_f, lone_g]) = nested();

    // Pass 1 finishes [c, e, d, b, a, f, g], so pass 2 walks that in
    // reverse and mints components from the roots g, f, a, d — the
    // isolated node first because it finished last.
    let result = sources_first(&graph);
    assert_eq!(
        sequence(&result, &graph),
        vec![vec!["g"], vec!["f"], vec!["a", "b", "c"], vec!["d", "e"]]
    );
    assert_eq!(result.component_index(lone_g), 0);
    assert_eq!(result.component_index(entry_f), 1);
    assert_eq!(result.component_index(outer_b), 2);
    assert_eq!(result.component_index(inner_e), 3);
}

#[test]
fn tarjan_numbering_matches_the_hand_computed_leaves_first_order() {
    let (graph, [outer_a, _, _, _, inner_e, entry_f, lone_g]) = nested();

    // One pass from `a`: `c` closes the cycle back to `a` without minting,
    // then `b`'s second successor `d` mints the inner cycle first, then
    // `a` mints its own. `f` and `g` follow as later roots, in node order.
    let result = leaves_first(&graph);
    assert_eq!(
        sequence(&result, &graph),
        vec![vec!["d", "e"], vec!["a", "b", "c"], vec!["f"], vec!["g"]]
    );
    assert_eq!(result.component_index(inner_e), 0);
    assert_eq!(result.component_index(outer_a), 1);
    assert_eq!(result.component_index(entry_f), 2);
    assert_eq!(result.component_index(lone_g), 3);
}

/// The component pairs of a condensation, as a set.
fn condensed_pairs(dag: &Graph<(), ()>) -> BTreeSet<(usize, usize)> {
    dag.edges()
        .map(|edge| (edge.source().index(), edge.target().index()))
        .collect()
}

#[test]
fn condensation_numbers_its_nodes_by_the_decomposition() {
    for (name, graph) in fixtures() {
        for (algorithm, components) in [
            ("tarjan", leaves_first(&graph)),
            ("kosaraju", sources_first(&graph)),
        ] {
            let dag = components.condensation(&graph);
            assert_eq!(
                dag.node_count(),
                components.component_count(),
                "{name}/{algorithm}"
            );

            // Node index IS component index: every graph edge crossing two
            // components appears as that pair, and nothing else does.
            let expected: BTreeSet<(usize, usize)> = graph
                .edges()
                .map(|edge| {
                    (
                        components.component_index(edge.source()),
                        components.component_index(edge.target()),
                    )
                })
                .filter(|&(from, to)| from != to)
                .collect();
            assert_eq!(condensed_pairs(&dag), expected, "{name}/{algorithm}");
            assert!(leaves_first(&dag).is_dag(&dag), "{name}/{algorithm}");
        }

        // And under sources-first numbering the DAG's own edges only ever
        // run upwards — the property a single ascending pass rides on.
        let dag = sources_first(&graph).condensation(&graph);
        for (from, to) in condensed_pairs(&dag) {
            assert!(from < to, "{name}: {from} !< {to}");
        }
    }
}

#[test]
fn condensation_deduplicates_parallel_and_repeated_component_edges() {
    // Two nodes of the same component both point into the other, one of
    // them twice: three graph edges, one component edge.
    let mut graph = Graph::<(), ()>::new();
    let left = graph.add_node(());
    let right = graph.add_node(());
    let sink = graph.add_node(());
    graph.add_edge(left, right, ());
    graph.add_edge(right, left, ());
    graph.add_edge(left, sink, ());
    graph.add_edge(left, sink, ());
    graph.add_edge(right, sink, ());

    let components = sources_first(&graph);
    let dag = components.condensation(&graph);
    assert_eq!(dag.node_count(), 2);
    assert_eq!(dag.edge_count(), 1);
    assert_eq!(
        condensed_pairs(&dag),
        [(
            components.component_index(left),
            components.component_index(sink)
        )]
        .into_iter()
        .collect()
    );
}

/// `a <-> b`, a removed slot `c`, and `d -> a`: the slot sits between live
/// nodes, so every dense table the decompositions build has a hole in it.
fn with_removed_slot() -> (Graph<(), ()>, [NodeId; 4]) {
    let mut graph = Graph::<(), ()>::new();
    let [a, b, c, d] = [(); 4].map(|()| graph.add_node(()));
    graph.add_edge(a, b, ());
    graph.add_edge(b, a, ());
    graph.add_edge(d, a, ());
    assert!(graph.remove_node(c));
    (graph, [a, b, c, d])
}

#[test]
fn both_algorithms_partition_only_the_live_nodes_around_a_removed_slot() {
    let (graph, [a, b, c, d]) = with_removed_slot();
    let expected = BTreeSet::from([BTreeSet::from([a, b]), BTreeSet::from([d])]);
    for (algorithm, result) in [
        ("tarjan", leaves_first(&graph)),
        ("kosaraju", sources_first(&graph)),
    ] {
        assert_eq!(partition(&result), expected, "{algorithm}");
        assert!(
            result.components().all(|nodes| !nodes.contains(&c)),
            "{algorithm}"
        );
        assert_eq!(result.component_index(a), result.component_index(b));
        assert_ne!(result.component_index(a), result.component_index(d));
    }
}

#[test]
#[should_panic(expected = "is not a node of the decomposed graph")]
fn kosaraju_component_index_rejects_a_removed_slot() {
    let (graph, [_, _, c, _]) = with_removed_slot();
    let _ = sources_first(&graph).component_index(c);
}

#[test]
#[should_panic(expected = "is not a node of the decomposed graph")]
fn component_of_a_removed_slot_is_not_the_first_component() {
    // With no live node there is no component 0 to borrow, so the query has
    // to fail at the slot rather than at an offset table.
    let mut graph = Graph::<(), ()>::new();
    let only = graph.add_node(());
    assert!(graph.remove_node(only));
    let _ = leaves_first(&graph).component(only);
}

#[test]
#[should_panic(expected = "the decomposition covers 4 nodes but the graph has 2")]
fn condensation_of_another_graph_panics() {
    let (four, _) = nested_pair();
    let mut two = Graph::<&str, ()>::new();
    let first = two.add_node("a");
    let second = two.add_node("b");
    two.add_edge(first, second, ());
    let _ = sources_first(&four).condensation(&two);
}

/// A four-node graph for the mismatch test: `a <-> b`, `c -> d`.
fn nested_pair() -> (Graph<&'static str, ()>, [NodeId; 4]) {
    let mut graph = Graph::<&str, ()>::new();
    let a = graph.add_node("a");
    let b = graph.add_node("b");
    let c = graph.add_node("c");
    let d = graph.add_node("d");
    graph.add_edge(a, b, ());
    graph.add_edge(b, a, ());
    graph.add_edge(c, d, ());
    (graph, [a, b, c, d])
}

#[test]
fn self_edge_is_not_a_dag() {
    let mut graph = Graph::<(), ()>::new();
    let node = graph.add_node(());
    graph.add_edge(node, node, ());
    let result = leaves_first(&graph);
    assert!(result.component(node).is_singleton());
    assert!(!result.is_dag(&graph));
}

#[test]
fn a_component_is_cyclic_when_it_has_several_nodes_or_a_self_edge() {
    // `plain`, `looped -> looped`, and the pair `a <-> b`.
    let mut graph = Graph::<(), ()>::new();
    let [plain, looped, a, b] = [(); 4].map(|()| graph.add_node(()));
    graph.add_edge(looped, looped, ());
    graph.add_edge(a, b, ());
    graph.add_edge(b, a, ());
    graph.add_edge(plain, a, ());

    for result in [leaves_first(&graph), sources_first(&graph)] {
        assert!(!result.component(plain).is_cyclic(&graph));
        assert!(result.component(looped).is_cyclic(&graph));
        assert!(result.component(a).is_cyclic(&graph));
        assert!(result.component(b).is_cyclic(&graph));
        assert!(!result.is_dag(&graph));
    }
}

#[test]
fn a_graph_is_a_dag_exactly_when_no_component_is_cyclic() {
    for (name, graph) in fixtures() {
        for result in [leaves_first(&graph), sources_first(&graph)] {
            let any_cyclic = result
                .components()
                .any(|nodes| Scc { nodes }.is_cyclic(&graph));
            assert_eq!(result.is_dag(&graph), !any_cyclic, "{name}");
        }
    }
}

#[test]
fn components_are_ascending_runs_of_one_flat_array() {
    for (name, graph) in fixtures() {
        for (algorithm, result) in [
            ("tarjan", leaves_first(&graph)),
            ("kosaraju", sources_first(&graph)),
        ] {
            let label = format!("{name}/{algorithm}");
            assert_eq!(
                result.components().len(),
                result.component_count(),
                "{label}"
            );
            assert_eq!(result.is_empty(), graph.node_count() == 0, "{label}");

            let mut seen = BTreeSet::new();
            for (index, nodes) in result.components().enumerate() {
                assert!(!nodes.is_empty(), "{label}");
                assert!(nodes.windows(2).all(|pair| pair[0] < pair[1]), "{label}");
                for &node in nodes {
                    assert!(seen.insert(node), "{label}: {node:?} in two components");
                    assert_eq!(result.component_index(node), index, "{label}");
                    assert_eq!(result.component(node).nodes(), nodes, "{label}");
                }
            }
            assert_eq!(seen.len(), graph.node_count(), "{label}");

            let forward: Vec<&[NodeId]> = result.components().collect();
            let mut backward: Vec<&[NodeId]> = result.components().rev().collect();
            backward.reverse();
            assert_eq!(forward, backward, "{label}");
        }
    }
}

/// A cycle entered at its highest node, so the order in which each
/// algorithm collects the members is not ascending.
fn descending_cycle() -> (Graph<(), ()>, [NodeId; 5]) {
    let mut graph = Graph::<(), ()>::new();
    let nodes = [(); 5].map(|()| graph.add_node(()));
    let [root, first, second, third, gap] = nodes;
    graph.add_edge(root, third, ());
    graph.add_edge(third, first, ());
    graph.add_edge(first, second, ());
    graph.add_edge(second, third, ());
    graph.add_edge(gap, root, ());
    (graph, nodes)
}

#[test]
fn component_nodes_are_ascending_whatever_order_they_were_found_in() {
    let (graph, [root, first, second, third, gap]) = descending_cycle();
    for (algorithm, result) in [
        ("tarjan", leaves_first(&graph)),
        ("kosaraju", sources_first(&graph)),
    ] {
        let cycle = result.component(first);
        assert_eq!(cycle.nodes(), [first, second, third], "{algorithm}");
        assert!(!cycle.is_singleton(), "{algorithm}");
        for member in [first, second, third] {
            assert!(cycle.contains(member), "{algorithm}");
        }
        for outsider in [root, gap] {
            assert!(!cycle.contains(outsider), "{algorithm}");
            assert!(result.component(outsider).is_singleton(), "{algorithm}");
        }
    }
}

#[test]
fn component_views_compare_and_print_by_their_nodes() {
    let (graph, [_, first, second, third, gap]) = descending_cycle();
    let result = leaves_first(&graph);
    let cycle = result.component(second);
    assert_eq!(cycle, result.component(third));
    assert_ne!(cycle, result.component(gap));
    let rendered = format!("{cycle:?}");
    assert!(rendered.starts_with("Scc { nodes: ["), "{rendered}");
    assert!(rendered.contains(&format!("{first:?}")), "{rendered}");

    let copy = cycle;
    assert_eq!(copy, cycle);
    assert_eq!(
        copy.nodes(),
        result
            .components()
            .nth(result.component_index(first))
            .unwrap()
    );
}

/// Tarjan's algorithm with its separate index, low-link and on-stack arrays,
/// as the leaves-first decomposition ran before it moved to Pearce's single array: the oracle
/// for the numbering, which both must give in the same completion order.
fn classic_tarjan(graph: &Graph<(), ()>) -> SccDecomposition<NodeId> {
    let node_count = graph.node_bound();
    let mut next_index = 0_usize;
    let mut stack = Vec::new();
    let mut on_stack = vec![false; node_count];
    let mut indices = vec![usize::MAX; node_count];
    let mut lowlinks = vec![0_usize; node_count];
    let mut result = SccDecomposition::with_node_bound(node_count);
    let mut calls: Vec<(NodeId, Vec<NodeId>, usize)> = Vec::new();
    for start in graph.node_ids() {
        if indices[start.index()] != usize::MAX {
            continue;
        }
        indices[start.index()] = next_index;
        lowlinks[start.index()] = next_index;
        next_index += 1;
        stack.push(start);
        on_stack[start.index()] = true;
        calls.push((start, graph.successors(start).collect(), 0));
        while let Some((node, successors, cursor)) = calls.last_mut() {
            let node = *node;
            if let Some(&successor) = successors.get(*cursor) {
                *cursor += 1;
                if indices[successor.index()] == usize::MAX {
                    indices[successor.index()] = next_index;
                    lowlinks[successor.index()] = next_index;
                    next_index += 1;
                    stack.push(successor);
                    on_stack[successor.index()] = true;
                    calls.push((successor, graph.successors(successor).collect(), 0));
                } else if on_stack[successor.index()] {
                    lowlinks[node.index()] = lowlinks[node.index()].min(indices[successor.index()]);
                }
                continue;
            }
            if lowlinks[node.index()] == indices[node.index()] {
                let component_index = result.component_count();
                while let Some(member) = stack.pop() {
                    on_stack[member.index()] = false;
                    result.component_of[member.index()] = word(component_index);
                    result.nodes.push(member);
                    if member == node {
                        break;
                    }
                }
                result.close_component();
            }
            calls.pop();
            if let Some((parent, _, _)) = calls.last() {
                lowlinks[parent.index()] = lowlinks[parent.index()].min(lowlinks[node.index()]);
            }
        }
    }
    result
}

/// Random graphs with self-loops, parallel edges and removed slots: the
/// single-array walk returns the classic walk's decomposition, field for
/// field, so the numbering, the ascending slices and the slots that hold no
/// node are all unchanged.
#[test]
fn the_single_array_walk_decomposes_like_the_classic_one() {
    let mut rng = crate::test_util::rng::Rng::new(69);
    for _ in 0..500 {
        let node_count = 1 + rng.below(16);
        let mut graph = Graph::<(), ()>::new();
        let nodes: Vec<NodeId> = (0..node_count).map(|_| graph.add_node(())).collect();
        for _ in 0..rng.below(3 * node_count + 1) {
            graph.add_edge(
                nodes[rng.below(node_count)],
                nodes[rng.below(node_count)],
                (),
            );
        }
        for &node in &nodes {
            if rng.one_in(6) {
                graph.remove_node(node);
            }
        }
        assert_eq!(leaves_first(&graph), classic_tarjan(&graph));
    }
}

/// Hops from `start` to `goal` along successors, by a plain breadth-first
/// walk over the whole graph.
fn hops(graph: &Graph<(), ()>, start: NodeId, goal: NodeId) -> Option<usize> {
    let mut distance = vec![usize::MAX; graph.node_bound()];
    distance[start.index()] = 0;
    let mut queue = VecDeque::from([start]);
    while let Some(node) = queue.pop_front() {
        if node == goal {
            return Some(distance[node.index()]);
        }
        for successor in graph.successors(node) {
            if distance[successor.index()] == usize::MAX {
                distance[successor.index()] = distance[node.index()] + 1;
                queue.push_back(successor);
            }
        }
    }
    None
}

/// Checks `cycle_through` for every live node of random multigraphs with
/// self-loops, parallel edges and removed slots: the cycle starts at the node,
/// closes, repeats no node, stays in the node's component, and is as short as
/// the shortest way back from one of the node's successors.
#[test]
fn cycle_through_is_a_shortest_cycle_inside_the_component() {
    let mut rng = crate::test_util::rng::Rng::new(68);
    let mut found = 0;
    for _ in 0..300 {
        let node_count = 1 + rng.below(14);
        let mut graph = Graph::<(), ()>::new();
        let nodes: Vec<NodeId> = (0..node_count).map(|_| graph.add_node(())).collect();
        for _ in 0..rng.below(2 * node_count + 1) {
            graph.add_edge(
                nodes[rng.below(node_count)],
                nodes[rng.below(node_count)],
                (),
            );
        }
        for &node in &nodes {
            if rng.one_in(6) {
                graph.remove_node(node);
            }
        }
        let (sources_first, leaves_first) = (sources_first(&graph), leaves_first(&graph));
        for node in graph.node_ids() {
            let shortest = graph
                .successors(node)
                .filter_map(|successor| hops(&graph, successor, node))
                .min()
                .map(|back| back + 1);
            let cycle = sources_first.cycle_through(&graph, node);
            assert_eq!(cycle, leaves_first.cycle_through(&graph, node));
            assert_eq!(cycle.as_ref().map(Vec::len), shortest, "{node:?}");
            let Some(cycle) = cycle else { continue };
            found += 1;
            assert_eq!(cycle[0], node);
            assert_eq!(cycle.iter().collect::<BTreeSet<_>>().len(), cycle.len());
            let component = sources_first.component(node);
            for (position, &member) in cycle.iter().enumerate() {
                assert!(component.contains(member));
                let next = cycle[(position + 1) % cycle.len()];
                assert!(graph.successors(member).any(|successor| successor == next));
            }
        }
    }
    assert!(found > 300, "only {found} nodes lie on a cycle");
}

#[test]
fn cycle_through_a_self_loop_is_the_node_alone() {
    let mut graph = Graph::<(), ()>::new();
    let [a, b] = [(); 2].map(|()| graph.add_node(()));
    graph.add_edge(a, b, ());
    graph.add_edge(b, b, ());
    graph.add_edge(b, b, ());
    let components = sources_first(&graph);
    assert_eq!(components.cycle_through(&graph, b), Some(vec![b]));
    assert_eq!(components.cycle_through(&graph, a), None);
}

#[test]
#[should_panic(expected = "is not a node of the decomposed graph")]
fn cycle_through_rejects_a_removed_slot() {
    let (graph, [_, _, c, _]) = with_removed_slot();
    let _ = sources_first(&graph).cycle_through(&graph, c);
}
