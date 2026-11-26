use super::*;
use crate::graph::scc::SccNumbering;
use crate::graph::store::{Graph, NodeId};
use crate::test_util::rng::Rng;
use alloc::collections::{BTreeMap, BTreeSet};
use core::cell::RefCell;

/// A graph of `node_count` nodes and the given edges.
fn build(node_count: usize, edges: &[(usize, usize)]) -> Graph<(), ()> {
    let mut graph = Graph::new();
    for _ in 0..node_count {
        graph.add_node(());
    }
    for &(source, target) in edges {
        graph.add_edge(NodeId::from_index(source), NodeId::from_index(target), ());
    }
    graph
}

fn node(index: usize) -> NodeId {
    NodeId::from_index(index)
}

/// A view that records every node whose adjacency an algorithm reads, in
/// either direction.
struct Recording<'g> {
    graph: &'g Graph<(), ()>,
    read: RefCell<Vec<NodeId>>,
}

impl<'g> Recording<'g> {
    fn new(graph: &'g Graph<(), ()>) -> Self {
        Self {
            graph,
            read: RefCell::new(Vec::new()),
        }
    }

    fn take(&self) -> Vec<NodeId> {
        core::mem::take(&mut *self.read.borrow_mut())
    }
}

impl GraphView for Recording<'_> {
    type NodeId = NodeId;

    fn node_bound(&self) -> usize {
        self.graph.node_bound()
    }

    fn node_ids(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.graph.node_ids()
    }

    fn successors(&self, node: NodeId) -> impl Iterator<Item = NodeId> + '_ {
        self.read.borrow_mut().push(node);
        GraphView::successors(self.graph, node)
    }

    fn predecessors(&self, node: NodeId) -> impl Iterator<Item = NodeId> + '_ {
        self.read.borrow_mut().push(node);
        GraphView::predecessors(self.graph, node)
    }
}

/// Asserts that `components` holds the strongly connected components of
/// `graph` as the leaves-first decomposition finds them, in an order that
/// every edge agrees with and that lays the runs out back to back, and that
/// the snapshot keeps the sources-first numbering contract.
fn assert_current(graph: &Graph<(), ()>, components: &IncrementalScc<NodeId>) {
    let oracle = SccDecomposition::compute(graph, SccNumbering::LeavesFirst);
    for node in graph.node_ids() {
        assert_eq!(
            components.component_id(node),
            oracle.component(node).nodes()[0].index(),
            "the component of {node:?}"
        );
    }

    let snapshot = components.decomposition();
    assert_eq!(snapshot.len(), oracle.len());
    let mut position = 0;
    for nodes in snapshot.components() {
        assert_eq!(nodes, oracle.component(nodes[0]).nodes());
        assert_eq!(components.position(nodes[0].index()), position);
        position += nodes.len();
    }

    for source in graph.node_ids() {
        for target in GraphView::successors(graph, source) {
            let (from, to) = (
                components.component_id(source),
                components.component_id(target),
            );
            assert!(
                components.position(from) <= components.position(to),
                "{source:?} -> {target:?} runs against the order"
            );
            assert_eq!(
                components.position(from) == components.position(to),
                from == to
            );
            assert!(snapshot.component_index(source) <= snapshot.component_index(target));
        }
    }
}

/// The order as the public API shows it: each position's node with its
/// component id.
fn layout(components: &IncrementalScc<NodeId>) -> Vec<(NodeId, usize)> {
    components
        .decomposition()
        .components()
        .flatten()
        .map(|&node| (node, components.component_id(node)))
        .collect()
}

/// Asserts that `change` reports exactly how `after` differs from `before`.
fn assert_change(before: &[(NodeId, usize)], after: &[(NodeId, usize)], change: &SccChange) {
    assert_eq!(before.len(), after.len());
    let old_of: BTreeMap<NodeId, usize> = before.iter().copied().collect();
    let new_of: BTreeMap<NodeId, usize> = after.iter().copied().collect();
    let mut olds_by_new: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();
    let mut news_by_old: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();
    for (&node, &new) in &new_of {
        olds_by_new.entry(new).or_default().insert(old_of[&node]);
        news_by_old.entry(old_of[&node]).or_default().insert(new);
    }
    let merged: BTreeSet<usize> = olds_by_new
        .values()
        .filter(|olds| olds.len() > 1)
        .flatten()
        .copied()
        .collect();
    let split: BTreeSet<usize> = news_by_old
        .values()
        .filter(|news| news.len() > 1)
        .flatten()
        .copied()
        .collect();
    assert_eq!(change.merged, merged.into_iter().collect::<Vec<_>>());
    assert_eq!(change.split, split.into_iter().collect::<Vec<_>>());

    let rewritten: Vec<usize> = (0..before.len())
        .filter(|&position| before[position] != after[position])
        .collect();
    match (rewritten.first(), rewritten.last()) {
        (Some(&first), Some(&last)) => assert_eq!(change.reordered, first..last + 1),
        _ => assert_eq!(change.reordered, 0..0),
    }
}

/// A random multigraph of up to `max_nodes` slots, about one in six of them
/// removed again, with self-loops and parallel edges. Half the graphs start
/// acyclic.
fn random_graph(rng: &mut Rng, max_nodes: usize) -> Graph<(), ()> {
    let node_count = 1 + rng.below(max_nodes);
    let forward_only = rng.one_in(2);
    let mut graph = Graph::new();
    let nodes: Vec<NodeId> = (0..node_count).map(|_| graph.add_node(())).collect();
    for _ in 0..rng.below(2 * node_count) {
        let (mut source, mut target) = (rng.below(node_count), rng.below(node_count));
        if forward_only && source > target {
            (source, target) = (target, source);
        }
        graph.add_edge(nodes[source], nodes[target], ());
    }
    for &node in &nodes {
        if rng.one_in(6) {
            graph.remove_node(node);
        }
    }
    graph
}

/// Add a random edge between live nodes: a self-loop now and then, and a
/// parallel copy of an existing edge now and then.
fn random_edge(rng: &mut Rng, graph: &mut Graph<(), ()>, live: &[NodeId]) -> (NodeId, NodeId) {
    let existing: Vec<_> = graph.edge_ids().collect();
    let (source, target) = if !existing.is_empty() && rng.one_in(6) {
        let record = graph.edge(existing[rng.below(existing.len())]);
        (record.source(), record.target())
    } else {
        let source = live[rng.below(live.len())];
        let target = if rng.one_in(8) {
            source
        } else {
            live[rng.below(live.len())]
        };
        (source, target)
    };
    graph.add_edge(source, target, ());
    (source, target)
}

/// One random edit of `graph`, reported to `components`: a single insertion,
/// a batch of up to four, or the removal of a random edge.
fn random_edit(
    rng: &mut Rng,
    graph: &mut Graph<(), ()>,
    components: &mut IncrementalScc<NodeId>,
    live: &[NodeId],
) -> SccChange {
    match rng.below(3) {
        0 => {
            let (source, target) = random_edge(rng, graph, live);
            components.insert_edge(graph, source, target)
        }
        1 => {
            let batch: Vec<_> = (0..=rng.below(4))
                .map(|_| random_edge(rng, graph, live))
                .collect();
            components.insert_edges(graph, &batch)
        }
        _ => {
            let edges: Vec<_> = graph.edge_ids().collect();
            if edges.is_empty() {
                return components.insert_edges(graph, &[]);
            }
            let edge = edges[rng.below(edges.len())];
            let (source, target) = (graph.edge(edge).source(), graph.edge(edge).target());
            graph.remove_edge(edge);
            components.remove_edge(graph, source, target)
        }
    }
}

/// The changes and snapshots of one seeded run of random edits.
fn replay(seed: u64) -> Vec<(SccChange, SccDecomposition<NodeId>)> {
    let mut rng = Rng::new(seed);
    let mut graph = random_graph(&mut rng, 12);
    let live: Vec<NodeId> = graph.node_ids().collect();
    let mut components = IncrementalScc::compute(&graph);
    let mut run = Vec::new();
    if live.is_empty() {
        return run;
    }
    for _ in 0..24 {
        let change = random_edit(&mut rng, &mut graph, &mut components, &live);
        run.push((change, components.decomposition()));
    }
    run
}

/// Runs `steps` random edits for each seed, on a graph of up to `max_nodes`
/// slots, and checks the structure and the change against the oracle after
/// each. Returns how many updates merged components and how many split one.
fn check_random_edits(seeds: Range<u64>, max_nodes: usize, steps: usize) -> (usize, usize) {
    let (mut merges, mut splits) = (0, 0);
    for seed in seeds {
        let mut rng = Rng::new(seed);
        let mut graph = random_graph(&mut rng, max_nodes);
        let live: Vec<NodeId> = graph.node_ids().collect();
        let mut components = IncrementalScc::compute(&graph);
        assert_eq!(
            components.decomposition(),
            SccDecomposition::compute(&graph, SccNumbering::SourcesFirst)
        );
        assert_current(&graph, &components);
        if live.is_empty() {
            continue;
        }
        for _ in 0..steps {
            let before = layout(&components);
            let change = random_edit(&mut rng, &mut graph, &mut components, &live);
            assert_current(&graph, &components);
            assert_change(&before, &layout(&components), &change);
            merges += usize::from(!change.merged.is_empty());
            splits += usize::from(!change.split.is_empty());
        }
    }
    (merges, splits)
}

#[test]
fn random_edits_keep_the_components_and_a_topological_order() {
    for (seeds, max_nodes, steps) in [(0..400, 12, 24), (400..500, 48, 80)] {
        let (merges, splits) = check_random_edits(seeds, max_nodes, steps);
        // Enough of both that the runs exercise them, not only the O(1) cases.
        assert!(
            merges > 400 && splits > 100,
            "{merges} merges and {splits} splits"
        );
    }
}

#[test]
fn replaying_the_same_edits_gives_the_same_order_and_changes() {
    for seed in 0..50 {
        assert_eq!(replay(seed), replay(seed), "seed {seed}");
    }
}

#[test]
fn a_batch_is_one_walk_over_one_window() {
    // Two chains, `0 -> 1 -> 2` and `3 -> 4 -> 5`; a batch closes both into
    // cycles and links them.
    let mut graph = build(6, &[(0, 1), (1, 2), (3, 4), (4, 5)]);
    let mut components = IncrementalScc::compute(&graph);
    let batch = [(node(2), node(0)), (node(5), node(3)), (node(2), node(3))];
    for &(source, target) in &batch {
        graph.add_edge(source, target, ());
    }
    let recording = Recording::new(&graph);
    let change = components.insert_edges(&recording, &batch);
    assert_eq!(change.merged, [0, 1, 2, 3, 4, 5]);
    assert_current(&graph, &components);
    let read = recording.take();
    let distinct: BTreeSet<_> = read.iter().copied().collect();
    assert_eq!(distinct.len(), read.len(), "a node is read twice: {read:?}");
}

#[test]
fn a_batch_repairs_windows_apart_from_each_other_apart() {
    // A chain `0 -> 1 -> ... -> 9`. The batch closes `1..=3` and `6..=8`;
    // one walk over the stretch from 1 to 8 would read 4 and 5 as well.
    let mut graph = build(
        10,
        &(0..9).map(|index| (index, index + 1)).collect::<Vec<_>>(),
    );
    let mut components = IncrementalScc::compute(&graph);
    let batch = [(node(8), node(6)), (node(3), node(1))];
    for &(source, target) in &batch {
        graph.add_edge(source, target, ());
    }
    let recording = Recording::new(&graph);
    let change = components.insert_edges(&recording, &batch);
    assert_eq!(change.merged, [1, 2, 3, 6, 7, 8]);
    assert_eq!(change.reordered, 2..9);
    let read: BTreeSet<_> = recording.take().into_iter().collect();
    assert_eq!(read, BTreeSet::from([1, 2, 3, 6, 7, 8].map(node)));
    assert_current(&graph, &components);
}

#[test]
fn an_edge_that_agrees_with_the_order_reads_no_adjacency() {
    let mut graph = build(8, &[(0, 1), (1, 2), (2, 3), (3, 1), (3, 4), (5, 6)]);
    let mut components = IncrementalScc::compute(&graph);
    for (source, target) in [(0, 4), (1, 3), (2, 2), (6, 6)] {
        let (source, target) = (node(source), node(target));
        let from = components.position(components.component_id(source));
        let to = components.position(components.component_id(target));
        assert!(from <= to);
        graph.add_edge(source, target, ());
        let recording = Recording::new(&graph);
        assert_eq!(
            components.insert_edge(&recording, source, target),
            SccChange::default()
        );
        assert_eq!(recording.take(), Vec::<NodeId>::new());
    }
    // A removal between two components reads nothing either.
    let edge = graph
        .edge_ids()
        .find(|&edge| graph.edge(edge).source() == node(3) && graph.edge(edge).target() == node(4))
        .expect("the edge 3 -> 4");
    graph.remove_edge(edge);
    let recording = Recording::new(&graph);
    assert_eq!(
        components.remove_edge(&recording, node(3), node(4)),
        SccChange::default()
    );
    assert_eq!(recording.take(), Vec::<NodeId>::new());
    assert_current(&graph, &components);
}

/// The nodes whose components start between the positions of `target`'s
/// and `source`'s components, both included.
fn window(
    components: &IncrementalScc<NodeId>,
    graph: &Graph<(), ()>,
    source: NodeId,
    target: NodeId,
) -> BTreeSet<NodeId> {
    let first = components.position(components.component_id(target));
    let last = components.position(components.component_id(source));
    graph
        .node_ids()
        .filter(|&node| {
            let start = components.position(components.component_id(node));
            first <= start && start <= last
        })
        .collect()
}

#[test]
fn an_edge_against_the_order_reads_only_its_window() {
    // A chain `0 -> 1 -> ... -> 9`: the edge `6 -> 3` closes the cycle over
    // `3..=6`, and nothing before 3 or after 6 is read.
    let mut graph = build(
        10,
        &(0..9).map(|index| (index, index + 1)).collect::<Vec<_>>(),
    );
    let mut components = IncrementalScc::compute(&graph);
    let span = window(&components, &graph, node(6), node(3));
    assert_eq!(span, (3..=6).map(node).collect::<BTreeSet<_>>());
    graph.add_edge(node(6), node(3), ());
    let recording = Recording::new(&graph);
    let change = components.insert_edge(&recording, node(6), node(3));
    assert_eq!(change.merged, [3, 4, 5, 6]);
    let read: BTreeSet<_> = recording.take().into_iter().collect();
    assert_eq!(read, span);
    assert_current(&graph, &components);

    // Two chains, `0 -> 2 -> 4` and `1 -> 3 -> 5`, which Kosaraju orders odd
    // chain first. The edge `4 -> 1` spans every component, but the walk
    // reads only the odd chain, which `1` reaches, and moves it last.
    let mut graph = build(6, &[(0, 2), (2, 4), (1, 3), (3, 5)]);
    let mut components = IncrementalScc::compute(&graph);
    let span = window(&components, &graph, node(4), node(1));
    assert_eq!(span.len(), 6);
    graph.add_edge(node(4), node(1), ());
    let recording = Recording::new(&graph);
    let change = components.insert_edge(&recording, node(4), node(1));
    assert_eq!(change.merged, Vec::<usize>::new());
    let read: BTreeSet<_> = recording.take().into_iter().collect();
    assert_eq!(read, BTreeSet::from([1, 3, 5].map(node)));
    assert!(read.is_subset(&span));
    assert_current(&graph, &components);
    assert_eq!(change.reordered, 0..6);
}

#[test]
fn removing_one_of_two_parallel_edges_keeps_the_component() {
    let mut graph = build(3, &[(0, 1), (1, 0), (1, 0), (1, 2)]);
    let mut components = IncrementalScc::compute(&graph);
    let parallel = graph
        .edge_ids()
        .find(|&edge| graph.edge(edge).source() == node(1))
        .expect("an edge 1 -> 0");
    graph.remove_edge(parallel);
    assert_eq!(
        components.remove_edge(&graph, node(1), node(0)),
        SccChange::default()
    );
    assert_eq!(components.component_id(node(1)), 0);
    assert_current(&graph, &components);

    // The second copy was what held the cycle.
    let last = graph
        .edge_ids()
        .find(|&edge| graph.edge(edge).source() == node(1) && graph.edge(edge).target() == node(0))
        .expect("the other edge 1 -> 0");
    graph.remove_edge(last);
    let change = components.remove_edge(&graph, node(1), node(0));
    assert_eq!(change.split, [0, 1]);
    assert_eq!(change.reordered, 1..2);
    assert_current(&graph, &components);
}

#[test]
fn self_loops_change_nothing() {
    let mut graph = build(2, &[(0, 1)]);
    let mut components = IncrementalScc::compute(&graph);
    let edge = graph.add_edge(node(1), node(1), ());
    assert_eq!(
        components.insert_edge(&graph, node(1), node(1)),
        SccChange::default()
    );
    graph.remove_edge(edge);
    assert_eq!(
        components.remove_edge(&graph, node(1), node(1)),
        SccChange::default()
    );
    assert_current(&graph, &components);
}

#[test]
fn a_split_lays_the_parts_out_in_topological_order() {
    // The cycle `0 -> 1 -> 2 -> 0` with a chord `0 -> 2`, entered from 3 and
    // leaving to 4. Without `2 -> 0` it is the chain 0, 1, 2 with the chord.
    let mut graph = build(5, &[(3, 0), (0, 1), (1, 2), (2, 0), (0, 2), (2, 4)]);
    let mut components = IncrementalScc::compute(&graph);
    assert_eq!(components.position(3), 0);
    assert_eq!(components.position(0), 1);
    assert_eq!(components.position(4), 4);
    let back = graph
        .edge_ids()
        .find(|&edge| graph.edge(edge).source() == node(2) && graph.edge(edge).target() == node(0))
        .expect("the edge 2 -> 0");
    graph.remove_edge(back);
    let change = components.remove_edge(&graph, node(2), node(0));
    assert_eq!(change.split, [0, 1, 2]);
    assert_eq!(change.merged, Vec::<usize>::new());
    assert_eq!(
        [0, 1, 2, 3, 4].map(|id| components.position(id)),
        [1, 2, 3, 0, 4]
    );
    assert_current(&graph, &components);
}

#[test]
fn removed_slots_belong_to_no_component() {
    let mut graph = build(4, &[(0, 1), (1, 0), (1, 3)]);
    graph.remove_node(node(2));
    let mut components = IncrementalScc::compute(&graph);
    assert_current(&graph, &components);
    graph.add_edge(node(3), node(0), ());
    let change = components.insert_edge(&graph, node(3), node(0));
    assert_eq!(change.merged, [0, 3]);
    assert_eq!(components.decomposition().len(), 1);
    assert_current(&graph, &components);
}

#[test]
#[should_panic(expected = "is not a node of the graph the components were computed for")]
fn a_removed_slot_has_no_component_id() {
    let mut graph = build(3, &[(0, 1)]);
    graph.remove_node(node(2));
    let _ = IncrementalScc::compute(&graph).component_id(node(2));
}

#[test]
#[should_panic(expected = "is not a node of the graph the components were computed for")]
fn a_node_added_after_compute_panics() {
    let mut graph = build(2, &[(0, 1)]);
    let mut components = IncrementalScc::compute(&graph);
    let late = graph.add_node(());
    graph.add_edge(late, node(0), ());
    let _ = components.insert_edge(&graph, late, node(0));
}

#[test]
#[should_panic(expected = "is not the id of a component")]
fn a_position_needs_a_component_id() {
    let graph = build(2, &[(0, 1), (1, 0)]);
    let _ = IncrementalScc::compute(&graph).position(1);
}

#[test]
fn an_empty_graph_has_no_components() {
    let graph = build(0, &[]);
    let mut components = IncrementalScc::compute(&graph);
    assert!(components.decomposition().is_empty());
    assert_eq!(components.insert_edges(&graph, &[]), SccChange::default());
}
