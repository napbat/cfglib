//! Kahn's order with a cycle witness in place of `None`.

extern crate alloc;
use alloc::collections::VecDeque;
use alloc::vec;
use alloc::vec::Vec;

use crate::graph::view::{DenseId, GraphView};

/// Return a topological ordering of the live nodes, or one directed cycle
/// among them as the witness that there is none.
///
/// `Ok` holds exactly the order [`topological_sort`](super::topological_sort)
/// returns. The witness costs no second pass: O(n + m) in all.
///
/// # Errors
///
/// Returns `Err` with a cycle when the live nodes contain one. The cycle is in
/// successor order, starting at its smallest node and not repeating it at the
/// end: each node has an edge to the next, and the last node has an edge to
/// the first. No node repeats, so the nodes lie in one strongly connected
/// component; a node with an edge to itself is a one-node cycle.
///
/// Kahn's algorithm leaves unsorted exactly the nodes on or after a cycle,
/// each with an unsorted predecessor. A walk from the smallest of them,
/// stepping to the first unsorted predecessor in
/// [`predecessors`](GraphView::predecessors) order, steps on some node twice,
/// and the steps in between, reversed, are the cycle. It is deterministic but
/// not necessarily the shortest:
/// [`SccDecomposition::cycle_through`](crate::SccDecomposition::cycle_through)
/// gives a shortest cycle through a chosen node.
///
/// # Examples
///
/// ```
/// use cfglib::{Graph, topological_sort_or_cycle};
///
/// // A module graph: `app` imports `ui`, which imports `base`.
/// let mut graph = Graph::<&str, ()>::new();
/// let app = graph.add_node("app");
/// let ui = graph.add_node("ui");
/// let base = graph.add_node("base");
/// graph.add_edge(app, ui, ());
/// graph.add_edge(ui, base, ());
/// assert_eq!(topological_sort_or_cycle(&graph), Ok(vec![app, ui, base]));
///
/// // `base` importing `ui` back is the cycle a diagnostic reports.
/// graph.add_edge(base, ui, ());
/// assert_eq!(topological_sort_or_cycle(&graph), Err(vec![ui, base]));
/// ```
pub fn topological_sort_or_cycle<G: GraphView>(
    graph: &G,
) -> Result<Vec<G::NodeId>, Vec<G::NodeId>> {
    let (order, live, mut incoming_counts) = kahn_order(graph);
    if order.len() == live {
        Ok(order)
    } else {
        Err(unsorted_cycle(graph, &mut incoming_counts))
    }
}

/// Kahn's algorithm: the order it outputs, the number of live nodes, and each
/// slot's count of predecessor entries that were never output.
///
/// The order covers every live node exactly when the graph is acyclic. The
/// queue takes every node whose count reaches zero, so afterwards a live node
/// was left unsorted exactly when its count is positive.
///
/// This is the loop of [`topological_sort`](super::topological_sort), which
/// keeps its own copy: routed through this core, it measured about 4% slower
/// on a 1,024-node chain. A change to one loop belongs in both.
fn kahn_order<G: GraphView>(graph: &G) -> (Vec<G::NodeId>, usize, Vec<usize>) {
    let mut incoming_counts = vec![0_usize; graph.node_bound()];
    let mut live = 0;
    for node in graph.node_ids() {
        incoming_counts[node.index()] = graph.predecessors(node).count();
        live += 1;
    }

    let mut queue: VecDeque<G::NodeId> = graph
        .node_ids()
        .filter(|node| incoming_counts[node.index()] == 0)
        .collect();
    let mut order = Vec::with_capacity(live);

    while let Some(node) = queue.pop_front() {
        order.push(node);
        for successor in graph.successors(node) {
            let count = &mut incoming_counts[successor.index()];
            *count -= 1;
            if *count == 0 {
                queue.push_back(successor);
            }
        }
    }

    (order, live, incoming_counts)
}

/// One cycle among the nodes [`kahn_order`] left unsorted, given its counts.
///
/// A node the walk steps on gets the count `ON_WALK`: still positive, so it
/// stays a valid next step, and stepping on it again ends the walk.
fn unsorted_cycle<G: GraphView>(graph: &G, incoming_counts: &mut [usize]) -> Vec<G::NodeId> {
    const ON_WALK: usize = usize::MAX;
    let mut node = graph
        .node_ids()
        .filter(|node| incoming_counts[node.index()] > 0)
        .min()
        .expect("Kahn's algorithm leaves a node unsorted only when the graph has a cycle");
    let mut walk = Vec::new();
    while incoming_counts[node.index()] != ON_WALK {
        incoming_counts[node.index()] = ON_WALK;
        walk.push(node);
        node = graph
            .predecessors(node)
            .find(|predecessor| incoming_counts[predecessor.index()] > 0)
            .expect("an unsorted node has a predecessor that is unsorted too");
    }
    let first = walk
        .iter()
        .position(|&step| step == node)
        .expect("the walk ends on a node it stepped on");
    let mut cycle = walk.split_off(first);
    cycle.reverse();
    let smallest = (0..cycle.len())
        .min_by_key(|&position| cycle[position])
        .expect("a cycle has a node");
    cycle.rotate_left(smallest);
    cycle
}
