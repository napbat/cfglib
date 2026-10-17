//! A directed multigraph as an edge list with brute-force walks over it.

use std::collections::VecDeque;

use cfglib::{Graph, NodeId};

use crate::Lcg;

/// Edges in insertion order, plus the adjacency lists a store keeps for them: each node's
/// successors and predecessors in the order the edges arrived.
pub(crate) struct Digraph {
    pub(crate) nodes: usize,
    pub(crate) edges: Vec<(usize, usize)>,
    outgoing: Vec<Vec<usize>>,
    incoming: Vec<Vec<usize>>,
}

impl Digraph {
    pub(crate) fn new(nodes: usize, edges: Vec<(usize, usize)>) -> Self {
        let mut outgoing = vec![Vec::new(); nodes];
        let mut incoming = vec![Vec::new(); nodes];
        for &(source, target) in &edges {
            outgoing[source].push(target);
            incoming[target].push(source);
        }
        Self {
            nodes,
            edges,
            outgoing,
            incoming,
        }
    }

    /// Random edges, so parallel edges and self loops occur.
    pub(crate) fn random(rng: &mut Lcg, nodes: usize, edges: usize) -> Self {
        let edges = (0..edges)
            .map(|_| (rng.below(nodes), rng.below(nodes)))
            .collect();
        Self::new(nodes, edges)
    }

    /// The same edges as a store.
    pub(crate) fn graph(&self) -> Graph<(), ()> {
        let mut graph = Graph::new();
        let nodes: Vec<NodeId> = (0..self.nodes).map(|_| graph.add_node(())).collect();
        for &(source, target) in &self.edges {
            graph.add_edge(nodes[source], nodes[target], ());
        }
        graph
    }

    /// The neighbors of `node` along the edges (`forward`) or against them.
    pub(crate) fn neighbors(&self, node: usize, forward: bool) -> &[usize] {
        if forward {
            &self.outgoing[node]
        } else {
            &self.incoming[node]
        }
    }

    /// Hop counts from `start`, `None` where unreachable or beyond `limit` hops.
    pub(crate) fn distances(
        &self,
        start: usize,
        forward: bool,
        limit: Option<usize>,
    ) -> Vec<Option<usize>> {
        self.discovery(start, forward, limit).1
    }

    /// The order a breadth-first walk from `start` first reaches each node.
    pub(crate) fn discovery_order(
        &self,
        start: usize,
        forward: bool,
        limit: Option<usize>,
    ) -> Vec<usize> {
        self.discovery(start, forward, limit).0
    }

    fn discovery(
        &self,
        start: usize,
        forward: bool,
        limit: Option<usize>,
    ) -> (Vec<usize>, Vec<Option<usize>>) {
        let mut distance = vec![None; self.nodes];
        let mut order = vec![start];
        distance[start] = Some(0);
        let mut queue = VecDeque::from([start]);
        while let Some(node) = queue.pop_front() {
            let steps = distance[node].expect("a queued node has a distance");
            if limit.is_some_and(|limit| steps >= limit) {
                continue;
            }
            for &next in self.neighbors(node, forward) {
                if distance[next].is_none() {
                    distance[next] = Some(steps + 1);
                    order.push(next);
                    queue.push_back(next);
                }
            }
        }
        (order, distance)
    }
}
