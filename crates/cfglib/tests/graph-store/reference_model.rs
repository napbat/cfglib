//! The documented contract of the store, written as plain vectors.

use std::collections::VecDeque;

/// The documented contract written as plain vectors: every adjacency list is the live edges at a
/// node in the order they arrived there, and a redirected edge arrives at its new endpoint last.
pub(crate) struct Reference {
    pub(crate) node_live: Vec<bool>,
    /// The number in each node's payload name, fixed when the node was first added.
    pub(crate) node_tag: Vec<usize>,
    pub(crate) edges: Vec<Option<(usize, usize)>>,
    /// The number in each edge's payload label, fixed when the edge was first added.
    pub(crate) edge_tag: Vec<usize>,
    pub(crate) outgoing: Vec<Vec<usize>>,
    pub(crate) incoming: Vec<Vec<usize>>,
}

impl Reference {
    pub(crate) const fn new() -> Self {
        Self {
            node_live: Vec::new(),
            node_tag: Vec::new(),
            edges: Vec::new(),
            edge_tag: Vec::new(),
            outgoing: Vec::new(),
            incoming: Vec::new(),
        }
    }

    pub(crate) fn add_node(&mut self, tag: usize) -> usize {
        self.node_live.push(true);
        self.node_tag.push(tag);
        self.outgoing.push(Vec::new());
        self.incoming.push(Vec::new());
        self.node_live.len() - 1
    }

    pub(crate) fn add_edge(&mut self, source: usize, target: usize, tag: usize) -> usize {
        let slot = self.edges.len();
        self.edges.push(Some((source, target)));
        self.edge_tag.push(tag);
        self.outgoing[source].push(slot);
        self.incoming[target].push(slot);
        slot
    }

    pub(crate) fn remove_edge(&mut self, edge: usize) -> bool {
        let Some((source, target)) = self.edges[edge].take() else {
            return false;
        };
        self.outgoing[source].retain(|&slot| slot != edge);
        self.incoming[target].retain(|&slot| slot != edge);
        true
    }

    pub(crate) fn remove_node(&mut self, node: usize) -> bool {
        if !self.node_live[node] {
            return false;
        }
        let incident: Vec<usize> = self.outgoing[node]
            .iter()
            .chain(&self.incoming[node])
            .copied()
            .collect();
        for edge in incident {
            self.remove_edge(edge);
        }
        self.node_live[node] = false;
        true
    }

    pub(crate) fn redirect(&mut self, edge: usize, endpoint: usize, source_side: bool) {
        let (source, target) = self.edges[edge].expect("only a live edge is redirected");
        let (previous, lists) = if source_side {
            (source, &mut self.outgoing)
        } else {
            (target, &mut self.incoming)
        };
        if previous == endpoint {
            return;
        }
        lists[previous].retain(|&slot| slot != edge);
        lists[endpoint].push(edge);
        self.edges[edge] = Some(if source_side {
            (endpoint, target)
        } else {
            (source, endpoint)
        });
    }

    pub(crate) fn live_nodes(&self) -> Vec<usize> {
        (0..self.node_live.len())
            .filter(|&node| self.node_live[node])
            .collect()
    }

    pub(crate) fn live_edges(&self) -> Vec<usize> {
        (0..self.edges.len())
            .filter(|&edge| self.edges[edge].is_some())
            .collect()
    }

    pub(crate) fn neighbors(&self, node: usize, forward: bool) -> Vec<usize> {
        let list = if forward {
            &self.outgoing[node]
        } else {
            &self.incoming[node]
        };
        list.iter()
            .map(|&edge| {
                let (source, target) = self.edges[edge].expect("a listed edge is live");
                if forward { target } else { source }
            })
            .collect()
    }

    pub(crate) fn breadth_first(&self, start: usize, forward: bool) -> Vec<usize> {
        let mut seen = vec![false; self.node_live.len()];
        let mut order = Vec::new();
        let mut queue = VecDeque::from([start]);
        seen[start] = true;
        while let Some(node) = queue.pop_front() {
            order.push(node);
            for next in self.neighbors(node, forward) {
                if !seen[next] {
                    seen[next] = true;
                    queue.push_back(next);
                }
            }
        }
        order
    }

    pub(crate) fn preorder(
        &self,
        node: usize,
        forward: bool,
        seen: &mut Vec<bool>,
        order: &mut Vec<usize>,
    ) {
        if seen[node] {
            return;
        }
        seen[node] = true;
        order.push(node);
        for next in self.neighbors(node, forward) {
            self.preorder(next, forward, seen, order);
        }
    }

    pub(crate) fn distances(&self, start: usize) -> Vec<Option<usize>> {
        let mut distance = vec![None; self.node_live.len()];
        distance[start] = Some(0);
        let mut queue = VecDeque::from([start]);
        while let Some(node) = queue.pop_front() {
            let next_distance = distance[node].map(|steps| steps + 1);
            for next in self.neighbors(node, true) {
                if distance[next].is_none() {
                    distance[next] = next_distance;
                    queue.push_back(next);
                }
            }
        }
        distance
    }

    /// The compacted model, with adjacency rebuilt in edge-identity order, and the node and edge
    /// maps. Only an exact check without redirections relies on that order.
    pub(crate) fn compacted(&self) -> (Self, Vec<Option<usize>>, Vec<Option<usize>>) {
        let mut result = Self::new();
        let mut node_map = vec![None; self.node_live.len()];
        for node in self.live_nodes() {
            node_map[node] = Some(result.add_node(self.node_tag[node]));
        }
        let mut edge_map = vec![None; self.edges.len()];
        for (slot, edge) in self.edges.iter().enumerate() {
            if let Some((source, target)) = *edge {
                let source = node_map[source].expect("a live edge has live endpoints");
                let target = node_map[target].expect("a live edge has live endpoints");
                edge_map[slot] = Some(result.add_edge(source, target, self.edge_tag[slot]));
            }
        }
        (result, node_map, edge_map)
    }
}
