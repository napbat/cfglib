//! The worklist behind every fixpoint solver.

use super::bits::{DenseBits, bit_slot};
use crate::graph::view::{DenseId, GraphView};

/// A deduplicating queue of dense node indices, popped lowest first.
///
/// Every solver revisits a node only after one of its inputs changed, and
/// queuing a node that is already queued must not visit it twice. The lowest
/// index pops first, which is program order for a CFG whose blocks were
/// created in order and keeps every solve deterministic. The queue is a bit
/// row, so queuing allocates nothing.
pub(super) struct Worklist {
    queued: DenseBits,
    /// Words below this index hold no queued node.
    cursor: usize,
}

impl Worklist {
    /// An empty worklist for the indices `0..bound`.
    pub(super) fn new(bound: usize) -> Self {
        Self {
            queued: DenseBits::new(bound),
            cursor: 0,
        }
    }

    /// A worklist of `indices`, duplicates queued once.
    pub(super) fn of(bound: usize, indices: impl IntoIterator<Item = usize>) -> Self {
        let mut worklist = Self::new(bound);
        for index in indices {
            worklist.insert(index);
        }
        worklist
    }

    /// Every live node of `graph`. A removed node's slot is never queued.
    pub(super) fn live<G: GraphView>(graph: &G) -> Self {
        Self::of(graph.node_bound(), graph.node_ids().map(DenseId::index))
    }

    /// The `seeds`, each of which must be a live node of `graph`.
    ///
    /// # Panics
    ///
    /// Panics when a seed is out of range or names a removed node.
    pub(super) fn seeded<G: GraphView>(graph: &G, seeds: &[G::NodeId]) -> Self {
        let bound = graph.node_bound();
        if seeds.is_empty() {
            return Self::new(bound);
        }
        let live = Self::live(graph);
        Self::of(
            bound,
            seeds.iter().map(|seed| {
                let index = seed.index();
                assert!(
                    index < bound && live.queued.get(index),
                    "seed node is out of range"
                );
                index
            }),
        )
    }

    /// Queues `index` unless it is already queued.
    pub(super) fn insert(&mut self, index: usize) {
        self.cursor = self.cursor.min(bit_slot(index).0);
        self.queued.insert(index);
    }

    /// Removes and returns the lowest queued index.
    pub(super) fn pop_first(&mut self) -> Option<usize> {
        let words = self.queued.words();
        while let Some(&word) = words.get(self.cursor) {
            if word != 0 {
                let index = self.cursor * 64 + word.trailing_zeros() as usize;
                self.queued.remove(index);
                return Some(index);
            }
            self.cursor += 1;
        }
        None
    }
}

#[cfg(test)]
mod tests {
    extern crate alloc;

    use alloc::vec::Vec;

    use super::Worklist;
    use crate::graph::store::{Graph, NodeId};

    fn drain(mut worklist: Worklist) -> Vec<usize> {
        let mut order = Vec::new();
        while let Some(index) = worklist.pop_first() {
            order.push(index);
        }
        order
    }

    #[test]
    fn indices_pop_lowest_first_and_queue_once() {
        let worklist = Worklist::of(200, [130, 3, 64, 3, 63, 130]);
        assert_eq!(drain(worklist), [3, 63, 64, 130]);
    }

    #[test]
    fn an_insert_below_the_cursor_pops_next() {
        let mut worklist = Worklist::of(200, [5, 150]);
        assert_eq!(worklist.pop_first(), Some(5));
        assert_eq!(worklist.pop_first(), Some(150));
        worklist.insert(70);
        worklist.insert(2);
        assert_eq!(drain(worklist), [2, 70]);
    }

    #[test]
    fn a_popped_index_can_be_queued_again() {
        let mut worklist = Worklist::of(10, [4]);
        assert_eq!(worklist.pop_first(), Some(4));
        worklist.insert(4);
        assert_eq!(worklist.pop_first(), Some(4));
        assert_eq!(worklist.pop_first(), None);
    }

    #[test]
    fn a_live_worklist_skips_removed_nodes() {
        let mut graph: Graph<(), ()> = Graph::new();
        let nodes: Vec<NodeId> = (0..4).map(|_| graph.add_node(())).collect();
        assert!(graph.remove_node(nodes[1]));
        assert_eq!(drain(Worklist::live(&graph)), [0, 2, 3]);
    }

    #[test]
    fn seeds_queue_once_and_an_empty_seed_set_queues_nothing() {
        let mut graph: Graph<(), ()> = Graph::new();
        let nodes: Vec<NodeId> = (0..3).map(|_| graph.add_node(())).collect();
        let seeds = [nodes[2], nodes[0], nodes[2]];
        assert_eq!(drain(Worklist::seeded(&graph, &seeds)), [0, 2]);
        assert_eq!(drain(Worklist::seeded(&graph, &[])), Vec::<usize>::new());
    }

    #[test]
    #[should_panic(expected = "seed node is out of range")]
    fn a_seed_beyond_the_bound_panics() {
        let graph: Graph<(), ()> = Graph::new();
        let _ = Worklist::seeded(&graph, &[NodeId::from_index(0)]);
    }

    #[test]
    #[should_panic(expected = "seed node is out of range")]
    fn a_removed_seed_panics() {
        let mut graph: Graph<(), ()> = Graph::new();
        let node = graph.add_node(());
        assert!(graph.remove_node(node));
        let _ = Worklist::seeded(&graph, &[node]);
    }
}
