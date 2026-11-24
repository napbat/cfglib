//! The worklist behind every fixpoint solver.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use super::bits::{DenseBits, bit_slot};
use crate::graph::view::{DenseId, GraphView};

/// A deduplicating queue of dense node indices, popped lowest first.
///
/// Every solver revisits a node only after one of its inputs changed, and
/// queuing a node that is already queued must not visit it twice. The lowest
/// index pops first, which is program order for a CFG whose blocks were
/// created in order and keeps every solve deterministic. The queue is a bit
/// row, so queuing allocates nothing.
///
/// A [`schedule`]d worklist keeps the bit row but indexes it by each node's
/// position in an iteration order, so the node earliest in that order pops
/// first.
pub(super) struct Worklist {
    queued: DenseBits,
    /// Words below this index hold no queued node.
    cursor: usize,
    ranking: Option<Ranking>,
}

/// An iteration order as two tables: node to position and position to node.
struct Ranking {
    /// Per dense node index: its position, `u32::MAX` outside the order.
    position: Vec<u32>,
    /// Per position: the dense node index there.
    node: Vec<u32>,
}

impl Ranking {
    fn of<N: DenseId>(bound: usize, order: impl ExactSizeIterator<Item = N>) -> Self {
        let mut ranking = Self {
            position: vec![u32::MAX; bound],
            node: Vec::with_capacity(order.len()),
        };
        for (position, node) in order.enumerate() {
            ranking.position[node.index()] = to_u32(position);
            ranking.node.push(to_u32(node.index()));
        }
        ranking
    }
}

fn to_u32(value: usize) -> u32 {
    u32::try_from(value).expect("an iteration order holds fewer than u32::MAX nodes")
}

impl Worklist {
    /// An empty worklist for the indices `0..bound`.
    pub(super) fn new(bound: usize) -> Self {
        Self {
            queued: DenseBits::new(bound),
            cursor: 0,
            ranking: None,
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
    ///
    /// # Panics
    ///
    /// Panics when the iteration order of a scheduled worklist does not cover
    /// `index`.
    pub(super) fn insert(&mut self, index: usize) {
        let slot = self.slot(index);
        self.queue(slot);
    }

    /// Queues `index` unless it is already queued or comes before position
    /// `floor` of the iteration order.
    pub(super) fn insert_from(&mut self, index: usize, floor: usize) {
        let slot = self.slot(index);
        if slot >= floor {
            self.queue(slot);
        }
    }

    /// Where `index` sits in the bit row: its position in the iteration
    /// order of a scheduled worklist, and the index itself otherwise.
    fn slot(&self, index: usize) -> usize {
        match &self.ranking {
            None => index,
            Some(ranking) => {
                let position = ranking.position[index];
                assert!(
                    position != u32::MAX,
                    "the iteration order covers every node a solve queues"
                );
                position as usize
            }
        }
    }

    fn queue(&mut self, slot: usize) {
        self.cursor = self.cursor.min(bit_slot(slot).0);
        self.queued.insert(slot);
    }

    /// Removes and returns the queued node that comes first: the lowest
    /// index, or the earliest position of a scheduled worklist.
    pub(super) fn pop_first(&mut self) -> Option<usize> {
        let words = self.queued.words();
        while let Some(&word) = words.get(self.cursor) {
            if word != 0 {
                let slot = self.cursor * 64 + word.trailing_zeros() as usize;
                self.queued.remove(slot);
                return Some(match &self.ranking {
                    None => slot,
                    Some(ranking) => ranking.node[slot] as usize,
                });
            }
            self.cursor += 1;
        }
        None
    }

    /// Whether `index` is queued, before the worklist is ranked.
    pub(super) fn contains(&self, index: usize) -> bool {
        debug_assert!(self.ranking.is_none(), "only an unranked row holds indices");
        self.queued.get(index)
    }

    /// The queued indices, ascending, before the worklist is ranked.
    pub(super) fn queued(&self) -> impl Iterator<Item = usize> + '_ {
        debug_assert!(self.ranking.is_none(), "only an unranked row holds indices");
        self.queued.ones()
    }

    /// Re-queues every queued node by its position in `order`, and pops in
    /// that order from now on.
    pub(super) fn rank_by<N: DenseId>(
        &mut self,
        bound: usize,
        order: impl ExactSizeIterator<Item = N>,
    ) {
        let ranking = Ranking::of(bound, order);
        let queued = core::mem::replace(&mut self.queued, DenseBits::new(ranking.node.len()));
        self.cursor = 0;
        self.ranking = Some(ranking);
        for index in queued.ones() {
            self.insert(index);
        }
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

    #[test]
    fn a_ranked_worklist_pops_by_position_and_requeues_what_it_held() {
        let mut worklist = Worklist::of(200, [3, 130, 64]);
        // The order 130, 7, 64, 3, 199: positions 0 to 4.
        worklist.rank_by(200, [130_usize, 7, 64, 3, 199].into_iter());
        assert_eq!(worklist.pop_first(), Some(130));
        worklist.insert(199);
        worklist.insert(7);
        worklist.insert(7);
        assert_eq!(drain(worklist), [7, 64, 3, 199]);
    }

    #[test]
    #[should_panic(expected = "the iteration order covers every node a solve queues")]
    fn queuing_a_node_outside_the_order_panics() {
        let mut worklist = Worklist::new(10);
        worklist.rank_by(10, [4_usize, 2].into_iter());
        worklist.insert(5);
    }
}
