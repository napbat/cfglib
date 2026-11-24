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

    /// Whether `index` is queued.
    pub(super) fn contains(&self, index: usize) -> bool {
        self.queued.get(index)
    }

    /// The queued indices, ascending.
    pub(super) fn queued(&self) -> impl Iterator<Item = usize> + '_ {
        self.queued.ones()
    }
}

/// The queue a solver core pops from and pushes to: a [`Worklist`] in index
/// order, or a [`Ranked`] one in an iteration order.
///
/// A core is generic over its queue, so a solve in index order runs on the
/// plain worklist without a position lookup on each push and pop.
pub(super) trait Queue {
    /// Queues the node at `index` unless it is already queued.
    fn push(&mut self, index: usize);

    /// Queues the node at `index` unless it is already queued or comes before
    /// position `floor` of the iteration order.
    fn push_from(&mut self, index: usize, floor: usize);

    /// Removes and returns the queued node that comes first.
    fn pop(&mut self) -> Option<usize>;
}

/// The forwarders are inline, so a core that the caller's crate instantiates
/// calls the worklist itself rather than a wrapper around it.
impl Queue for Worklist {
    #[inline]
    fn push(&mut self, index: usize) {
        self.insert(index);
    }

    #[inline]
    fn push_from(&mut self, index: usize, floor: usize) {
        if index >= floor {
            self.insert(index);
        }
    }

    #[inline]
    fn pop(&mut self) -> Option<usize> {
        self.pop_first()
    }
}

/// A worklist that pops the node earliest in an iteration order: the bit row
/// of a [`Worklist`], indexed by position instead of by node.
pub(super) struct Ranked {
    slots: Worklist,
    /// Per dense node index: its position, `u32::MAX` outside the order.
    position: Vec<u32>,
    /// Per position: the dense node index there.
    node: Vec<u32>,
}

fn to_u32(value: usize) -> u32 {
    u32::try_from(value).expect("an iteration order holds fewer than u32::MAX nodes")
}

impl Ranked {
    /// Queues what `worklist` holds by each node's position in `order`, a
    /// sequence of distinct nodes below `bound`.
    pub(super) fn new<N: DenseId>(
        worklist: &Worklist,
        bound: usize,
        order: impl ExactSizeIterator<Item = N>,
    ) -> Self {
        let mut ranked = Self {
            slots: Worklist::new(order.len()),
            position: vec![u32::MAX; bound],
            node: Vec::with_capacity(order.len()),
        };
        for (position, node) in order.enumerate() {
            ranked.position[node.index()] = to_u32(position);
            ranked.node.push(to_u32(node.index()));
        }
        for index in worklist.queued() {
            ranked.push(index);
        }
        ranked
    }

    /// The position of the node at `index`.
    ///
    /// # Panics
    ///
    /// Panics when the order does not cover `index`.
    fn position(&self, index: usize) -> usize {
        let position = self.position[index];
        assert!(
            position != u32::MAX,
            "the iteration order covers every node a solve queues"
        );
        position as usize
    }
}

impl Queue for Ranked {
    fn push(&mut self, index: usize) {
        let position = self.position(index);
        self.slots.insert(position);
    }

    fn push_from(&mut self, index: usize, floor: usize) {
        let position = self.position(index);
        if position >= floor {
            self.slots.insert(position);
        }
    }

    fn pop(&mut self) -> Option<usize> {
        let position = self.slots.pop_first()?;
        Some(self.node[position] as usize)
    }
}

#[cfg(test)]
mod tests {
    extern crate alloc;

    use alloc::vec::Vec;

    use super::{Queue, Ranked, Worklist};
    use crate::graph::store::{Graph, NodeId};

    fn drain(mut queue: impl Queue) -> Vec<usize> {
        let mut order = Vec::new();
        while let Some(index) = queue.pop() {
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
        let worklist = Worklist::of(200, [3, 130, 64]);
        // The order 130, 7, 64, 3, 199: positions 0 to 4.
        let mut ranked = Ranked::new(&worklist, 200, [130_usize, 7, 64, 3, 199].into_iter());
        assert_eq!(ranked.pop(), Some(130));
        ranked.push(199);
        ranked.push(7);
        ranked.push(7);
        // Position 1 is before the floor 2, so 7 stays where it is queued.
        ranked.push_from(130, 2);
        assert_eq!(drain(ranked), [7, 64, 3, 199]);
    }

    #[test]
    fn an_index_worklist_pushes_from_a_floor_of_indices() {
        let mut worklist = Worklist::new(10);
        worklist.push_from(3, 4);
        worklist.push_from(6, 4);
        assert_eq!(drain(worklist), [6]);
    }

    #[test]
    #[should_panic(expected = "the iteration order covers every node a solve queues")]
    fn queuing_a_node_outside_the_order_panics() {
        let mut ranked = Ranked::new(&Worklist::new(10), 10, [4_usize, 2].into_iter());
        ranked.push(5);
    }
}
