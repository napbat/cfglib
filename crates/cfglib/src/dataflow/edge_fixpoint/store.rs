//! Where the edge solver core keeps its facts.

extern crate alloc;
use alloc::vec::Vec;

use crate::graph::weak_topological_order::WeakTopologicalOrder;

/// Where an edge solve keeps its facts: each node's in program order, and
/// each live edge's.
pub(in crate::dataflow) trait EdgeStore<N, F> {
    /// Whether the solve calls [`finish`](Self::finish) as it leaves each
    /// outermost element of its weak topological order.
    const FINISHES: bool;

    /// The fact of a node with no boundary fact and no upstream edge.
    fn bottom(&self) -> &F;

    /// The physical input fact of the node at `index`.
    fn input(&self, index: usize) -> &F;

    /// The physical output fact of the node at `index`.
    fn output(&self, index: usize) -> &F;

    fn set_node(&mut self, index: usize, input: F, output: F);

    /// The fact on the live edge at `index`.
    fn edge(&self, index: usize) -> &F;

    /// Puts `fact` on the live edge at `index` unless it holds it already,
    /// and returns whether the edge's fact changed.
    fn update_edge(&mut self, index: usize, fact: F) -> bool;

    /// Every node before position `end` of `order` holds its final facts,
    /// and so does every edge into one of them.
    fn finish(&mut self, order: &WeakTopologicalOrder<N>, end: usize);
}

/// Every node's and edge's facts, for a solve that returns them all.
pub(in crate::dataflow) struct Dense<F> {
    pub(in crate::dataflow) bottom: F,
    pub(in crate::dataflow) input: Vec<F>,
    pub(in crate::dataflow) output: Vec<F>,
    pub(in crate::dataflow) edge: Vec<Option<F>>,
}

impl<N, F: PartialEq> EdgeStore<N, F> for Dense<F> {
    const FINISHES: bool = false;

    fn bottom(&self) -> &F {
        &self.bottom
    }

    fn input(&self, index: usize) -> &F {
        &self.input[index]
    }

    fn output(&self, index: usize) -> &F {
        &self.output[index]
    }

    fn set_node(&mut self, index: usize, input: F, output: F) {
        self.input[index] = input;
        self.output[index] = output;
    }

    fn edge(&self, index: usize) -> &F {
        self.edge[index]
            .as_ref()
            .expect("adjacency contains an edge excluded from the view")
    }

    fn update_edge(&mut self, index: usize, fact: F) -> bool {
        let held = self.edge[index]
            .as_mut()
            .expect("adjacency contains an edge excluded from the view");
        let changed = fact != *held;
        if changed {
            *held = fact;
        }
        changed
    }

    fn finish(&mut self, _order: &WeakTopologicalOrder<N>, _end: usize) {}
}
