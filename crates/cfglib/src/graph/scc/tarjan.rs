//! Tarjan's algorithm, in Pearce's single-array form: the one walk behind
//! the leaves-first decomposition and the [`IncrementalScc`](super::IncrementalScc)
//! repairs.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;
use core::cell::Cell;
use core::ops::ControlFlow;

use super::SccDecomposition;
use crate::graph::search::Visit;
use crate::graph::traverse::{
    DepthFirstMarks, DepthFirstStack, DepthFirstVisitor, Mark, Outgoing, StepKind, depth_first_core,
};
use crate::graph::view::{DenseId, GraphView};
use crate::identity::{NO_INDEX, check_bound, slot, word};

/// Decompose `graph` with one Tarjan walk over every node, numbering the
/// components in the order the walk completes them: leaves first.
pub(super) fn leaves_first<G: GraphView>(graph: &G) -> SccDecomposition<G::NodeId> {
    let node_count = graph.node_bound();
    let mut walk = Tarjan::with_node_bound(node_count);
    walk.found.reserve(node_count);
    walk.run(graph, graph.node_ids(), |_| true);
    let Tarjan {
        mut rindex,
        found,
        offsets,
        ..
    } = walk;
    // Component numbers count down from the node count in completion order;
    // the decomposition numbers them up from 0 in the same order. A slot the
    // walk never reached holds no node.
    let top = word(node_count);
    for entry in &mut rindex {
        *entry = if *entry == 0 { NO_INDEX } else { top - *entry };
    }
    SccDecomposition {
        nodes: found,
        offsets,
        component_of: rindex,
    }
}

/// Tarjan's walk from chosen roots, with its buffers kept across walks.
///
/// Pearce's single array ("A space-efficient algorithm for finding strongly
/// connected components", IPL 2016) stands in for Tarjan's index, low-link
/// and on-stack arrays. Each slot holds one word:
/// - 0 is a node not yet visited;
/// - `1..next_index` is a node still open or waiting for its component,
///   holding the lowest index it has seen;
/// - above `next_component` is a finished node's component, counted down
///   from the node bound.
///
/// The waiting nodes hold exactly the indices `1..next_index`, because a
/// component takes the most recently visited of them and gives its indices
/// back, so every waiting index stays below every component number and one
/// `<` tells a waiting node from a finished one. A walk runs every root to
/// completion, so afterwards every node it visited is finished.
#[derive(Debug, Clone)]
pub(super) struct Tarjan<N> {
    /// Pearce's word per slot. Every word is 0 before a walk;
    /// [`clear`](Self::clear) resets only the nodes the last walk visited.
    rindex: Vec<u32>,
    /// The finished nodes that were not the root of their component, waiting
    /// for that root to finish.
    waiting: Vec<N>,
    /// The open nodes, each with the index it was entered at: it is the root
    /// of its component exactly when it finishes still holding that index.
    path: Vec<(N, u32)>,
    /// The frames and pending successors of the depth-first core.
    stack: DepthFirstStack<N, N>,
    /// The components found, each an ascending run, leaves first.
    found: Vec<N>,
    /// `offsets[i]..offsets[i + 1]` is the `i`th component within `found`.
    offsets: Vec<u32>,
}

impl<N: DenseId> Tarjan<N> {
    /// A walk over the slots `0..node_bound`, which a graph whose node bound
    /// is larger must not lead it out of.
    ///
    /// # Panics
    ///
    /// Panics when `node_bound` exceeds the range of a dense `u32` table.
    pub(super) fn with_node_bound(node_bound: usize) -> Self {
        check_bound(
            node_bound,
            "node bound of a strongly connected component walk",
        );
        Self {
            rindex: vec![0; node_bound],
            waiting: Vec::new(),
            path: Vec::new(),
            stack: DepthFirstStack::default(),
            found: Vec::new(),
            offsets: vec![0],
        }
    }

    /// Find the components of the nodes `roots` reach through nodes `within`
    /// admits, in completion order, leaves first.
    ///
    /// Every node outside `within` is left out of the graph. A component that
    /// a root reaches is found whole, since all its nodes are reachable from
    /// any one of them, as long as `within` admits them all. A root already
    /// visited is passed over; every root must be admitted by `within`.
    pub(super) fn run<G>(
        &mut self,
        graph: &G,
        roots: impl IntoIterator<Item = N>,
        within: impl Fn(N) -> bool,
    ) where
        G: GraphView<NodeId = N>,
    {
        // The core keeps a frame per open node, and the path one word pair
        // more, so a walk as deep as the graph grows neither by doubling.
        self.path.reserve(graph.node_bound());
        // The core's marks and the visitor both read and write Pearce's
        // array, so they share it as cells.
        let rindex = Cell::from_mut(self.rindex.as_mut_slice()).as_slice_of_cells();
        let mut marks = Unvisited {
            rindex,
            within: |index| within(N::from_index(index)),
        };
        let mut visitor = Visitor {
            rindex,
            waiting: &mut self.waiting,
            path: &mut self.path,
            found: &mut self.found,
            offsets: &mut self.offsets,
            next_index: 1,
            next_component: word(rindex.len()),
        };
        depth_first_core::<_, _, _, ()>(
            Outgoing,
            graph,
            roots,
            &mut marks,
            &mut self.stack,
            &mut visitor,
        );
    }

    /// Whether the last walk put `node` into a component.
    pub(super) fn reached(&self, node: N) -> bool {
        self.rindex[node.index()] != 0
    }

    /// The components the last walk found, leaves first.
    pub(super) fn components(&self) -> impl DoubleEndedIterator<Item = &[N]> + '_ {
        self.offsets
            .windows(2)
            .map(|run| &self.found[slot(run[0])..slot(run[1])])
    }

    /// How many components the last walk found.
    pub(super) fn component_count(&self) -> usize {
        self.offsets.len() - 1
    }

    /// Forget the last walk, leaving every word of the array 0 again.
    pub(super) fn clear(&mut self) {
        for &node in &self.found {
            self.rindex[node.index()] = 0;
        }
        self.found.clear();
        self.offsets.truncate(1);
    }
}

/// The core's marks, read off Pearce's array: a slot is unseen while its
/// word is 0 and `within` admits it, and done otherwise. A step to a node
/// `within` refuses is therefore a step to a done node whose word is 0, which
/// the visitor passes over.
struct Unvisited<'a, F> {
    rindex: &'a [Cell<u32>],
    within: F,
}

impl<F: Fn(usize) -> bool> DepthFirstMarks for Unvisited<'_, F> {
    #[inline]
    fn mark(&self, index: usize) -> Mark {
        if self.rindex[index].get() == 0 && (self.within)(index) {
            Mark::Unseen
        } else {
            Mark::Done
        }
    }

    /// The visitor's discovery writes the node's index right after.
    #[inline]
    fn open(&mut self, _index: usize) {}

    #[inline]
    fn close(&mut self, _index: usize) {}
}

/// Pearce's bookkeeping on the core's callbacks.
struct Visitor<'a, N> {
    rindex: &'a [Cell<u32>],
    waiting: &'a mut Vec<N>,
    path: &'a mut Vec<(N, u32)>,
    found: &'a mut Vec<N>,
    offsets: &'a mut Vec<u32>,
    next_index: u32,
    next_component: u32,
}

impl<N: DenseId> DepthFirstVisitor<N, N, ()> for Visitor<'_, N> {
    #[inline]
    fn discover(&mut self, node: N, _depth: usize) -> ControlFlow<(), Visit> {
        self.rindex[node.index()].set(self.next_index);
        self.path.push((node, self.next_index));
        self.next_index += 1;
        ControlFlow::Continue(Visit::Descend)
    }

    /// A step to a node seen before lowers this node's word to that node's,
    /// when it is lower: a node still waiting, which this one therefore
    /// shares a component with. A finished node's word is above every
    /// waiting one, and a refused node's is 0.
    #[inline]
    fn step(&mut self, from: N, _step: N, to: N, kind: StepKind) -> ControlFlow<()> {
        if kind != StepKind::Tree {
            let seen = self.rindex[to.index()].get();
            let own = &self.rindex[from.index()];
            if seen != 0 && seen < own.get() {
                own.set(seen);
            }
        }
        ControlFlow::Continue(())
    }

    #[inline]
    fn finish(&mut self, node: N) -> ControlFlow<()> {
        let (_, entered) = self
            .path
            .pop()
            .expect("the core finishes only a node it entered");
        let cell = &self.rindex[node.index()];
        if cell.get() == entered {
            // The root has the component's lowest index. Its members are the
            // waiting nodes that hold an index no lower; they come off in
            // reverse finish order, which along a path is visit order, so a
            // cycle entered at its lowest id is ascending already.
            let first = self.found.len();
            cell.set(self.next_component);
            self.found.push(node);
            while let Some(&member) = self.waiting.last() {
                let member_word = &self.rindex[member.index()];
                if member_word.get() < entered {
                    break;
                }
                self.waiting.pop();
                member_word.set(self.next_component);
                self.found.push(member);
            }
            self.next_index -= word(self.found.len() - first);
            self.next_component -= 1;
            self.found[first..].sort_unstable();
            self.offsets.push(word(self.found.len()));
        } else {
            self.waiting.push(node);
        }
        // The parent shares a component with whatever this node's subtree
        // reached that is still waiting.
        if let Some(&(parent, _)) = self.path.last() {
            let low = cell.get();
            let theirs = &self.rindex[parent.index()];
            if low < theirs.get() {
                theirs.set(low);
            }
        }
        ControlFlow::Continue(())
    }
}
