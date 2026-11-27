//! Tarjan's algorithm, in Pearce's single-array form: the one walk behind
//! the leaves-first decomposition and the [`IncrementalScc`](super::IncrementalScc)
//! repairs.
//!
//! The walk keeps its own loop instead of running the shared depth-first
//! core. The core keeps the marks and the visitor apart, so Pearce's array
//! had to be shared between them as cells, and every open node took a second
//! stack entry for the index it was entered at. On 4,096 nodes the core form
//! measured 1.13x the time of the previous release's walk
//! (`directed_scc_leaves_first`), and this loop 0.98x.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use super::SccDecomposition;
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
    /// Every open node's successors, appended as its frame is pushed and
    /// truncated away as it pops.
    arena: Vec<N>,
    /// The open nodes, innermost last.
    frames: Vec<Frame<N>>,
    /// The components found, each an ascending run, leaves first.
    found: Vec<N>,
    /// `offsets[i]..offsets[i + 1]` is the `i`th component within `found`.
    offsets: Vec<u32>,
}

/// An open node of the walk.
#[derive(Debug, Clone, Copy)]
struct Frame<N> {
    node: N,
    /// Where the node's successors start in the arena; the pop truncates to
    /// it.
    start: usize,
    /// The next successor to examine, as an arena index. The top frame's
    /// successors run to the arena's end.
    cursor: usize,
    /// Whether no successor has yet shown the node a lower index, so it is
    /// still the root of its component.
    root: bool,
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
            arena: Vec::new(),
            frames: Vec::new(),
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
    /// visited, or one `within` refuses, is passed over.
    pub(super) fn run<G>(
        &mut self,
        graph: &G,
        roots: impl IntoIterator<Item = N>,
        within: impl Fn(N) -> bool,
    ) where
        G: GraphView<NodeId = N>,
    {
        let Self {
            rindex,
            waiting,
            arena,
            frames,
            found,
            offsets,
        } = self;
        let mut next_index = 1;
        let mut next_component = word(rindex.len());
        for root in roots {
            if rindex[root.index()] != 0 || !within(root) {
                continue;
            }
            rindex[root.index()] = next_index;
            next_index += 1;
            enter(graph, root, arena, frames);
            while let Some(frame) = frames.last_mut() {
                let node = frame.node;
                if let Some(&successor) = arena.get(frame.cursor) {
                    frame.cursor += 1;
                    // A node seen before lowers this node's word to its own
                    // when that is lower: a node still waiting, which this one
                    // therefore shares a component with. A finished node's
                    // word is above every waiting one.
                    let seen = rindex[successor.index()];
                    if seen == 0 {
                        if within(successor) {
                            rindex[successor.index()] = next_index;
                            next_index += 1;
                            enter(graph, successor, arena, frames);
                        }
                    } else if seen < rindex[node.index()] {
                        rindex[node.index()] = seen;
                        frame.root = false;
                    }
                    continue;
                }

                let Some(finished) = frames.pop() else { break };
                arena.truncate(finished.start);
                if finished.root {
                    // The root has the component's lowest index. Its members
                    // are the waiting nodes that hold an index no lower; they
                    // come off in reverse finish order, which along a path is
                    // visit order, so a cycle entered at its lowest id is
                    // ascending already.
                    let first = found.len();
                    let root_index = rindex[node.index()];
                    rindex[node.index()] = next_component;
                    found.push(node);
                    while let Some(&member) = waiting.last() {
                        if rindex[member.index()] < root_index {
                            break;
                        }
                        waiting.pop();
                        rindex[member.index()] = next_component;
                        found.push(member);
                    }
                    next_index -= word(found.len() - first);
                    next_component -= 1;
                    found[first..].sort_unstable();
                    offsets.push(word(found.len()));
                } else {
                    waiting.push(node);
                }
                // The parent shares a component with whatever this node's
                // subtree reached that is still waiting.
                if let Some(parent) = frames.last_mut()
                    && rindex[node.index()] < rindex[parent.node.index()]
                {
                    rindex[parent.node.index()] = rindex[node.index()];
                    parent.root = false;
                }
            }
        }
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

/// Open `node`: push its frame, with its successors on the arena's tail.
#[inline]
fn enter<G: GraphView>(
    graph: &G,
    node: G::NodeId,
    arena: &mut Vec<G::NodeId>,
    frames: &mut Vec<Frame<G::NodeId>>,
) {
    let start = arena.len();
    arena.extend(graph.successors(node));
    frames.push(Frame {
        node,
        start,
        cursor: start,
        root: true,
    });
}
