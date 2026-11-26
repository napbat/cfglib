//! The node orders a compaction can number the survivors in.
//!
//! A node's identity decides where its payload, its adjacency offsets, and
//! its entry in every node-indexed side table live, so an order that numbers
//! neighbors close together turns a scan of a large graph into reads of
//! nearby memory. Each order here costs one pass over the edges at most;
//! heavyweight orders that model the cache are left to a consumer, which can
//! hand its result in as [`NodeOrder::Explicit`].

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;
use core::ops::ControlFlow;

use super::Graph;
use super::id::{Id, IdTag, NodeTag};
use crate::graph::search::Visit;
use crate::graph::traverse::{Outgoing, breadth_first_core, postorder_in};
use crate::graph::view::Direction;
use crate::identity::NO_INDEX;

/// How [`Graph::compact`] numbers the surviving nodes.
///
/// Every order is deterministic: ties break by adjacency order, then by slot.
/// The searches follow outgoing edges, read each node's edges in adjacency
/// order, start from each root in turn, and skip the nodes an earlier root
/// already reached, so a root listed twice or reached from an earlier one
/// adds nothing. The live nodes a search does not reach follow in slot
/// order.
///
/// # Examples
///
/// ```
/// use cfglib::{Graph, NodeOrder};
///
/// let mut graph = Graph::<&'static str, ()>::new();
/// let exit = graph.add_node("exit");
/// let entry = graph.add_node("entry");
/// let body = graph.add_node("body");
/// graph.add_edge(entry, body, ());
/// graph.add_edge(body, exit, ());
///
/// graph.compact(NodeOrder::BreadthFirst(&[entry]));
/// assert_eq!(graph.nodes().copied().collect::<Vec<_>>(), ["entry", "body", "exit"]);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum NodeOrder<'a, NT: IdTag = NodeTag> {
    /// Slot order: the survivors keep their relative order, so node
    /// identity order still means insertion order.
    ///
    /// Needs no roots, no search and no payload moves, which makes it the
    /// cheapest order and the one to use when nothing reads the graph in
    /// bulk.
    Insertion,
    /// Breadth-first from the roots in turn: each root, then the nodes its
    /// search discovers, in the order it discovers them.
    ///
    /// Nodes at the same distance from a root sit together, which keeps the
    /// numbers a node's successors carry close to each other.
    BreadthFirst(&'a [Id<NT>]),
    /// Reverse postorder of a depth-first search that starts from each root
    /// in turn: the reverse postorder from a virtual root whose successors
    /// are the roots, in order.
    ///
    /// Every edge between two reached nodes that is not a back edge of that
    /// search goes from a lower to a higher identity, so iterating the
    /// identities in order is a forward dataflow solver's iteration order. A
    /// single root is numbered first; with several, the nodes a later root
    /// reached first come before those of an earlier root.
    ReversePostorder(&'a [Id<NT>]),
    /// A caller-supplied sequence naming every live node exactly once: the
    /// node at position `i` gets identity `i`.
    Explicit(&'a [Id<NT>]),
}

/// New numbers, handed out in the order the nodes first ask for one.
struct Numbering {
    map: Vec<u32>,
    next: u32,
}

impl Numbering {
    fn new(slots: usize) -> Self {
        Self {
            map: vec![NO_INDEX; slots],
            next: 0,
        }
    }

    /// Give `node` the next number unless it has one, and report whether it
    /// was numbered now.
    fn assign<T: IdTag>(&mut self, node: Id<T>) -> bool {
        let entry = &mut self.map[node.index()];
        if *entry != NO_INDEX {
            return false;
        }
        *entry = self.next;
        self.next += 1;
        true
    }
}

impl<N, E, NT: IdTag, ET: IdTag> Graph<N, E, NT, ET> {
    /// The new number of every node slot under `order`, with [`NO_INDEX`] for
    /// the removed.
    pub(super) fn ordered_slots(&self, order: NodeOrder<'_, NT>) -> Vec<u32> {
        let mut numbering = Numbering::new(self.node_bound());
        match order {
            // Slot order is what the loop below hands out on its own.
            NodeOrder::Insertion => {}
            NodeOrder::BreadthFirst(roots) => self.number_breadth_first(roots, &mut numbering),
            NodeOrder::ReversePostorder(roots) => {
                self.number_reverse_postorder(roots, &mut numbering);
            }
            NodeOrder::Explicit(nodes) => {
                for &node in nodes {
                    self.assert_live_endpoint(node, "explicitly ordered");
                    assert!(
                        numbering.assign(node),
                        "an explicit order names node {node} twice"
                    );
                }
                assert_eq!(
                    numbering.next as usize, self.live_node_count,
                    "an explicit order must name every live node"
                );
            }
        }
        for node in self.node_ids() {
            numbering.assign(node);
        }
        numbering.map
    }

    fn number_breadth_first(&self, roots: &[Id<NT>], numbering: &mut Numbering) {
        // Numbering a node when it is discovered makes the queue the order, so
        // the search's own marks are the numbering.
        let mut queue = Vec::new();
        for &root in roots {
            self.assert_live_endpoint(root, "root");
            if !numbering.assign(root) {
                continue;
            }
            queue.clear();
            queue.push(root);
            breadth_first_core::<_, _, ()>(
                Outgoing,
                self,
                &mut queue,
                |_, _| ControlFlow::Continue(Visit::Descend),
                |_, _, to, _| numbering.assign(to),
            );
        }
    }

    fn number_reverse_postorder(&self, roots: &[Id<NT>], numbering: &mut Numbering) {
        for &root in roots {
            self.assert_live_endpoint(root, "root");
        }
        let postorder = postorder_in(
            self,
            roots.iter().copied(),
            Direction::Forward,
            &mut Vec::new(),
        );
        for &node in postorder.iter().rev() {
            numbering.assign(node);
        }
    }
}
