//! Immediate dominators by the simple Lengauer-Tarjan algorithm.
//!
//! The algorithm numbers the nodes reachable from the root in depth-first
//! preorder and finds each node's semidominator as a minimum over a link-eval
//! forest. The forest uses path compression and no balancing, which bounds
//! the whole computation by O(m log n) for m edges and n nodes. A second
//! forward pass turns the semidominators into immediate dominators.
//!
//! Nothing here recurses. The search keeps its pending nodes on a stack and
//! path compression keeps its path on another, so the depth of the graph is
//! limited by memory and not by the call stack. Unreachable predecessors
//! have no depth-first number and are skipped.
//!
//! Every table below is indexed by depth-first number, except `number`, which
//! maps a dense node index to its depth-first number.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use super::{ABSENT, DominatorTree, slot, word};
use crate::graph::view::{DenseId, GraphView};

/// A node the depth-first search still has to number, and the depth-first
/// number of the node that reached it.
#[derive(Debug, Clone, Copy)]
struct Pending {
    node: u32,
    parent: u32,
}

/// Every table the Lengauer-Tarjan computation fills, owned by the caller.
#[derive(Debug, Clone, Default)]
pub(super) struct ConstructionBuffers {
    /// Depth-first number of each node, [`ABSENT`] when the root misses it.
    number: Vec<u32>,
    /// Node index of each depth-first number.
    vertex: Vec<u32>,
    /// Depth-first number of each node's parent in the search tree.
    parent: Vec<u32>,
    /// Semidominator of each node, as a depth-first number.
    semi: Vec<u32>,
    /// The node of least semidominator on the compressed path above a node.
    label: Vec<u32>,
    /// The link-eval forest, [`ABSENT`] for a node not yet linked.
    ancestor: Vec<u32>,
    /// Relative dominator of each node, then its immediate dominator.
    dominator: Vec<u32>,
    /// First node of the bucket of each node.
    bucket_head: Vec<u32>,
    /// The node after each node in its bucket.
    bucket_next: Vec<u32>,
    /// Nodes the search has yet to number.
    pending: Vec<Pending>,
    /// The path path compression is flattening.
    path: Vec<u32>,
}

impl ConstructionBuffers {
    /// The number of nodes the last call to
    /// [`immediate_dominators`](Self::immediate_dominators) reached.
    pub(super) fn reached(&self) -> usize {
        self.vertex.len()
    }

    /// The parent table of the dominator tree of `graph` from `root`, indexed
    /// by dense node index, with [`ABSENT`] for the root and for every node
    /// the root does not reach.
    pub(super) fn immediate_dominators<G: GraphView>(
        &mut self,
        graph: &G,
        root: usize,
    ) -> Vec<u32> {
        self.search(graph, root);
        let reached = self.vertex.len();
        self.prepare(reached);
        for node in (1..reached).rev() {
            self.process(graph, node);
        }

        let mut idom = vec![ABSENT; graph.node_bound()];
        for node in 1..reached {
            if self.dominator[node] != self.semi[node] {
                self.dominator[node] = self.dominator[slot(self.dominator[node])];
            }
            idom[slot(self.vertex[node])] = self.vertex[slot(self.dominator[node])];
        }
        idom
    }

    /// Number the nodes reachable from `root` in depth-first preorder.
    ///
    /// A node is numbered when it is popped, and its successors are pushed
    /// above everything still pending, so the numbering is the one a recursive
    /// search would produce.
    fn search<G: GraphView>(&mut self, graph: &G, root: usize) {
        self.number.clear();
        self.number.resize(graph.node_bound(), ABSENT);
        self.vertex.clear();
        self.parent.clear();
        self.pending.clear();
        self.pending.push(Pending {
            node: word(root),
            parent: ABSENT,
        });
        while let Some(Pending { node, parent }) = self.pending.pop() {
            if self.number[slot(node)] != ABSENT {
                continue;
            }
            let number = word(self.vertex.len());
            self.number[slot(node)] = number;
            self.vertex.push(node);
            self.parent.push(parent);
            for successor in graph.successors(G::NodeId::from_index(slot(node))) {
                if self.number[successor.index()] == ABSENT {
                    self.pending.push(Pending {
                        node: word(successor.index()),
                        parent: number,
                    });
                }
            }
        }
    }

    /// Reset the link-eval tables for `reached` nodes.
    ///
    /// A node whose semidominator is its search parent is already settled, so
    /// the relative dominator table starts as the parent table and such a node
    /// never enters a bucket.
    fn prepare(&mut self, reached: usize) {
        let count = word(reached);
        self.semi.clear();
        self.semi.extend(0..count);
        self.label.clear();
        self.label.extend(0..count);
        self.ancestor.clear();
        self.ancestor.resize(reached, ABSENT);
        self.dominator.clear();
        self.dominator.extend_from_slice(&self.parent);
        self.dominator[0] = 0;
        self.bucket_head.clear();
        self.bucket_head.resize(reached, ABSENT);
        self.bucket_next.clear();
        self.bucket_next.resize(reached, ABSENT);
    }

    /// Find the semidominator of `node`, link it below its search parent, and
    /// settle every node whose semidominator is that parent.
    fn process<G: GraphView>(&mut self, graph: &G, node: usize) {
        let mut semi = self.semi[node];
        for predecessor in graph.predecessors(G::NodeId::from_index(slot(self.vertex[node]))) {
            let number = self.number[predecessor.index()];
            if number == ABSENT {
                continue;
            }
            let least = self.evaluate(number);
            semi = semi.min(self.semi[slot(least)]);
        }
        self.semi[node] = semi;

        let parent = self.parent[node];
        if semi != parent {
            self.bucket_next[node] = self.bucket_head[slot(semi)];
            self.bucket_head[slot(semi)] = word(node);
        }
        self.ancestor[node] = parent;

        let mut waiting = self.bucket_head[slot(parent)];
        self.bucket_head[slot(parent)] = ABSENT;
        while waiting != ABSENT {
            let least = self.evaluate(waiting);
            self.dominator[slot(waiting)] = if self.semi[slot(least)] < self.semi[slot(waiting)] {
                least
            } else {
                parent
            };
            waiting = self.bucket_next[slot(waiting)];
        }
    }

    /// The node of least semidominator on the path from `node` up to, but not
    /// including, the root of its tree in the link-eval forest.
    fn evaluate(&mut self, node: u32) -> u32 {
        if self.ancestor[slot(node)] == ABSENT {
            return node;
        }
        self.compress(node);
        self.label[slot(node)]
    }

    /// Point every node on the path above `start` at the node below the root,
    /// carrying the least label down the path.
    ///
    /// The recursive form compresses the ancestor first and then the node, so
    /// the nodes are collected on the way up and fixed from the top down.
    fn compress(&mut self, start: u32) {
        self.path.clear();
        let mut node = start;
        loop {
            let ancestor = self.ancestor[slot(node)];
            if self.ancestor[slot(ancestor)] == ABSENT {
                break;
            }
            self.path.push(node);
            node = ancestor;
        }
        while let Some(node) = self.path.pop() {
            let ancestor = self.ancestor[slot(node)];
            let above = self.label[slot(ancestor)];
            if self.semi[slot(above)] < self.semi[slot(self.label[slot(node)])] {
                self.label[slot(node)] = above;
            }
            self.ancestor[slot(node)] = self.ancestor[slot(ancestor)];
        }
    }
}

impl<N: DenseId> DominatorTree<N> {
    /// Whether the parent property holds on `graph`: every edge out of a
    /// reachable node ends at a reachable node whose immediate dominator is an
    /// ancestor of the edge's tail, or at the root.
    ///
    /// With the sibling property it characterizes the dominator tree. The check
    /// costs one interval test per edge and backs the debug assertion that
    /// follows every construction.
    pub(super) fn satisfies_parent_property<G: GraphView<NodeId = N>>(&self, graph: &G) -> bool {
        self.order.iter().all(|&tail| {
            graph.successors(tail).all(|head| {
                self.is_reachable(head)
                    && self
                        .idom(head)
                        .is_none_or(|parent| self.dominates(parent, tail))
            })
        })
    }
}
