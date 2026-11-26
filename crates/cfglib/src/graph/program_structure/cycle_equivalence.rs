//! Cycle equivalence by one undirected depth-first search and bracket lists.
//!
//! Two edges of a strongly connected graph are cycle equivalent when every
//! cycle through one passes through the other. Johnson, Pearson and Pingali
//! show that this does not change when edge directions are dropped, and that
//! in a depth-first spanning tree of the undirected multigraph two tree edges
//! are equivalent exactly when the same backedges bracket them, while a tree
//! edge and a backedge are equivalent when that backedge is the tree edge's
//! only bracket. Two backedges never are.
//!
//! Walking the nodes in reverse preorder, each node gathers the bracket lists
//! of its children, drops the brackets that end at it, and adds the ones that
//! leave it. A bracket list is named by its top bracket and its size, which
//! identifies the set once a capping bracket is added wherever two children's
//! lists merge and the second-highest of them reaches above the node and above
//! its own backedges. The names cost O(1) per edge, so the whole computation is
//! O(n + m).

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

use super::SeseEdge;
use super::closed::{ClosedGraph, edge_slot};
use crate::block::BlockId;
use crate::edge::EdgeId;
use crate::graph::edge_view::EdgeView;
use crate::graph::view::{DenseId, RootedView};
use crate::identity::{NO_INDEX, slot, word};

/// The cycle-equivalence classes of the edges of a rooted graph.
///
/// Two edges are cycle equivalent when every cycle through one passes
/// through the other. The cycles are those of the closed graph the
/// [`ProgramStructureTree`](super::ProgramStructureTree) is built on: the
/// blocks the root reaches, with a virtual edge from every exit block and
/// every cut root to a virtual exit, and a virtual edge from the virtual exit
/// back into the root. Every edge of that graph lies on a cycle, so two
/// equivalent edges are exactly the two boundaries of a single-entry
/// single-exit region, in one order or the other.
///
/// Parallel edges stay distinct, so two of them are never equivalent, and a
/// self-loop is a class of its own. An edge whose source the root does not
/// reach has no class.
///
/// The computation is one undirected depth-first search on an explicit stack
/// and one pass in reverse preorder, O(n + m) in time and space for n blocks
/// and m edges, on irreducible graphs too. Class numbers depend on adjacency
/// order; which edges share a class does not.
///
/// # Examples
///
/// ```
/// use cfglib::{Cfg, CycleEquivalence, EdgeKind, SeseEdge};
///
/// // entry -> header; header -> body -> header; header -> exit.
/// let mut cfg = Cfg::<u32>::new();
/// let header = cfg.new_block();
/// let body = cfg.new_block();
/// let exit = cfg.new_block();
/// let enter = cfg.add_edge(cfg.entry(), header, EdgeKind::Fallthrough);
/// let stay = cfg.add_edge(header, body, EdgeKind::ConditionalTrue);
/// let back = cfg.add_edge(body, header, EdgeKind::Back);
/// let leave = cfg.add_edge(header, exit, EdgeKind::ConditionalFalse);
///
/// let classes = CycleEquivalence::compute(&cfg);
/// let class = |edge| classes.class(SeseEdge::Edge(edge));
/// // The loop body is entered and left on the same cycles.
/// assert_eq!(class(stay), class(back));
/// // Entering the loop and leaving it lie on the cycle through the exit.
/// assert_eq!(class(enter), class(leave));
/// assert_eq!(class(enter), classes.class(SeseEdge::Exit(exit)));
/// assert_ne!(class(enter), class(stay));
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CycleEquivalence<N = BlockId, E = EdgeId> {
    /// The class of each edge slot of the closed graph, [`NO_INDEX`] for a slot
    /// outside it.
    classes: Vec<u32>,
    /// The edge bound of the graph, which is also the slot of the virtual
    /// edge into the root.
    edge_bound: usize,
    /// The blocks with a virtual edge to the virtual exit, ascending.
    outlets: Vec<N>,
    /// The number of classes.
    class_count: usize,
    /// The edge identity the classes are queried by.
    edge: PhantomData<E>,
}

impl<N: DenseId, E: DenseId> CycleEquivalence<N, E> {
    /// Compute the cycle-equivalence classes of the edges the root of
    /// `graph` reaches.
    ///
    /// # Panics
    ///
    /// Panics when the root is not a node of `graph`, or when the nodes or
    /// the edges of the closed graph exceed the range of a dense `u32` table.
    #[must_use]
    pub fn compute<G>(graph: &G) -> Self
    where
        G: RootedView<NodeId = N> + EdgeView<EdgeId = E>,
    {
        let closed = ClosedGraph::new(graph);
        let (classes, class_count) = cycle_classes(&closed);
        Self {
            classes,
            edge_bound: closed.edge_bound,
            outlets: closed.outlets,
            class_count,
            edge: PhantomData,
        }
    }

    /// The number of classes, virtual edges' classes included.
    #[must_use]
    pub fn class_count(&self) -> usize {
        self.class_count
    }

    /// The class of `edge`, below [`class_count`](Self::class_count).
    ///
    /// `None` for an edge the root does not reach, a removed edge, an edge
    /// beyond the edge bound, and a [`SeseEdge::Exit`] from a block that is
    /// neither an exit nor a cut root.
    #[must_use]
    pub fn class(&self, edge: SeseEdge<N, E>) -> Option<usize> {
        let edge = edge_slot(self.edge_bound, &self.outlets, edge)?;
        let class = *self.classes.get(edge)?;
        (class != NO_INDEX).then_some(slot(class))
    }
}

/// The class of every slot of `closed`, [`NO_INDEX`] outside it, and the number
/// of classes.
pub(super) fn cycle_classes<N: DenseId>(closed: &ClosedGraph<N>) -> (Vec<u32>, usize) {
    let mut classes = Classes {
        of: vec![NO_INDEX; closed.slot_count()],
        count: 0,
    };
    let tree = SpanningTree::new(closed, &mut classes);
    tree.name_brackets(closed, &mut classes);
    (classes.of, slot(classes.count))
}

/// The class of each slot and the number handed out so far.
struct Classes {
    of: Vec<u32>,
    count: u32,
}

impl Classes {
    /// A class no slot has yet.
    fn fresh(&mut self) -> u32 {
        let class = self.count;
        self.count += 1;
        class
    }
}

/// An undirected depth-first spanning tree of a closed graph, rooted at the
/// virtual exit.
///
/// Every edge that is not a tree edge or a self-loop is a backedge between a
/// node and one of its ancestors, because an undirected search has no cross
/// edges. A backedge is listed at its lower end, where it starts a bracket,
/// and at its upper end, where the bracket ends.
struct SpanningTree {
    /// The preorder number of each node.
    pre: Vec<u32>,
    /// The nodes in preorder.
    order: Vec<u32>,
    /// The tree edge into each node, [`NO_INDEX`] for the root.
    tree_edge: Vec<u32>,
    /// The first backedge whose lower end is each node.
    up_head: Vec<u32>,
    /// The next backedge with the same lower end, by slot.
    up_next: Vec<u32>,
    /// The first backedge whose upper end is each node.
    down_head: Vec<u32>,
    /// The next backedge with the same upper end, by slot.
    down_next: Vec<u32>,
}

impl SpanningTree {
    /// Search `closed` from the virtual exit, giving each self-loop a class of
    /// its own on the way.
    fn new<N: DenseId>(closed: &ClosedGraph<N>, classes: &mut Classes) -> Self {
        let nodes = closed.node_count();
        let slots = closed.slot_count();
        let root = closed.bound;
        let mut tree = Self {
            pre: vec![NO_INDEX; nodes],
            order: Vec::with_capacity(nodes),
            tree_edge: vec![NO_INDEX; nodes],
            up_head: vec![NO_INDEX; nodes],
            up_next: vec![NO_INDEX; slots],
            down_head: vec![NO_INDEX; nodes],
            down_next: vec![NO_INDEX; slots],
        };
        tree.discover(root, NO_INDEX);
        let mut stack = vec![(root, 0_usize)];
        while let Some(frame) = stack.last_mut() {
            let (node, position) = *frame;
            let Some(&edge) = closed.incident(node).get(position) else {
                stack.pop();
                continue;
            };
            frame.1 += 1;
            // The tree edge is skipped by identity, so a parallel edge to the
            // parent is a backedge.
            if edge == tree.tree_edge[node] {
                continue;
            }
            let edge = slot(edge);
            let other = closed.other(edge, node);
            if other == node {
                if classes.of[edge] == NO_INDEX {
                    classes.of[edge] = classes.fresh();
                }
            } else if tree.pre[other] == NO_INDEX {
                tree.discover(other, word(edge));
                stack.push((other, 0));
            } else if tree.pre[other] < tree.pre[node] {
                // Met from its lower end first: from the upper end, the
                // descendant has been searched already.
                tree.up_next[edge] = tree.up_head[node];
                tree.up_head[node] = word(edge);
                tree.down_next[edge] = tree.down_head[other];
                tree.down_head[other] = word(edge);
            }
        }
        tree
    }

    /// Number `node`, reached over `edge`.
    fn discover(&mut self, node: usize, edge: u32) {
        self.pre[node] = word(self.order.len());
        self.order.push(word(node));
        self.tree_edge[node] = edge;
    }

    /// The backedges from `node` to its ancestors.
    fn ups(&self, node: usize) -> impl Iterator<Item = usize> + '_ {
        chain(self.up_head[node], &self.up_next)
    }

    /// The backedges from the descendants of `node` to it.
    fn downs(&self, node: usize) -> impl Iterator<Item = usize> + '_ {
        chain(self.down_head[node], &self.down_next)
    }

    /// Name the bracket set of every tree edge, and give each edge its class.
    fn name_brackets<N: DenseId>(&self, closed: &ClosedGraph<N>, classes: &mut Classes) {
        let nodes = closed.node_count();
        let slots = closed.slot_count();
        let mut lists = BracketLists::new(nodes, slots);
        // The smallest and second-smallest `hi` of the children finished so
        // far, the second from a child other than the first's.
        let mut child_hi = vec![(NO_INDEX, NO_INDEX); nodes];
        // The capping brackets ending at each node.
        let mut capping_head = vec![NO_INDEX; nodes];
        for &node in self.order.iter().rev() {
            let node = slot(node);
            // The highest node a backedge from `node` reaches, by preorder.
            let hi0 = self
                .ups(node)
                .map(|edge| self.pre[closed.other(edge, node)])
                .min()
                .unwrap_or(NO_INDEX);
            let (hi1, hi2) = child_hi[node];

            let mut capping = capping_head[node];
            while capping != NO_INDEX {
                lists.delete(node, slot(capping));
                capping = lists.next_capping[slot(capping)];
            }
            for edge in self.downs(node) {
                lists.delete(node, edge);
                if classes.of[edge] == NO_INDEX {
                    classes.of[edge] = classes.fresh();
                }
            }
            for edge in self.ups(node) {
                lists.push(node, edge);
            }
            // A second child's brackets reach above `node` and above its own
            // backedges: cap them, so that the merged list's top changes until
            // they end. A child whose brackets all end at `node` needs no cap.
            if hi2 < hi0 && hi2 < self.pre[node] {
                let capping = lists.add_capping();
                lists.push(node, capping);
                let ancestor = slot(self.order[slot(hi2)]);
                lists.next_capping[capping] = capping_head[ancestor];
                capping_head[ancestor] = word(capping);
            }

            let tree_edge = self.tree_edge[node];
            if tree_edge == NO_INDEX {
                continue;
            }
            let tree_edge = slot(tree_edge);
            let top = lists.top[node];
            assert!(
                top != NO_INDEX,
                "every edge of the strongly connected closed graph lies on a cycle"
            );
            let top = slot(top);
            let size = lists.size[node];
            if lists.recent_size[top] != size {
                lists.recent_size[top] = size;
                lists.recent_class[top] = classes.fresh();
            }
            let class = lists.recent_class[top];
            classes.of[tree_edge] = class;
            if size == 1 {
                // The tree edge's only bracket, which a capping bracket never
                // is: one rides beside the bracket it caps.
                classes.of[top] = class;
            }

            let parent = closed.other(tree_edge, node);
            lists.splice(node, parent);
            let hi = hi0.min(hi1);
            let (first, second) = &mut child_hi[parent];
            if hi < *first {
                *second = *first;
                *first = hi;
            } else if hi < *second {
                *second = hi;
            }
        }
    }
}

/// The slots of a linked list from `head` through `next`.
fn chain(head: u32, next: &[u32]) -> impl Iterator<Item = usize> + '_ {
    let start = (head != NO_INDEX).then_some(slot(head));
    core::iter::successors(start, move |&edge| {
        let following = next[edge];
        (following != NO_INDEX).then_some(slot(following))
    })
}

/// The bracket list of every node, as doubly linked lists over one arena.
///
/// A bracket below the slot count is the backedge in that slot; the rest are
/// capping brackets. Every operation is O(1).
struct BracketLists {
    /// The bracket above each one in its list, toward the top.
    above: Vec<u32>,
    /// The bracket below each one in its list.
    below: Vec<u32>,
    /// The size of the list when each bracket was last a top, `0` before.
    recent_size: Vec<u32>,
    /// The class given to the tree edge that last had each bracket on top.
    recent_class: Vec<u32>,
    /// The next capping bracket that ends at the same node.
    next_capping: Vec<u32>,
    /// The top bracket of each node's list.
    top: Vec<u32>,
    /// The bottom bracket of each node's list.
    bottom: Vec<u32>,
    /// The number of brackets in each node's list.
    size: Vec<u32>,
}

impl BracketLists {
    /// Empty lists for `nodes` nodes, with room for `slots` backedges.
    fn new(nodes: usize, slots: usize) -> Self {
        Self {
            above: vec![NO_INDEX; slots],
            below: vec![NO_INDEX; slots],
            recent_size: vec![0; slots],
            recent_class: vec![NO_INDEX; slots],
            next_capping: vec![NO_INDEX; slots],
            top: vec![NO_INDEX; nodes],
            bottom: vec![NO_INDEX; nodes],
            size: vec![0; nodes],
        }
    }

    /// A new capping bracket, in no list yet.
    fn add_capping(&mut self) -> usize {
        let capping = self.above.len();
        self.above.push(NO_INDEX);
        self.below.push(NO_INDEX);
        self.recent_size.push(0);
        self.recent_class.push(NO_INDEX);
        self.next_capping.push(NO_INDEX);
        capping
    }

    /// Put `bracket` on top of the list of `node`.
    fn push(&mut self, node: usize, bracket: usize) {
        let old = self.top[node];
        self.above[bracket] = NO_INDEX;
        self.below[bracket] = old;
        if old == NO_INDEX {
            self.bottom[node] = word(bracket);
        } else {
            self.above[slot(old)] = word(bracket);
        }
        self.top[node] = word(bracket);
        self.size[node] += 1;
    }

    /// Take `bracket` out of the list of `node`, wherever it is.
    fn delete(&mut self, node: usize, bracket: usize) {
        let (above, below) = (self.above[bracket], self.below[bracket]);
        if above == NO_INDEX {
            self.top[node] = below;
        } else {
            self.below[slot(above)] = below;
        }
        if below == NO_INDEX {
            self.bottom[node] = above;
        } else {
            self.above[slot(below)] = above;
        }
        self.size[node] -= 1;
    }

    /// Move the list of `child` under the list of `parent`.
    fn splice(&mut self, child: usize, parent: usize) {
        let (top, bottom) = (self.top[child], self.bottom[child]);
        if top == NO_INDEX {
            return;
        }
        let parent_bottom = self.bottom[parent];
        if parent_bottom == NO_INDEX {
            self.top[parent] = top;
        } else {
            self.below[slot(parent_bottom)] = top;
            self.above[slot(top)] = parent_bottom;
        }
        self.bottom[parent] = bottom;
        self.size[parent] += self.size[child];
    }
}
