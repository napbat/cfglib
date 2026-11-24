//! A low-high order of a dominator tree, from the by-products of the run
//! that built it.
//!
//! A low-high order is a preorder of the tree in which every node other than
//! the root has an edge from its parent, or two edges from a node before it
//! and from a node after it that it does not dominate. It certifies the tree
//! (see `certificate`). This module builds one in two steps of Georgiadis and
//! Tarjan, "Dominator Tree Certification and Divergent Spanning Trees" (TALG
//! 12(1), 2016), section 6:
//!
//! 1. Algorithm 8 colors each node from its semidominator and relative
//!    dominator, and so picks two of its in-edges, from its search parent and
//!    from the last node of a semidominator path, as its parents in two
//!    spanning trees `B` and `R` whose paths to a node share only its
//!    dominators.
//! 2. Algorithm 7 rewrites each parent to the child of the node's dominator
//!    above it (the derived arc), then removes nodes one at a time, each one
//!    with more parents than children, handing a child it has to its own
//!    parent. Put back in reverse, each node goes next to one of its parents,
//!    on the side of the other, so it lands between them. Algorithm 5 of the
//!    same paper answers the "which side" questions off-line: every node gets
//!    an interval within its anchor's, sized by the nodes that will later be
//!    placed against it.
//!
//! Every step is linear in the size of the tree once the tree is known, and
//! none recurses. The checker does not trust any of this: a wrong order fails
//! the check like a wrong tree does.

extern crate alloc;
use alloc::vec::Vec;

use super::ConstructionBuffers;
use crate::graph::dominator::numbering::Numbered;
use crate::graph::dominator::{ABSENT, LowHighOrder, slot, word};
use crate::graph::view::{DenseId, GraphView};

/// Every buffer [`ConstructionBuffers::low_high_order`] fills, owned by the
/// caller.
#[derive(Debug, Clone, Default)]
pub(in crate::graph::dominator) struct LowHighBuffers {
    /// Whether each node, by depth-first number, takes its semidominator path
    /// as its parent in `B`.
    blue: Vec<bool>,
    /// Each node's parent in `B`, then its derived arc, by dense index.
    first: Vec<u32>,
    /// Each node's parent in `R`, then its derived arc, by dense index.
    second: Vec<u32>,
    /// The first derived-arc query whose tail is each node.
    query_head: Vec<u32>,
    /// The next query with the same tail. Query `2v` asks for `first[v]`, and
    /// `2v + 1` for `second[v]`.
    query_next: Vec<u32>,
    /// The dominator-tree path from the root to the node being swept.
    path: Vec<u32>,
    /// How many nodes have each node as their `first`, then as their `second`.
    first_children: Vec<u32>,
    second_children: Vec<u32>,
    /// The exclusive or of those nodes, which is the node itself when there
    /// is one.
    first_child: Vec<u32>,
    second_child: Vec<u32>,
    /// Whether each node has been queued for removal.
    queued: Vec<bool>,
    /// The nodes that can be removed.
    work: Vec<u32>,
    /// The nodes in the order they were removed.
    removed: Vec<u32>,
    /// The sibling each node is placed against, [`ABSENT`] for anywhere among
    /// its siblings, and the sibling whose side it goes to.
    anchor: Vec<u32>,
    side: Vec<u32>,
    /// The number of positions each node's interval spans: itself and every
    /// node placed against it, directly or not. For a node as a parent, the
    /// number of its children.
    size: Vec<u32>,
    children: Vec<u32>,
    /// The interval of each node among its siblings, and the upper end of the
    /// interval still free in front of a parent's children.
    low: Vec<u32>,
    high: Vec<u32>,
    free: Vec<u32>,
    /// Where each parent's children start in `slots`.
    start: Vec<u32>,
    /// Every parent's children in their final order, parent after parent.
    slots: Vec<u32>,
}

impl ConstructionBuffers {
    /// Write to `certificate` a low-high order of the tree whose parent table
    /// is `idom` and whose numbering is `tree`.
    ///
    /// The last run must have recorded, over `graph`, and built that tree.
    pub(in crate::graph::dominator) fn low_high_order<G: GraphView, N: DenseId>(
        &self,
        graph: &G,
        idom: &[u32],
        tree: &Numbered<N>,
        buffers: &mut LowHighBuffers,
        certificate: &mut LowHighOrder<N>,
    ) {
        debug_assert!(self.record, "a low-high order needs a recorded run");
        let root = tree
            .order
            .first()
            .expect("a recorded run reaches at least its root")
            .index();
        self.independent_parents(graph, idom, buffers);
        derive(idom, tree, buffers);
        let removed = remove(root, idom.len(), buffers);
        debug_assert_eq!(
            removed,
            tree.order.len() - 1,
            "every node but the root can be removed when the tree is the dominator tree"
        );
        place(idom, buffers);
        number(root, idom, tree, buffers, certificate);
    }

    /// Algorithm 8, then step 1 of Algorithm 7: give every reachable node
    /// other than the root two parents, the same one when its dominator has
    /// an edge to it.
    fn independent_parents<G: GraphView>(
        &self,
        graph: &G,
        idom: &[u32],
        buffers: &mut LowHighBuffers,
    ) {
        let reached = self.vertex.len();
        let LowHighBuffers {
            blue,
            first,
            second,
            ..
        } = buffers;
        blue.clear();
        blue.resize(reached, false);
        first.clear();
        first.resize(idom.len(), ABSENT);
        second.clear();
        second.resize(idom.len(), ABSENT);
        for number in 1..reached {
            // A node is blue when its semidominator is its dominator, which
            // is when its relative dominator has the same semidominator (the
            // relative dominator is the node itself when its semidominator is
            // its parent), or when its relative dominator is red.
            let relative = slot(self.relative[number]);
            blue[number] = self.semi[number] == self.semi[relative] || !blue[relative];
            let node = slot(self.vertex[number]);
            let parent = self.vertex[slot(self.parent[number])];
            let last = self.vertex[slot(self.best[number])];
            (first[node], second[node]) = if blue[number] {
                (last, parent)
            } else {
                (parent, last)
            };
            let dominator = idom[node];
            if graph
                .predecessors(G::NodeId::from_index(node))
                .any(|before| before.index() == slot(dominator))
            {
                first[node] = dominator;
                second[node] = dominator;
            }
        }
    }
}

/// Step 2 of Algorithm 7: replace each parent that is not the node's
/// dominator by its ancestor among the node's siblings, found by one sweep of
/// the tree in preorder that keeps the path from the root.
fn derive<N: DenseId>(idom: &[u32], tree: &Numbered<N>, buffers: &mut LowHighBuffers) {
    let bound = idom.len();
    let LowHighBuffers {
        first,
        second,
        query_head,
        query_next,
        path,
        ..
    } = buffers;
    query_head.clear();
    query_head.resize(bound, ABSENT);
    query_next.clear();
    query_next.resize(2 * bound, ABSENT);
    for node in tree.order.iter().map(|node| node.index()) {
        for (which, tail) in [first[node], second[node]].into_iter().enumerate() {
            if tail != ABSENT && tail != idom[node] {
                let query = word(2 * node + which);
                query_next[slot(query)] = query_head[slot(tail)];
                query_head[slot(tail)] = query;
            }
        }
    }
    path.clear();
    for tail in tree.order.iter().map(|node| node.index()) {
        path.truncate(slot(tree.level[tail]));
        path.push(word(tail));
        let mut query = query_head[tail];
        while query != ABSENT {
            let node = slot(query) / 2;
            let sibling = path[slot(tree.level[node])];
            if slot(query).is_multiple_of(2) {
                first[node] = sibling;
            } else {
                second[node] = sibling;
            }
            query = query_next[slot(query)];
        }
    }
}

/// Step 4 of Algorithm 7, without the recursion: remove, while one is left,
/// a node with more parents than children, and give a child it has its own
/// parent in that tree. Returns how many nodes were removed.
///
/// A node has one parent when both are its dominator and two otherwise, and
/// removing a node never adds a child to a node that stays, so a node that
/// can be removed stays removable and one worklist pass finds them all.
fn remove(root: usize, bound: usize, buffers: &mut LowHighBuffers) -> usize {
    let LowHighBuffers {
        first,
        second,
        first_children,
        second_children,
        first_child,
        second_child,
        queued,
        work,
        removed,
        anchor,
        side,
        ..
    } = buffers;
    for table in [
        &mut *first_children,
        &mut *second_children,
        &mut *first_child,
        &mut *second_child,
    ] {
        table.clear();
        table.resize(bound, 0);
    }
    for node in 0..bound {
        if first[node] != ABSENT {
            first_children[slot(first[node])] += 1;
            first_child[slot(first[node])] ^= word(node);
            second_children[slot(second[node])] += 1;
            second_child[slot(second[node])] ^= word(node);
        }
    }
    queued.clear();
    queued.extend((0..bound).map(|node| {
        node != root && removable(node, first, second, first_children, second_children)
    }));
    work.clear();
    work.extend((0..bound).filter(|&node| queued[node]).map(word));
    anchor.clear();
    anchor.resize(bound, ABSENT);
    side.clear();
    side.resize(bound, ABSENT);
    removed.clear();
    while let Some(node) = work.pop() {
        let node = slot(node);
        removed.push(word(node));
        let (low, high) = (slot(first[node]), slot(second[node]));
        first_children[low] -= 1;
        first_child[low] ^= word(node);
        second_children[high] -= 1;
        second_child[high] ^= word(node);
        if low != high {
            if first_children[node] == 1 {
                let child = slot(first_child[node]);
                first[child] = word(low);
                first_children[low] += 1;
                first_child[low] ^= word(child);
                (anchor[node], side[node]) = (word(low), word(high));
            } else if second_children[node] == 1 {
                let child = slot(second_child[node]);
                second[child] = word(high);
                second_children[high] += 1;
                second_child[high] ^= word(child);
                (anchor[node], side[node]) = (word(high), word(low));
            } else {
                (anchor[node], side[node]) = (word(low), word(high));
            }
        }
        for parent in [low, high] {
            if parent != root
                && !queued[parent]
                && removable(parent, first, second, first_children, second_children)
            {
                queued[parent] = true;
                work.push(word(parent));
            }
        }
    }
    removed.len()
}

/// Whether `node` has more parents than children, and so can be removed.
fn removable(
    node: usize,
    first: &[u32],
    second: &[u32],
    first_children: &[u32],
    second_children: &[u32],
) -> bool {
    let parents = if first[node] == second[node] { 1 } else { 2 };
    first[node] != ABSENT && parents > first_children[node] + second_children[node]
}

/// Put the removed nodes back in reverse, each next to its anchor on the side
/// of the other parent, with Algorithm 5: every node first learns how many
/// positions it and the nodes placed against it will take, then takes that
/// many from its anchor's interval, or from its parent's free space when it
/// may go anywhere. When all are placed, each node's interval is the single
/// position among its siblings, counted from one.
fn place(idom: &[u32], buffers: &mut LowHighBuffers) {
    let bound = idom.len();
    let LowHighBuffers {
        removed,
        anchor,
        side,
        size,
        children,
        low,
        high,
        free,
        ..
    } = buffers;
    size.clear();
    size.resize(bound, 1);
    children.clear();
    children.resize(bound, 0);
    // A node is removed before every node placed against it, so the removal
    // order sees every contribution to a size before the size is passed on.
    for &node in removed.iter() {
        let node = slot(node);
        if anchor[node] == ABSENT {
            children[slot(idom[node])] += size[node];
        } else {
            size[slot(anchor[node])] += size[node];
        }
    }
    free.clear();
    free.extend_from_slice(children);
    low.clear();
    low.resize(bound, 0);
    high.clear();
    high.resize(bound, 0);
    for &node in removed.iter().rev() {
        let node = slot(node);
        let span = size[node];
        if anchor[node] == ABSENT {
            let parent = slot(idom[node]);
            high[node] = free[parent];
            low[node] = free[parent] + 1 - span;
            free[parent] -= span;
            continue;
        }
        let (anchor, other) = (slot(anchor[node]), slot(side[node]));
        if high[other] < low[anchor] {
            low[node] = low[anchor];
            high[node] = low[anchor] + span - 1;
            low[anchor] += span;
        } else {
            high[node] = high[anchor];
            low[node] = high[anchor] + 1 - span;
            high[anchor] -= span;
        }
    }
}

/// Write the tree's preorder with every node's children in their placed
/// order, and each node's position in it: a node's position follows its
/// parent's, after the subtrees of the siblings placed before it.
fn number<N: DenseId>(
    root: usize,
    idom: &[u32],
    tree: &Numbered<N>,
    buffers: &mut LowHighBuffers,
    certificate: &mut LowHighOrder<N>,
) {
    let LowHighBuffers {
        children,
        low,
        start,
        slots,
        ..
    } = buffers;
    let LowHighOrder { order, position } = certificate;
    start.clear();
    let mut next = 0;
    for &count in children.iter() {
        start.push(next);
        next += count;
    }
    slots.clear();
    slots.resize(slot(next), ABSENT);
    for node in tree.order.iter().map(|node| node.index()) {
        if node != root {
            slots[slot(start[slot(idom[node])] + low[node] - 1)] = word(node);
        }
    }
    position.clear();
    position.resize(idom.len(), ABSENT);
    position[root] = 0;
    for parent in tree.order.iter().map(|node| node.index()) {
        let mut next = position[parent] + 1;
        let first = slot(start[parent]);
        for &child in &slots[first..first + slot(children[parent])] {
            position[slot(child)] = next;
            next += tree.last[slot(child)] - tree.pre[slot(child)] + 1;
        }
    }
    order.clear();
    order.extend_from_slice(&tree.order);
    for &node in &tree.order {
        order[slot(position[node.index()])] = node;
    }
}
