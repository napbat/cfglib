//! Bidirected Dyck reachability: the classes of nodes that reach each other
//! along balanced-parenthesis paths.
//!
//! An edge `u --(k--> v` opens parenthesis `k`, and in a *bidirected* graph
//! it comes with its inverse `v --)k--> u`, which closes it. A path is
//! balanced when its labels spell a word of the Dyck language
//! `S -> S S | (k S )k | ε`. Field-sensitive alias analysis has this shape
//! (store `x.f = y` is `y --(f--> x`, and load `y = x.f` is its inverse), and
//! so has context-sensitive data dependence over call and return edges.
//!
//! On a bidirected graph, balanced reachability is an equivalence relation,
//! and it is the least one that contains the plain edges and merges `a` with
//! `c` whenever `a --(k--> b` and `c --(k--> d` with `b` and `d` already
//! equivalent: the path `a (k b … d )k c` is balanced. [`BidirectedDyck`]
//! computes that closure with a union-find over the nodes and one table per
//! class that keeps a single source node for each label entering the class,
//! after K. Chatterjee, B. Choudhary and A. Pavlogiannis, *Optimal Dyck
//! reachability for data-dependence and alias analysis*, POPL 2018. Because
//! classes only ever merge, the same state accepts edges one at a time
//! ([`BidirectedDyck::insert`]), and the answer does not depend on the order
//! in which the edges arrive. Deleting an edge is not supported.
//!
//! The consumer builds the labeled graph from its IR (field accesses,
//! calls); this module owns only the reachability.

extern crate alloc;
use alloc::vec::Vec;
use core::mem;

use crate::graph::edge_view::{EdgeRef, EdgeView};
use crate::graph::view::DenseId;
use crate::union_find::DisjointSet;

/// The balanced-path classes of a bidirected Dyck graph, open to new edges.
///
/// Nodes are `0..node_bound`. Every node starts in a class of its own; an
/// edge merges classes as the closure in the [module docs](self) requires.
/// The representative of a class is its minimum member, so
/// [`class_of`](Self::class_of) is deterministic whatever order the edges
/// were inserted in.
///
/// # Cost
///
/// An insertion looks its label up in the table of one class, and a merge of
/// two classes moves the smaller label table into the larger one. With a
/// constant number of distinct labels this is O(1) per edge and per merge,
/// and the union-find operations dominate: amortized O(log n) each with
/// minimum representatives, so a whole build is O((m + n) log n) for `m`
/// edges and `n` nodes, and an insertion is amortized near-constant plus the
/// merges it causes.
///
/// # Examples
///
/// ```
/// use cfglib::BidirectedDyck;
///
/// // `p.f = a; q.f = b; p = q`: the stores are `a --(f--> p` and
/// // `b --(f--> q`, the copy a plain edge. Loading `p.f` may read `b`.
/// let (a, b, p, q) = (0, 1, 2, 3);
/// let field_f = 0;
/// let mut dyck = BidirectedDyck::new(4);
/// dyck.insert(a, p, field_f);
/// dyck.insert(b, q, field_f);
/// assert!(!dyck.same_class(a, b));
///
/// dyck.insert_plain(p, q);
/// assert!(dyck.same_class(a, b));
/// assert_eq!(dyck.class_of(b), a);
/// ```
#[derive(Debug, Clone)]
pub struct BidirectedDyck {
    /// The classes, each represented by its minimum member.
    sets: DisjointSet,
    /// Per class representative, one source node for each label with an
    /// edge into the class, sorted by label. Every source of one label into
    /// one class belongs to one class, so one source stands for them all.
    /// Empty for a node that no longer represents its class.
    sources: Vec<Vec<(u32, usize)>>,
    /// Pairs of nodes that belong to one class and may not be merged yet.
    pending: Vec<(usize, usize)>,
}

impl BidirectedDyck {
    /// The `node_bound` nodes `0..node_bound`, each in a class of its own.
    #[must_use]
    pub fn new(node_bound: usize) -> Self {
        Self {
            sets: DisjointSet::new(node_bound),
            sources: alloc::vec![Vec::new(); node_bound],
            pending: Vec::new(),
        }
    }

    /// The number of nodes, the bound every node id is below.
    #[must_use]
    pub fn node_bound(&self) -> usize {
        self.sets.len()
    }

    /// Insert the edge `from --(label--> to` and its inverse
    /// `to --)label--> from`, merging the classes it makes equivalent.
    ///
    /// # Panics
    ///
    /// Panics when `from` or `to` is not below [`node_bound`](Self::node_bound).
    pub fn insert(&mut self, from: usize, to: usize, label: u32) {
        self.add_labeled(from, to, label);
        self.settle();
    }

    /// Insert the plain (unlabeled) edge between `a` and `b`, which a
    /// balanced path crosses in either direction, merging their classes and
    /// every class that merge makes equivalent.
    ///
    /// # Panics
    ///
    /// Panics when `a` or `b` is not below [`node_bound`](Self::node_bound).
    pub fn insert_plain(&mut self, a: usize, b: usize) {
        self.add_plain(a, b);
        self.settle();
    }

    /// Whether `a` reaches `b` along a balanced path (and so `b` reaches `a`).
    ///
    /// # Panics
    ///
    /// Panics when `a` or `b` is not below [`node_bound`](Self::node_bound).
    pub fn same_class(&mut self, a: usize, b: usize) -> bool {
        self.sets.find(a) == self.sets.find(b)
    }

    /// The minimum node of `node`'s class.
    ///
    /// # Panics
    ///
    /// Panics when `node` is not below [`node_bound`](Self::node_bound).
    pub fn class_of(&mut self, node: usize) -> usize {
        self.sets.find(node)
    }

    /// Record `from --(label--> to` without merging the classes it implies.
    fn add_labeled(&mut self, from: usize, to: usize, label: u32) {
        assert!(
            from < self.node_bound() && to < self.node_bound(),
            "edge endpoint is out of range"
        );
        let class = self.sets.find(to);
        let table = &mut self.sources[class];
        match table.binary_search_by_key(&label, |&(key, _)| key) {
            Ok(at) => self.pending.push((table[at].1, from)),
            Err(at) => table.insert(at, (label, from)),
        }
    }

    /// Record the plain edge `a — b` without merging the classes it implies.
    fn add_plain(&mut self, a: usize, b: usize) {
        assert!(
            a < self.node_bound() && b < self.node_bound(),
            "edge endpoint is out of range"
        );
        self.pending.push((a, b));
    }

    /// Merge every pending pair, and every pair those merges imply, until
    /// no class has two sources of one label.
    fn settle(&mut self) {
        while let Some((a, b)) = self.pending.pop() {
            let (left, right) = (self.sets.find(a), self.sets.find(b));
            if left == right {
                continue;
            }
            self.sets.union_toward_min(left, right);
            let (root, merged) = (left.min(right), left.max(right));
            let mut kept = mem::take(&mut self.sources[root]);
            let mut moved = mem::take(&mut self.sources[merged]);
            if kept.len() < moved.len() {
                mem::swap(&mut kept, &mut moved);
            }
            for (label, source) in moved {
                match kept.binary_search_by_key(&label, |&(key, _)| key) {
                    Ok(at) => self.pending.push((kept[at].1, source)),
                    Err(at) => kept.insert(at, (label, source)),
                }
            }
            self.sources[root] = kept;
        }
    }
}

/// The bidirected Dyck classes of every edge of `graph`.
///
/// `label` names the parenthesis an edge opens: `Some(k)` reads the edge as
/// `source --(k--> target` with the implicit inverse `target --)k--> source`,
/// and `None` as a plain edge that a balanced path crosses either way. Nodes
/// are the dense indices `0..graph.node_bound()`; a removed node keeps a
/// class of its own. The result accepts more edges through
/// [`BidirectedDyck::insert`].
///
/// The batch build records every edge first and merges once, which is the
/// algorithm of Chatterjee, Choudhary and Pavlogiannis; see
/// [`BidirectedDyck`] for the cost.
///
/// # Examples
///
/// ```
/// use cfglib::{Graph, bidirected_dyck_classes};
///
/// // Two calls of one function: `call_k` opens parenthesis k on the way in.
/// let mut graph = Graph::new();
/// let [arg1, arg2, param, ret, out1, out2] = [(); 6].map(|()| graph.add_node(()));
/// graph.add_edge(arg1, param, Some(1));
/// graph.add_edge(arg2, param, Some(2));
/// graph.add_edge(out1, ret, Some(1));
/// graph.add_edge(out2, ret, Some(2));
/// graph.add_edge(param, ret, None); // the function returns its parameter
///
/// let mut classes = bidirected_dyck_classes(&graph, |edge| *edge.data());
/// assert!(classes.same_class(arg1.index(), out1.index()));
/// assert!(!classes.same_class(arg1.index(), out2.index()), "matched calls only");
/// ```
#[must_use]
pub fn bidirected_dyck_classes<G: EdgeView>(
    graph: &G,
    label: impl Fn(EdgeRef<'_, G::NodeId, G::EdgeId, G::EdgeData>) -> Option<u32>,
) -> BidirectedDyck {
    let mut dyck = BidirectedDyck::new(graph.node_bound());
    for id in graph.edge_ids() {
        let edge = graph.edge(id);
        let (source, target) = (edge.source().index(), edge.target().index());
        match label(edge) {
            Some(label) => dyck.add_labeled(source, target, label),
            None => dyck.add_plain(source, target),
        }
    }
    dyck.settle();
    dyck
}

#[cfg(test)]
mod tests;
