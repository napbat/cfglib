//! Weak control-closure of a set of blocks, on any directed graph.
//!
//! Danicic et al. ("A unifying theory of control dependence and its
//! application to arbitrary program structures", TCS 412(49), 2011) close a
//! set `V'` of vertices under control without an exit, a root or a
//! post-dominator tree:
//!
//! - a `V'`-path is a nontrivial path whose last vertex is the only one in
//!   `V'`;
//! - a vertex is `V'`-weakly deciding when two `V'`-paths leave it that share
//!   no vertex but it, so it chooses which vertex of `V'` comes next;
//! - the weak control-closure of `V'` is `V'` with every `V'`-weakly deciding
//!   vertex reachable from `V'`: the smallest superset of `V'` none of whose
//!   reachable vertices decides for it (Léchenet, Kosmatov and Le Gall,
//!   "Efficient computation of arbitrary control dependencies", TCS 969,
//!   2023, lemma 3).
//!
//! The closure here is lDFS, the algorithm of Léchenet et al. proved in Why3.
//! Every vertex of the growing set is propagated once, backward, labeling the
//! vertices that reach it before any other vertex of the set. A vertex that
//! already carried another label and has two successors is a candidate, and
//! it is confirmed when one of its successors still carries a different
//! label; confirmed vertices join the set and are propagated in turn. At the
//! end the set is `V'` with every `V'`-weakly deciding vertex, and the closure
//! is what of it `V'` reaches.

extern crate alloc;
use alloc::vec::Vec;

use crate::graph::view::{DenseId, GraphView};

/// Marks a vertex without a label.
const UNLABELED: usize = usize::MAX;

/// The buffers [`weak_control_closure`] fills, owned by the caller.
///
/// The tables grow to the largest graph seen and are restored by every call
/// over the vertices it labeled, so a call that closes a few blocks of a
/// large graph costs only the blocks it visits once the tables have grown.
#[derive(Debug, Clone, Default)]
pub struct ControlClosureScratch {
    /// Whether each vertex is in the growing set.
    closed: Vec<bool>,
    /// The vertex of the set each vertex last reached first, [`UNLABELED`]
    /// for none.
    label: Vec<usize>,
    /// The search that last visited each vertex, so a search needs no
    /// clearing pass.
    visit: Vec<u32>,
    /// The number of the current search.
    epoch: u32,
    /// Vertices of the set not propagated yet.
    work: Vec<usize>,
    /// Vertices a search has yet to expand.
    stack: Vec<usize>,
    /// Candidates of the current propagation.
    candidates: Vec<usize>,
    /// Every vertex labeled, each once, to read and restore.
    touched: Vec<usize>,
}

impl ControlClosureScratch {
    /// Scratch holding nothing yet.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

/// Write to `out` the weak control-closure of `seeds` in `graph`: the seeds,
/// and every block reachable from them that is seed-weakly deciding,
/// ascending by dense index.
///
/// A block is seed-weakly deciding when two paths leave it for two different
/// seeds, sharing no block but it and passing no other seed on the way. Any
/// directed graph will do: the closure needs no root, no exit and no
/// post-dominator tree, and it is a set, so it does not depend on the order
/// of successor lists or of `seeds`.
///
/// The closure takes no account of the choice between reaching a seed and
/// reaching none: a loop's exit test does not decide for a seed in the loop
/// body, because leaving the loop leads to no seed. It also keeps only what
/// the seeds reach, so a branch before every seed joins only when a seed
/// reaches it, as the entry does. On a graph whose reachable blocks all
/// reach an exit, the closure of a set that holds the entry and every exit
/// is that set with its iterated control-dependence frontier under
/// [`PostDominatorTree`](crate::PostDominatorTree), as the tests check on
/// every small graph.
///
/// The algorithm is lDFS (Léchenet, Kosmatov and Le Gall, TCS 969, 2023):
/// each block of the closure propagates once, backward, so the cost is
/// O(n·m) in the worst case, O(n²) for bounded out-degree, and a call
/// allocates nothing once `scratch` and `out` have grown. `out` is cleared
/// first.
///
/// # Panics
///
/// Panics when a seed, or a block a successor or predecessor list names, has
/// an index beyond the graph's node bound.
///
/// # Examples
///
/// The branch decides which of two seeds runs next, so it joins their
/// closure:
///
/// ```
/// use cfglib::{Cfg, ControlClosureScratch, EdgeKind, weak_control_closure};
///
/// let mut cfg = Cfg::<u32>::new();
/// let entry = cfg.entry();
/// let branch = cfg.new_block();
/// let left = cfg.new_block();
/// let right = cfg.new_block();
/// let join = cfg.new_block();
/// cfg.add_edge(entry, branch, EdgeKind::Unconditional);
/// cfg.add_edge(branch, left, EdgeKind::ConditionalTrue);
/// cfg.add_edge(branch, right, EdgeKind::ConditionalFalse);
/// cfg.add_edge(left, join, EdgeKind::Unconditional);
/// cfg.add_edge(right, join, EdgeKind::Unconditional);
///
/// let mut scratch = ControlClosureScratch::new();
/// let mut closure = Vec::new();
/// weak_control_closure(&cfg, [entry, left, right], &mut scratch, &mut closure);
/// assert_eq!(closure, [entry, branch, left, right]);
///
/// // A seed after the join leaves the branch nothing to decide.
/// weak_control_closure(&cfg, [entry, join], &mut scratch, &mut closure);
/// assert_eq!(closure, [entry, join]);
/// ```
pub fn weak_control_closure<G: GraphView>(
    graph: &G,
    seeds: impl IntoIterator<Item = G::NodeId>,
    scratch: &mut ControlClosureScratch,
    out: &mut Vec<G::NodeId>,
) {
    out.clear();
    let bound = graph.node_bound();
    if scratch.closed.len() < bound {
        scratch.closed.resize(bound, false);
        scratch.label.resize(bound, UNLABELED);
        scratch.visit.resize(bound, 0);
    }
    // `out` holds the distinct seeds until the reachability search.
    for seed in seeds {
        let index = seed.index();
        assert!(index < bound, "a seed is not a node of the graph");
        if !scratch.closed[index] {
            scratch.close(index);
            out.push(seed);
        }
    }

    while let Some(vertex) = scratch.work.pop() {
        scratch.propagate(graph, vertex);
        let ControlClosureScratch {
            label, candidates, ..
        } = &mut *scratch;
        // A candidate decides when one of its successors still reaches
        // another vertex of the set first.
        candidates.retain(|&candidate| {
            graph
                .successors(G::NodeId::from_index(candidate))
                .any(|next| {
                    let next = label[next.index()];
                    next != UNLABELED && next != vertex
                })
        });
        for position in 0..scratch.candidates.len() {
            scratch.close(scratch.candidates[position]);
        }
    }

    // The set is now the seeds with every seed-weakly deciding vertex; the
    // closure keeps what the seeds reach.
    let epoch = scratch.next_epoch();
    let ControlClosureScratch {
        closed,
        label,
        visit,
        stack,
        touched,
        ..
    } = &mut *scratch;
    stack.clear();
    for seed in out.iter() {
        visit[seed.index()] = epoch;
        stack.push(seed.index());
    }
    while let Some(vertex) = stack.pop() {
        for next in graph.successors(G::NodeId::from_index(vertex)) {
            let next = next.index();
            if visit[next] != epoch {
                visit[next] = epoch;
                stack.push(next);
            }
        }
    }
    out.clear();
    out.extend(
        touched
            .iter()
            .filter(|&&vertex| closed[vertex] && visit[vertex] == epoch)
            .map(|&vertex| G::NodeId::from_index(vertex)),
    );
    out.sort_unstable_by_key(|vertex| vertex.index());

    for &vertex in touched.iter() {
        closed[vertex] = false;
        label[vertex] = UNLABELED;
    }
    touched.clear();
}

impl ControlClosureScratch {
    /// Put `vertex` in the set, as its own label, to be propagated.
    fn close(&mut self, vertex: usize) {
        if self.label[vertex] == UNLABELED {
            self.touched.push(vertex);
        }
        self.closed[vertex] = true;
        self.label[vertex] = vertex;
        self.work.push(vertex);
    }

    /// Number a new search, so that no vertex looks visited by it.
    fn next_epoch(&mut self) -> u32 {
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.visit.fill(0);
            self.epoch = 1;
        }
        self.epoch
    }

    /// Label with `vertex` every vertex whose paths reach `vertex` before
    /// any other vertex of the set, and collect as candidates those of them
    /// that carried another label and have two successors.
    fn propagate<G: GraphView>(&mut self, graph: &G, vertex: usize) {
        let epoch = self.next_epoch();
        self.candidates.clear();
        self.stack.clear();
        self.stack.push(vertex);
        self.visit[vertex] = epoch;
        while let Some(current) = self.stack.pop() {
            for before in graph.predecessors(G::NodeId::from_index(current)) {
                let index = before.index();
                if self.closed[index] || self.visit[index] == epoch {
                    continue;
                }
                self.visit[index] = epoch;
                if self.label[index] == UNLABELED {
                    self.touched.push(index);
                } else if graph.successors(before).nth(1).is_some() {
                    self.candidates.push(index);
                }
                self.label[index] = vertex;
                self.stack.push(index);
            }
        }
    }
}

#[cfg(test)]
mod tests;
