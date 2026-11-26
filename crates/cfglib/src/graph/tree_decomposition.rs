//! Tree decompositions by greedy elimination.
//!
//! A tree decomposition of a graph is a tree of bags of nodes such that every
//! node is in some bag, both ends of every edge are in one bag, and the bags
//! that hold a node form a connected subtree. Its width is the size of its
//! largest bag less one. Dynamic programs over a decomposition of small width
//! (register allocation, constraint satisfaction) run in time linear in the
//! graph, so a graph whose structured form has gotos or irreducible loops,
//! which [`SplDecomposition::compute`](crate::SplDecomposition::compute)
//! does not decompose, can still be decomposed here.
//!
//! [`tree_decomposition`] eliminates the nodes of the undirected simple graph
//! underlying a view one at a time, as Bodlaender and Koster survey
//! ("Treewidth computations I. Upper bounds", Information and Computation,
//! 2010): the next node is the one with the fewest neighbors
//! ([`EliminationHeuristic::MinDegree`]) or the fewest missing edges among
//! them ([`EliminationHeuristic::MinFill`]), ties broken by the smallest
//! dense index. Eliminating a node makes its neighbors a clique and emits the
//! bag of the node and its neighbors, whose parent is the bag of the neighbor
//! eliminated next. The heuristics carry no guarantee on the width. Thorup
//! ("All Structured Programs Have Small Tree Width and Good Register
//! Allocation", Information and Computation, 1998) shows that structured
//! programs have small treewidth; how close the heuristics come to it on
//! real control-flow graphs has not been measured here.

extern crate alloc;
use alloc::collections::BTreeSet;
use alloc::vec;
use alloc::vec::Vec;

use crate::graph::epoch::EpochMarks;
use crate::graph::view::{DenseId, GraphView};
use crate::identity::{NO_INDEX, check_bound, slot, word};

#[cfg(test)]
mod tests;

/// The rule that picks the next node to eliminate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EliminationHeuristic {
    /// The node with the fewest neighbors left. Cheaper per step.
    MinDegree,
    /// The node whose neighbors miss the fewest edges between them, the fewest
    /// edges its elimination adds. Usually narrower, and slower by about the
    /// width per step.
    MinFill,
}

/// A tree decomposition of the undirected simple graph underlying a view.
///
/// There is one bag per node, in the order the nodes were eliminated: the bag
/// of a node holds it and the neighbors it had left, and its parent is a
/// later bag. A graph of several components has a forest, one tree per
/// component, and joining the roots in any way keeps it a decomposition. The
/// bags are one flat table of runs.
///
/// # Examples
///
/// ```
/// use cfglib::{Cfg, EdgeKind, EliminationHeuristic, tree_decomposition};
///
/// // A diamond: entry -> then | else -> merge.
/// let mut cfg = Cfg::<u32>::new();
/// let then_block = cfg.new_block();
/// let else_block = cfg.new_block();
/// let merge = cfg.new_block();
/// cfg.add_edge(cfg.entry(), then_block, EdgeKind::ConditionalTrue);
/// cfg.add_edge(cfg.entry(), else_block, EdgeKind::ConditionalFalse);
/// cfg.add_edge(then_block, merge, EdgeKind::Fallthrough);
/// cfg.add_edge(else_block, merge, EdgeKind::Fallthrough);
///
/// let decomposition = tree_decomposition(&cfg, EliminationHeuristic::MinDegree);
/// assert_eq!(decomposition.len(), 4);
/// // A cycle of four has treewidth 2.
/// assert_eq!(decomposition.width(), 2);
/// let bag = decomposition.bag_of(then_block).expect("every block has a bag");
/// assert!(decomposition.bag(bag).contains(&then_block));
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TreeDecomposition<N> {
    /// The nodes of every bag, bag after bag, the eliminated node first and
    /// its neighbors in ascending dense index.
    bags: Vec<N>,
    /// Where each bag starts in `bags`, with one more entry for the end.
    starts: Vec<usize>,
    /// The parent of each bag, [`NO_INDEX`] for a root.
    parents: Vec<u32>,
    /// The bag of each node by dense index, [`NO_INDEX`] for a slot the view
    /// does not hold.
    bag_of: Vec<u32>,
}

impl<N: DenseId> TreeDecomposition<N> {
    /// The number of bags, one per node.
    #[must_use]
    pub fn len(&self) -> usize {
        self.parents.len()
    }

    /// Whether there is no bag: the graph has no node.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.parents.is_empty()
    }

    /// The size of the largest bag less one, `0` without a bag.
    #[must_use]
    pub fn width(&self) -> usize {
        (0..self.len())
            .map(|bag| self.bag(bag).len())
            .max()
            .unwrap_or(1)
            - 1
    }

    /// The nodes of bag `bag`: the node eliminated there first, then its
    /// neighbors at that point in ascending dense index.
    ///
    /// # Panics
    ///
    /// Panics when `bag` is not below [`len`](Self::len).
    #[must_use]
    pub fn bag(&self, bag: usize) -> &[N] {
        &self.bags[self.starts[bag]..self.starts[bag + 1]]
    }

    /// The parent of bag `bag`, a later bag, or `None` for the root of a
    /// component.
    ///
    /// # Panics
    ///
    /// Panics when `bag` is not below [`len`](Self::len).
    #[must_use]
    pub fn parent(&self, bag: usize) -> Option<usize> {
        let parent = self.parents[bag];
        (parent != NO_INDEX).then_some(parent as usize)
    }

    /// The bag where `node` was eliminated, which holds it, or `None` for a
    /// node the graph does not hold.
    #[must_use]
    pub fn bag_of(&self, node: N) -> Option<usize> {
        match self.bag_of.get(node.index()).copied() {
            None | Some(NO_INDEX) => None,
            Some(bag) => Some(bag as usize),
        }
    }
}

/// A tree decomposition of the undirected simple graph underlying `graph`:
/// edge directions, parallel edges and self-loops are ignored.
///
/// See [`TreeDecomposition`] for the result and the [module
/// documentation](self) for the heuristics. The result depends only on the
/// set of edges, not on adjacency order or parallel edges.
///
/// Min-degree costs O(n + m + Σ d²) for n nodes, m edges, and bags of `d + 1`
/// nodes, with a logarithmic factor for its ordered queue: about O(n·w²) for
/// width w. Min-fill also recomputes the fill of the nodes next to each new
/// edge, which costs about another factor of w.
///
/// # Panics
///
/// Panics when the node bound exceeds the range of a dense `u32` table.
#[must_use]
pub fn tree_decomposition<G: GraphView>(
    graph: &G,
    heuristic: EliminationHeuristic,
) -> TreeDecomposition<G::NodeId> {
    let mut elimination = Elimination::new(graph, heuristic);
    elimination.run();
    elimination.finish()
}

/// The state of one elimination.
struct Elimination {
    heuristic: EliminationHeuristic,
    /// The neighbors each node has left, ascending, without itself.
    neighbors: Vec<Vec<u32>>,
    /// Whether the view holds each slot.
    live: Vec<bool>,
    /// The key each live, uneliminated node is queued under.
    key: Vec<u64>,
    /// The uneliminated nodes by key, then by dense index.
    queue: BTreeSet<(u64, u32)>,
    /// The position of each eliminated node in the order, [`NO_INDEX`] before.
    position: Vec<u32>,
    /// The bags in elimination order, flat.
    bags: Vec<u32>,
    starts: Vec<usize>,
    /// The slots one requeue has collected, to collect a set without sorting.
    collected: EpochMarks,
}

impl Elimination {
    fn new<G: GraphView>(graph: &G, heuristic: EliminationHeuristic) -> Self {
        let bound = graph.node_bound();
        check_bound(bound, "node bound of a tree decomposition");
        let mut live = vec![false; bound];
        let mut neighbors: Vec<Vec<u32>> = vec![Vec::new(); bound];
        for node in graph.node_ids() {
            let index = node.index();
            live[index] = true;
            for next in graph.successors(node) {
                let other = next.index();
                if other != index {
                    neighbors[index].push(word(other));
                    neighbors[other].push(word(index));
                }
            }
        }
        for list in &mut neighbors {
            list.sort_unstable();
            list.dedup();
        }
        let mut elimination = Self {
            heuristic,
            neighbors,
            live,
            key: vec![0; bound],
            queue: BTreeSet::new(),
            position: vec![NO_INDEX; bound],
            bags: Vec::new(),
            starts: vec![0],
            collected: EpochMarks::new(bound),
        };
        for node in 0..bound {
            if elimination.live[node] {
                let key = elimination.key_of(node);
                elimination.key[node] = key;
                elimination.queue.insert((key, word(node)));
            }
        }
        elimination
    }

    /// Eliminate every node.
    fn run(&mut self) {
        let mut step = 0;
        while let Some((_, node)) = self.queue.pop_first() {
            let node = slot(node);
            self.position[node] = word(step);
            step += 1;
            let around = core::mem::take(&mut self.neighbors[node]);
            self.bags.push(word(node));
            self.bags.extend_from_slice(&around);
            self.starts.push(self.bags.len());

            // The neighbors become a clique without `node`; note who gained an
            // edge, since the fill of their neighbors changes too.
            let mut gained = Vec::new();
            for &member in &around {
                let list = &mut self.neighbors[slot(member)];
                let before = list.len();
                let merged = merge_without(list, &around, member, word(node));
                if merged.len() + 1 > before {
                    gained.push(member);
                }
                *list = merged;
            }
            self.requeue(&around, &gained);
        }
    }

    /// Requeue the nodes whose key the last elimination changed: `around`,
    /// the neighbors of the eliminated node, and under min-fill the neighbors
    /// of the members that `gained` an edge.
    fn requeue(&mut self, around: &[u32], gained: &[u32]) {
        self.collected.reset();
        let mut changed: Vec<u32> = Vec::new();
        let mut mark = |member: u32, collected: &mut EpochMarks| {
            if !collected.is_marked(slot(member)) {
                collected.mark(slot(member));
                changed.push(member);
            }
        };
        for &member in around {
            mark(member, &mut self.collected);
        }
        if self.heuristic == EliminationHeuristic::MinFill {
            for &member in gained {
                for &next in &self.neighbors[slot(member)] {
                    mark(next, &mut self.collected);
                }
            }
        }
        for member in changed {
            let index = slot(member);
            if self.position[index] != NO_INDEX {
                continue;
            }
            let key = self.key_of(index);
            if key != self.key[index] {
                self.queue.remove(&(self.key[index], member));
                self.key[index] = key;
                self.queue.insert((key, member));
            }
        }
    }

    /// The key of `node` under the heuristic.
    fn key_of(&self, node: usize) -> u64 {
        let around = &self.neighbors[node];
        let degree = around.len() as u64;
        match self.heuristic {
            EliminationHeuristic::MinDegree => degree,
            EliminationHeuristic::MinFill => {
                // Each edge among the neighbors is seen from both its ends.
                let seen: u64 = around
                    .iter()
                    .map(|&member| common(&self.neighbors[slot(member)], around) as u64)
                    .sum();
                degree * degree.saturating_sub(1) / 2 - seen / 2
            }
        }
    }

    /// The finished decomposition: each bag's parent is the bag of the
    /// neighbor eliminated next.
    fn finish<N: DenseId>(self) -> TreeDecomposition<N> {
        let count = self.starts.len() - 1;
        let mut bag_of = vec![NO_INDEX; self.position.len()];
        for (node, &position) in self.position.iter().enumerate() {
            bag_of[node] = position;
        }
        let parents = (0..count)
            .map(|bag| {
                self.bags[self.starts[bag] + 1..self.starts[bag + 1]]
                    .iter()
                    .map(|&member| self.position[slot(member)])
                    .min()
                    .unwrap_or(NO_INDEX)
            })
            .collect();
        TreeDecomposition {
            bags: self
                .bags
                .iter()
                .map(|&node| N::from_index(slot(node)))
                .collect(),
            starts: self.starts,
            parents,
            bag_of,
        }
    }
}

/// `list ∪ clique`, ascending, without `own` and `gone`.
fn merge_without(list: &[u32], clique: &[u32], own: u32, gone: u32) -> Vec<u32> {
    let mut merged = Vec::with_capacity(list.len() + clique.len());
    let (mut left, mut right) = (list.iter().peekable(), clique.iter().peekable());
    loop {
        let next = match (left.peek(), right.peek()) {
            (Some(&&a), Some(&&b)) => {
                if a <= b {
                    left.next();
                    if a == b {
                        right.next();
                    }
                    a
                } else {
                    right.next();
                    b
                }
            }
            (Some(&&a), None) => {
                left.next();
                a
            }
            (None, Some(&&b)) => {
                right.next();
                b
            }
            (None, None) => return merged,
        };
        if next != own && next != gone {
            merged.push(next);
        }
    }
}

/// The number of members two ascending lists share.
fn common(first: &[u32], second: &[u32]) -> usize {
    let (mut i, mut j, mut shared) = (0, 0, 0);
    while i < first.len() && j < second.len() {
        match first[i].cmp(&second[j]) {
            core::cmp::Ordering::Less => i += 1,
            core::cmp::Ordering::Greater => j += 1,
            core::cmp::Ordering::Equal => {
                shared += 1;
                i += 1;
                j += 1;
            }
        }
    }
    shared
}
