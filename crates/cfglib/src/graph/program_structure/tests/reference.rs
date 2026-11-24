//! What the regions of a small graph are by definition, found by brute force.
//!
//! The reference builds the control flow graph of the paper's Definition 1
//! out of the closed graph: a start node with the virtual entry edge into the
//! root, the reachable blocks, and an end node that every exit block and every
//! cut root has a virtual edge to. Cycle equivalence comes from enumerating the
//! simple cycles of that graph plus an edge from end to start. Dominance and
//! post-dominance of edges come from removing one edge and searching again.
//! The canonical regions then follow Definitions 3 and 5, and their blocks
//! Definition 6 restricted to the blocks between the two edges, with no
//! cleverness, so a disagreement convicts the implementation.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use super::super::SeseEdge;
use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::edge::EdgeId;
use crate::graph::dominator::PostDominatorTree;

/// A boundary of a region, as the tree reports it.
pub(super) type Boundary = SeseEdge<BlockId, EdgeId>;

/// The definitions evaluated on one graph.
pub(super) struct Reference {
    /// The edges of the control flow graph, with the boundary each one is.
    /// Node `blocks` is the start and node `blocks + 1` the end.
    edges: Vec<(usize, usize, Boundary)>,
    /// The node bound of the graph.
    blocks: usize,
    /// For each edge, the simple cycles of the closed graph through it, as a
    /// bitset over the enumerated cycles.
    cycles: Vec<Vec<u64>>,
    /// For each edge, the nodes the start reaches without it.
    reach_without: Vec<u64>,
    /// For each edge, the nodes that reach the end without it.
    coreach_without: Vec<u64>,
    /// The nodes the start reaches.
    reachable: u64,
}

impl Reference {
    /// Evaluate the definitions on `cfg`.
    pub(super) fn new<I>(cfg: &Cfg<I>) -> Self {
        let blocks = cfg.block_bound();
        let (start, end) = (blocks, blocks + 1);
        assert!(blocks + 2 <= 64, "the reference handles small graphs");
        let mut edges = vec![(start, cfg.entry().index(), SeseEdge::Entry)];
        let reachable_blocks = reach(blocks + 2, &edges_of(cfg), cfg.entry().index(), None);
        for edge in cfg.edges() {
            if reachable_blocks & bit(edge.source().index()) != 0 {
                edges.push((
                    edge.source().index(),
                    edge.target().index(),
                    SeseEdge::Edge(edge.id()),
                ));
            }
        }
        let post = PostDominatorTree::compute(cfg);
        for &block in post.exits().iter().chain(post.cut_roots()) {
            edges.push((block.index(), end, SeseEdge::Exit(block)));
        }
        let count = edges.len();
        let plain: Vec<(usize, usize)> = edges.iter().map(|&(s, t, _)| (s, t)).collect();
        let reach_without = (0..count)
            .map(|skip| reach(blocks + 2, &plain, start, Some(skip)))
            .collect();
        let reversed: Vec<(usize, usize)> = plain.iter().map(|&(s, t)| (t, s)).collect();
        let coreach_without = (0..count)
            .map(|skip| reach(blocks + 2, &reversed, end, Some(skip)))
            .collect();
        let reachable = reach(blocks + 2, &plain, start, None);

        let mut closed = plain;
        closed.push((end, start));
        let found = simple_cycles(blocks + 2, &closed);
        let words = found.len().div_ceil(64).max(1);
        let mut cycles = vec![vec![0_u64; words]; count];
        for (index, &cycle) in found.iter().enumerate() {
            for (edge, membership) in cycles.iter_mut().enumerate() {
                if cycle & bit(edge) != 0 {
                    membership[index / 64] |= bit(index % 64);
                }
            }
        }
        Self {
            edges,
            blocks,
            cycles,
            reach_without,
            coreach_without,
            reachable,
        }
    }

    /// Every edge of the control flow graph, as a boundary.
    pub(super) fn boundaries(&self) -> impl Iterator<Item = Boundary> + '_ {
        self.edges.iter().map(|&(_, _, boundary)| boundary)
    }

    /// Whether the start reaches `block`.
    pub(super) fn reaches(&self, block: usize) -> bool {
        self.reachable & bit(block) != 0
    }

    /// Whether every cycle through edge `a` passes through edge `b` and the
    /// other way around.
    pub(super) fn equivalent(&self, a: usize, b: usize) -> bool {
        self.cycles[a] == self.cycles[b]
    }

    /// Whether every path from the start through edge `b` passes through `a`.
    fn edge_dominates(&self, a: usize, b: usize) -> bool {
        self.reach_without[a] & bit(self.edges[b].0) == 0
    }

    /// Whether every path from edge `a` to the end passes through `b`.
    fn edge_post_dominates(&self, b: usize, a: usize) -> bool {
        self.coreach_without[b] & bit(self.edges[a].1) == 0
    }

    /// Definition 3: whether `(a, b)` is a single-entry single-exit region.
    fn is_region(&self, a: usize, b: usize) -> bool {
        a != b
            && self.equivalent(a, b)
            && self.edge_dominates(a, b)
            && self.edge_post_dominates(b, a)
    }

    /// Definition 5: the canonical regions, as edge pairs, each with its
    /// blocks: those Definition 6 contains that control reaches from the
    /// entry edge without crossing the exit edge.
    ///
    /// Definition 6 alone also counts a block that only a path out of the
    /// exit edge and around a loop back to the entry edge reaches. Such a block
    /// is entered from outside the region, so it is not between the two edges.
    pub(super) fn canonical_regions(&self) -> Vec<(Boundary, Boundary, Vec<usize>)> {
        let count = self.edges.len();
        let plain: Vec<(usize, usize)> = self.edges.iter().map(|&(s, t, _)| (s, t)).collect();
        let mut found = Vec::new();
        for a in 0..count {
            for b in 0..count {
                if !self.is_region(a, b) {
                    continue;
                }
                let first_exit = (0..count)
                    .all(|c| c == b || !self.is_region(a, c) || self.edge_dominates(b, c));
                let last_entry = (0..count)
                    .all(|c| c == a || !self.is_region(c, b) || self.edge_post_dominates(a, c));
                if first_exit && last_entry {
                    let between = reach(self.blocks + 2, &plain, self.edges[a].1, Some(b));
                    let blocks = (0..self.blocks)
                        .filter(|&block| {
                            between & bit(block) != 0
                                && self.reach_without[a] & bit(block) == 0
                                && self.coreach_without[b] & bit(block) == 0
                        })
                        .collect();
                    found.push((self.edges[a].2, self.edges[b].2, blocks));
                }
            }
        }
        found
    }

    /// The edges of the control flow graph that enter `blocks` from outside
    /// it, and those that leave it.
    pub(super) fn crossings(&self, blocks: &[usize]) -> (Vec<Boundary>, Vec<Boundary>) {
        let inside = |node: usize| blocks.contains(&node);
        let entering = self
            .edges
            .iter()
            .filter(|&&(source, target, _)| !inside(source) && inside(target))
            .map(|&(_, _, boundary)| boundary)
            .collect();
        let leaving = self
            .edges
            .iter()
            .filter(|&&(source, target, _)| inside(source) && !inside(target))
            .map(|&(_, _, boundary)| boundary)
            .collect();
        (entering, leaving)
    }
}

/// The edges of `cfg` as endpoint pairs.
fn edges_of<I>(cfg: &Cfg<I>) -> Vec<(usize, usize)> {
    cfg.edges()
        .map(|edge| (edge.source().index(), edge.target().index()))
        .collect()
}

/// The single bit for `index`.
const fn bit(index: usize) -> u64 {
    1 << index
}

/// The nodes `from` reaches over `edges`, leaving out edge `skip`.
fn reach(nodes: usize, edges: &[(usize, usize)], from: usize, skip: Option<usize>) -> u64 {
    let mut seen = bit(from);
    let mut stack = vec![from];
    while let Some(node) = stack.pop() {
        for (index, &(source, target)) in edges.iter().enumerate() {
            if source == node && Some(index) != skip && seen & bit(target) == 0 {
                debug_assert!(target < nodes);
                seen |= bit(target);
                stack.push(target);
            }
        }
    }
    seen
}

/// Every simple cycle over `edges`, as a bitset of edge indices.
///
/// A cycle is found once, from its smallest node, by extending simple paths
/// through larger nodes only.
fn simple_cycles(nodes: usize, edges: &[(usize, usize)]) -> Vec<u64> {
    assert!(edges.len() <= 64, "the reference handles small graphs");
    let mut cycles = Vec::new();
    for start in 0..nodes {
        // (node, next edge to try, edges on the path, nodes on the path)
        let mut stack = vec![(start, 0_usize, 0_u64, bit(start))];
        while let Some(frame) = stack.last_mut() {
            let (node, next, path, on_path) = *frame;
            let Some(offset) = edges[next..].iter().position(|&(source, _)| source == node) else {
                stack.pop();
                continue;
            };
            let edge = next + offset;
            frame.1 = edge + 1;
            let target = edges[edge].1;
            if target == start {
                cycles.push(path | bit(edge));
            } else if target > start && on_path & bit(target) == 0 {
                stack.push((target, 0, path | bit(edge), on_path | bit(target)));
            }
        }
    }
    cycles
}
