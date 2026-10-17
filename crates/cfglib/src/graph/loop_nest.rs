//! Loop nesting tree — hierarchical representation of loop nesting.
//!
//! Builds a proper tree from the flat [`NaturalLoop`] list returned by
//! [`detect_loops`](crate::graph::structure::detect_loops), enabling
//! parent/child queries and nesting depth.
//!
//! The tree is also the one statement of nesting depth: detection fills
//! [`NaturalLoop::depth`] from the same parent rule, so a loop's depth in the
//! flat list is its depth in the tree.

extern crate alloc;
use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Reverse;

use crate::block::BlockId;
use crate::graph::structure::NaturalLoop;

/// Where each loop of a slice sits in the nest.
pub(crate) struct Nesting {
    /// The loop each loop is nested in, `None` for an outermost loop.
    pub(crate) parents: Vec<Option<usize>>,
    /// The number of loops each loop is nested in.
    pub(crate) depths: Vec<usize>,
}

/// Nest `loops`: the parent of a loop is the smallest loop with a strictly
/// larger body that contains its header, the first listed among equals.
///
/// Requiring a strictly larger body keeps every parent chain finite. On a
/// reducible graph the bodies of two loops are nested or disjoint, so a loop
/// whose header lies in a larger body lies inside it. The loops of an
/// irreducible region can contain each other's headers, and two of them with
/// the same body are siblings rather than each other's parent.
///
/// Each body node is looked up once among the headers, so the cost is the
/// total size of the bodies times a logarithm, however deep the nest.
pub(crate) fn nesting<N: Copy + Ord>(loops: &[NaturalLoop<N>]) -> Nesting {
    let mut by_header: Vec<(N, usize)> = loops
        .iter()
        .enumerate()
        .map(|(index, lp)| (lp.header, index))
        .collect();
    by_header.sort_unstable();

    let mut parents: Vec<Option<usize>> = vec![None; loops.len()];
    for (outer_index, outer) in loops.iter().enumerate() {
        let outer_size = outer.body.len();
        for &node in &outer.body {
            let first = by_header.partition_point(|&(header, _)| header < node);
            for &(header, inner_index) in &by_header[first..] {
                if header != node {
                    break;
                }
                let tighter = parents[inner_index]
                    .is_none_or(|current| outer_size < loops[current].body.len());
                if outer_size > loops[inner_index].body.len() && tighter {
                    parents[inner_index] = Some(outer_index);
                }
            }
        }
    }

    // A parent's body is strictly larger than its child's, so largest first
    // numbers every parent before its children.
    let mut order: Vec<usize> = (0..loops.len()).collect();
    order.sort_unstable_by_key(|&index| Reverse(loops[index].body.len()));
    let mut depths = vec![0; loops.len()];
    for index in order {
        depths[index] = parents[index].map_or(0, |parent| depths[parent] + 1);
    }
    Nesting { parents, depths }
}

/// A node in the loop nesting tree, over node identity `N`.
#[derive(Debug, Clone)]
pub struct LoopNestNode<N = BlockId> {
    /// Index into the original `NaturalLoop` slice.
    pub loop_index: usize,
    /// Header node of this loop.
    pub header: N,
    /// Parent loop index (None for outermost loops).
    pub parent: Option<usize>,
    /// Indices of directly nested child loops.
    pub children: Vec<usize>,
    /// Nesting depth (0 for outermost).
    pub depth: usize,
}

/// A loop nesting tree built from detected natural loops.
///
/// # Examples
///
/// ```
/// use cfglib::{Cfg, EdgeKind, DominatorTree, detect_loops, LoopNestingTree};
///
/// let mut cfg = Cfg::<u32>::new();
/// let b0 = cfg.entry();
/// let b1 = cfg.new_block();
/// let b2 = cfg.new_block();
/// cfg.add_edge(b0, b1, EdgeKind::Fallthrough);
/// cfg.add_edge(b1, b0, EdgeKind::Back);
/// cfg.add_edge(b0, b2, EdgeKind::ConditionalTrue);
///
/// let dom = DominatorTree::compute(&cfg);
/// let loops = detect_loops(&cfg, &dom);
/// let tree = LoopNestingTree::compute(&loops);
/// assert_eq!(tree.len(), 1);
/// assert_eq!(tree.innermost_loop(b1), Some(0));
/// ```
#[derive(Debug, Clone)]
pub struct LoopNestingTree<N = BlockId> {
    /// One node per natural loop, indexed by loop index.
    pub nodes: Vec<LoopNestNode<N>>,
    /// Node → innermost containing loop index.
    block_to_loop: BTreeMap<N, usize>,
}

impl<N: Copy + Ord> LoopNestingTree<N> {
    /// Compute the nesting tree from a slice of natural loops.
    ///
    /// A loop's parent is the smallest loop with a strictly larger body that
    /// contains its header. Loops whose bodies are the same, which an
    /// irreducible region can produce, are siblings. The order of `loops`
    /// does not matter, and each node keeps the index of its loop.
    #[must_use]
    pub fn compute(loops: &[NaturalLoop<N>]) -> Self {
        let Nesting { parents, depths } = nesting(loops);
        let mut nodes: Vec<LoopNestNode<N>> = loops
            .iter()
            .enumerate()
            .map(|(i, lp)| LoopNestNode {
                loop_index: i,
                header: lp.header,
                parent: parents[i],
                children: Vec::new(),
                depth: depths[i],
            })
            .collect();
        for (child, parent) in parents.iter().enumerate() {
            if let Some(parent) = *parent {
                nodes[parent].children.push(child);
            }
        }

        // Map blocks to innermost loop.
        let mut block_to_loop = BTreeMap::new();
        for (i, lp) in loops.iter().enumerate() {
            for &bid in &lp.body {
                let entry = block_to_loop.entry(bid).or_insert(i);
                // Prefer deeper (more nested) loop.
                if nodes[i].depth > nodes[*entry].depth {
                    *entry = i;
                }
            }
        }

        Self {
            nodes,
            block_to_loop,
        }
    }

    /// Get the innermost loop containing a node, if any.
    #[must_use]
    pub fn innermost_loop(&self, block: N) -> Option<usize> {
        self.block_to_loop.get(&block).copied()
    }

    /// Get outermost (root) loops — those with no parent.
    #[must_use]
    pub fn roots(&self) -> Vec<usize> {
        self.nodes
            .iter()
            .filter(|n| n.parent.is_none())
            .map(|n| n.loop_index)
            .collect()
    }

    /// Nesting depth of a loop (0 for outermost).
    #[must_use]
    pub fn depth(&self, loop_index: usize) -> usize {
        self.nodes[loop_index].depth
    }

    /// Number of loops.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the tree is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::collections::BTreeSet;

    /// A loop with the given header and body. The `depth` field is a
    /// product of detection, which `compute` must not read, so it is a
    /// value no nesting produces.
    fn make_loop(header: u32, body: &[u32]) -> NaturalLoop {
        NaturalLoop {
            header: BlockId::from_raw(header),
            body: body
                .iter()
                .map(|&b| BlockId::from_raw(b))
                .collect::<BTreeSet<_>>(),
            latches: BTreeSet::new(),
            depth: 99,
        }
    }

    #[test]
    fn nested_loops() {
        let outer = make_loop(1, &[1, 2, 3]);
        let inner = make_loop(2, &[2, 3]);
        let tree = LoopNestingTree::compute(&[outer, inner]);
        assert_eq!(tree.len(), 2);
        assert_eq!(tree.depth(0), 0);
        assert_eq!(tree.depth(1), 1);
        assert_eq!(tree.nodes[1].parent, Some(0));
        assert!(tree.nodes[0].children.contains(&1));
    }

    #[test]
    fn the_order_of_the_loops_does_not_matter() {
        let outer = make_loop(1, &[1, 2, 3]);
        let inner = make_loop(2, &[2, 3]);
        let tree = LoopNestingTree::compute(&[inner, outer]);
        assert_eq!(tree.nodes[0].parent, Some(1));
        assert_eq!(tree.depth(0), 1);
        assert_eq!(tree.depth(1), 0);
        assert_eq!(tree.roots(), [1]);
    }

    #[test]
    fn the_smallest_enclosing_loop_is_the_parent() {
        let outer = make_loop(1, &[1, 2, 3, 4]);
        let middle = make_loop(2, &[2, 3, 4]);
        let inner = make_loop(3, &[3, 4]);
        let tree = LoopNestingTree::compute(&[outer, middle, inner]);
        assert_eq!(
            tree.nodes
                .iter()
                .map(|node| node.parent)
                .collect::<Vec<_>>(),
            [None, Some(0), Some(1)]
        );
        assert_eq!(tree.nodes[0].children, [1]);
        assert_eq!(tree.depth(2), 2);
    }

    #[test]
    fn innermost_loop_query() {
        let outer = make_loop(1, &[1, 2, 3]);
        let inner = make_loop(2, &[2, 3]);
        let tree = LoopNestingTree::compute(&[outer, inner]);
        // Block 3 is in both loops, innermost is index 1.
        assert_eq!(tree.innermost_loop(BlockId::from_raw(3)), Some(1));
        // Block 1 is only in outer.
        assert_eq!(tree.innermost_loop(BlockId::from_raw(1)), Some(0));
    }

    #[test]
    fn loops_that_contain_each_others_headers_do_not_nest_in_a_cycle() {
        // An irreducible region: each header lies in the other loop's body.
        // The bigger loop is the parent, never the other way round.
        let big = make_loop(1, &[1, 2, 3]);
        let small = make_loop(2, &[1, 2]);
        let tree = LoopNestingTree::compute(&[big, small]);
        assert_eq!(tree.nodes[0].parent, None);
        assert_eq!(tree.nodes[1].parent, Some(0));
        assert_eq!((tree.depth(0), tree.depth(1)), (0, 1));
    }

    #[test]
    fn loops_with_the_same_body_are_siblings() {
        let first = make_loop(1, &[1, 2]);
        let second = make_loop(2, &[1, 2]);
        let tree = LoopNestingTree::compute(&[first, second]);
        assert_eq!(tree.roots(), [0, 1]);
        assert_eq!((tree.depth(0), tree.depth(1)), (0, 0));
        assert!(tree.nodes.iter().all(|node| node.children.is_empty()));
    }

    #[test]
    fn equally_small_enclosing_loops_pick_the_first_listed() {
        let first = make_loop(1, &[1, 2, 3]);
        let second = make_loop(2, &[1, 2, 3]);
        let inner = make_loop(3, &[3]);
        let tree = LoopNestingTree::compute(&[first, second, inner]);
        assert_eq!(tree.nodes[2].parent, Some(0));
    }
}
