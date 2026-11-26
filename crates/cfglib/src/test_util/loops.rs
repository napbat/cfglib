//! Loops as owned sets of blocks, the shape the loop tests compare against,
//! and the nesting rule stated over that shape.

extern crate alloc;
use alloc::collections::BTreeSet;
use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Reverse;

use crate::block::BlockId;
use crate::graph::loop_forest::LoopForest;
use crate::graph::view::DenseId;

/// One loop with an owned set of its blocks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct NaturalLoop<N = BlockId> {
    /// The loop header.
    pub(crate) header: N,
    /// Every block of the loop, its header included.
    pub(crate) body: BTreeSet<N>,
    /// The tails of the back edges into the header.
    pub(crate) latches: BTreeSet<N>,
    /// The number of loops this one is nested in.
    pub(crate) depth: usize,
}

/// The loops of `forest` as owned sets, outermost first and by header within
/// a depth.
pub(crate) fn natural_loops<N: DenseId>(forest: &LoopForest<N>) -> Vec<NaturalLoop<N>> {
    let mut loops: Vec<NaturalLoop<N>> = forest
        .loops()
        .map(|lp| NaturalLoop {
            header: forest.header(lp),
            body: forest.body(lp).collect(),
            latches: forest.latches(lp).iter().copied().collect(),
            depth: forest.depth(lp),
        })
        .collect();
    loops.sort_by(|a, b| a.depth.cmp(&b.depth).then(a.header.cmp(&b.header)));
    loops
}

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
