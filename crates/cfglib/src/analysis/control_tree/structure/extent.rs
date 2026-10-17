//! The order in which a handler's extent is laid out.
//!
//! A handler's blocks are listed in flow order: the blocks of its known extent
//! in the reverse postorder of a walk from its entry. The walk reaches
//! everything the entry reaches, which for a pad that returns to the code
//! after its `try` is the rest of the function, and a function with `r`
//! handlers of `n` blocks then costs `r * n`, in time and in the zeroed
//! visited table of each walk.
//!
//! Only the order of the extent's blocks is read, and a block that cannot
//! reach any of them has no effect on it: the walk marks only blocks that it
//! cannot reach any extent block from, and no block that can reach one
//! looks at those marks. So the walk skips every block that no extent block
//! can be reached from. That is read off the strongly connected components,
//! which [`tarjan_scc`] numbers leaves first: a block reaches only blocks of a
//! number at most its own, so a block whose number is below the lowest number
//! of the extent cannot reach it. A walk that skips them is the walk of the
//! blocks around the extent, and the visited table is shared by every walk,
//! told apart by a generation.

extern crate alloc;
use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::graph::scc::tarjan_scc;

/// The state every walk shares: the components of the graph, found by the
/// first walk, and the table of visited blocks.
#[derive(Default)]
pub(super) struct ExtentWalk {
    state: Option<State>,
}

struct State {
    /// The leaves-first number of the component of each block, `usize::MAX`
    /// for a slot that holds no block.
    component: Vec<usize>,
    /// The walk that last visited each block.
    seen: Vec<u32>,
    walk: u32,
    stack: Vec<(BlockId, bool)>,
    successors: Vec<BlockId>,
}

impl State {
    fn new<I, E>(cfg: &Cfg<I, E>) -> Self {
        let bound = cfg.block_bound();
        let mut component = alloc::vec![usize::MAX; bound];
        for (number, blocks) in tarjan_scc(cfg).components().enumerate() {
            for block in blocks {
                component[block.index()] = number;
            }
        }
        Self {
            component,
            seen: alloc::vec![0; bound],
            walk: 0,
            stack: Vec::new(),
            successors: Vec::new(),
        }
    }
}

impl ExtentWalk {
    /// The blocks of `extent` in the reverse postorder of a walk from
    /// `entry`: the order of those the walk reaches, which is that of the walk
    /// over every block, since a block that cannot reach `extent` changes
    /// nothing.
    ///
    /// # Panics
    ///
    /// Panics when `entry` is beyond the block bound of `cfg`.
    pub(super) fn reverse_postorder_of<I, E>(
        &mut self,
        cfg: &Cfg<I, E>,
        entry: BlockId,
        extent: &BTreeSet<BlockId>,
    ) -> Vec<BlockId> {
        assert!(
            entry.index() < cfg.block_bound(),
            "start node {} is beyond the block bound",
            entry.index()
        );
        let state = self.state.get_or_insert_with(|| State::new(cfg));
        // Nothing below the lowest component of the extent reaches it.
        let floor = extent
            .iter()
            .filter_map(|block| state.component.get(block.index()).copied())
            .min()
            .unwrap_or(usize::MAX);
        state.walk += 1;
        let walk = state.walk;
        let mut order = Vec::new();
        state.stack.clear();
        state.stack.push((entry, false));
        // The flag is whether the block's successors have been pushed, so the
        // second pop is its place in the postorder.
        while let Some((block, processed)) = state.stack.pop() {
            if processed {
                if extent.contains(&block) {
                    order.push(block);
                }
                continue;
            }
            if state.seen[block.index()] == walk {
                continue;
            }
            state.seen[block.index()] = walk;
            state.stack.push((block, true));
            state.successors.clear();
            state.successors.extend(cfg.successors(block));
            for &successor in state.successors.iter().rev() {
                if state.seen[successor.index()] != walk
                    && state.component[successor.index()] >= floor
                {
                    state.stack.push((successor, false));
                }
            }
        }
        order.reverse();
        order
    }
}
