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
//! which [`SccNumbering::LeavesFirst`] numbers leaves first: a block reaches only blocks of a
//! number at most its own, so a block whose number is below the lowest number
//! of the extent cannot reach it. A walk that skips them is the walk of the
//! blocks around the extent, and the visited table is shared by every walk,
//! told apart by a generation.
//!
//! The walk runs on the crate's depth-first core. A block below the floor is
//! entered but never expanded, and it is in no extent, so the extent's blocks
//! finish in the order they would if the walk never reached it.

extern crate alloc;
use alloc::collections::BTreeSet;
use alloc::vec::Vec;
use core::ops::ControlFlow;

use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::graph::epoch::EpochMarks;
use crate::graph::scc::{SccDecomposition, SccNumbering};
use crate::graph::search::Visit;
use crate::graph::traverse::{DepthFirstStack, DepthFirstVisitor, Outgoing, depth_first_core};
use crate::identity::{NO_INDEX, word};

/// The state every walk shares: the components of the graph, found by the
/// first walk, and the table of visited blocks.
#[derive(Default)]
pub(super) struct ExtentWalk {
    state: Option<State>,
}

struct State {
    /// The leaves-first number of the component of each block, [`NO_INDEX`]
    /// for a slot that holds no block.
    component: Vec<u32>,
    /// The blocks the current walk has visited.
    seen: EpochMarks,
    stack: DepthFirstStack<BlockId, BlockId>,
}

impl State {
    fn new<I, E>(cfg: &Cfg<I, E>) -> Self {
        let bound = cfg.block_bound();
        let mut component = alloc::vec![NO_INDEX; bound];
        for (number, blocks) in SccDecomposition::compute(cfg, SccNumbering::LeavesFirst)
            .components()
            .enumerate()
        {
            for block in blocks {
                component[block.index()] = word(number);
            }
        }
        Self {
            component,
            seen: EpochMarks::new(bound),
            stack: DepthFirstStack::default(),
        }
    }
}

/// The visitor of one walk: it expands only the blocks at or above the floor,
/// and records the extent's blocks as they finish.
struct ExtentPostorder<'a> {
    component: &'a [u32],
    /// The lowest component of the extent.
    floor: u32,
    extent: &'a BTreeSet<BlockId>,
    order: Vec<BlockId>,
}

impl DepthFirstVisitor<BlockId, BlockId, ()> for ExtentPostorder<'_> {
    #[inline]
    fn discover(&mut self, block: BlockId, _depth: usize) -> ControlFlow<(), Visit> {
        // Nothing below the floor reaches the extent.
        if self.component[block.index()] >= self.floor {
            ControlFlow::Continue(Visit::Descend)
        } else {
            ControlFlow::Continue(Visit::Skip)
        }
    }

    #[inline]
    fn finish(&mut self, block: BlockId) -> ControlFlow<()> {
        if self.extent.contains(&block) {
            self.order.push(block);
        }
        ControlFlow::Continue(())
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
        let floor = extent
            .iter()
            .filter_map(|block| state.component.get(block.index()).copied())
            .min()
            .unwrap_or(NO_INDEX);
        state.seen.reset();
        let mut visitor = ExtentPostorder {
            component: &state.component,
            floor,
            extent,
            order: Vec::new(),
        };
        depth_first_core(
            Outgoing,
            cfg,
            [entry],
            &mut state.seen,
            &mut state.stack,
            &mut visitor,
        );
        let mut order = visitor.order;
        order.reverse();
        order
    }
}
