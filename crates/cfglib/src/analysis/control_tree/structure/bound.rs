//! The blocks a walk may emit: the structural bound of the construct it is in.

extern crate alloc;
use alloc::collections::BTreeSet;
use alloc::rc::Rc;

use crate::block::BlockId;
use crate::graph::loop_forest::{LoopForest, LoopId};

/// The blocks a walk may emit: every block, the blocks of a set, or the blocks
/// of a loop that an outer bound also admits.
///
/// A loop is held by its place in the loop forest, not as a set of its blocks:
/// loops nested `n` deep share most of their blocks, so a set per loop is
/// quadratic in `n`. A bound is shared by every frame under it, so it is
/// reference counted and never changes.
#[derive(Clone)]
pub(super) enum Bound {
    /// No limit.
    Whole,
    /// The blocks of a set: a `try` body, a handler, or the unclaimed code.
    Set(Rc<BTreeSet<BlockId>>),
    /// The blocks of a loop that `outer` admits as well.
    Loop(Rc<WithinLoop>),
}

/// A loop and the bound it was entered under.
pub(super) struct WithinLoop {
    forest: Rc<LoopForest>,
    lp: LoopId,
    outer: Bound,
}

impl Bound {
    /// Whether the bound admits `block`.
    pub(super) fn contains(&self, block: BlockId) -> bool {
        let mut bound = self;
        loop {
            match bound {
                Self::Whole => return true,
                Self::Set(blocks) => return blocks.contains(&block),
                Self::Loop(within) => {
                    if !within.forest.contains(within.lp, block) {
                        return false;
                    }
                    bound = &within.outer;
                }
            }
        }
    }

    /// The blocks of `blocks` that the bound admits.
    pub(super) fn blocks_within(&self, blocks: &BTreeSet<BlockId>) -> BTreeSet<BlockId> {
        blocks
            .iter()
            .copied()
            .filter(|&block| self.contains(block))
            .collect()
    }

    /// The bound inside the loop `lp`: the blocks of the loop that this bound
    /// admits.
    ///
    /// A loop that lies inside the loop of this bound is all of its blocks
    /// that the outer bound admits already, so the outer loop is dropped and a
    /// bound over `n` nested loops stays one loop deep.
    pub(super) fn within_loop(&self, forest: &Rc<LoopForest>, lp: LoopId) -> Self {
        let outer = match self {
            Self::Loop(outer) if forest.encloses(outer.lp, lp) => outer.outer.clone(),
            other => other.clone(),
        };
        Self::Loop(Rc::new(WithinLoop {
            forest: Rc::clone(forest),
            lp,
            outer,
        }))
    }
}

impl Drop for WithinLoop {
    /// Release a long chain of bounds one link at a time: dropping each link
    /// inside the drop of the one above would recurse once per link.
    fn drop(&mut self) {
        let mut outer = core::mem::replace(&mut self.outer, Bound::Whole);
        while let Bound::Loop(within) = outer {
            match Rc::try_unwrap(within) {
                Ok(mut within) => outer = core::mem::replace(&mut within.outer, Bound::Whole),
                Err(_) => break,
            }
        }
    }
}
