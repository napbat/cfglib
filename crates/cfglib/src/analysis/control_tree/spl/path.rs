//! A path decomposition of bounded width, read off the decomposition.
//!
//! The decomposition is walked in program order with a stack of blocks that
//! every bag of the current fragment holds:
//!
//! - inside a series, the entry block of the next statement, which every
//!   edge falling out of the current one ends at;
//! - inside a parallel, its branch block, which every arm's first edge
//!   leaves;
//! - inside a loop, its header, which the body's back edges and continues
//!   end at. Its breaks end at the next statement after the loop, which the
//!   enclosing series holds already.
//!
//! A leaf's bag is the stack and its own block. Every edge then has both ends
//! in a bag, and a block is in the bags of the statement before it and of
//! its own fragment, which are consecutive. The stack grows by one for a
//! series and one for a parallel or loop, and a series directly inside a
//! series is flattened, so a leaf under `d` parallels and loops has a bag of
//! at most `2·d + 2` blocks.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use super::{SplDecomposition, SplNode, SplNodeId};
use crate::block::BlockId;

/// A path decomposition: bags of blocks in order.
///
/// Every edge of the graph has both ends in one bag, and the bags that hold a
/// block are consecutive. The bags are one flat table of runs.
/// [`SplDecomposition::path_decomposition`] returns one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PathDecomposition {
    /// The blocks of every bag, bag after bag.
    blocks: Vec<BlockId>,
    /// Where each bag starts in `blocks`, with one more entry for the end.
    /// The bags of a deep nest hold far more blocks than the graph, so the
    /// offsets are not narrowed.
    starts: Vec<usize>,
}

impl PathDecomposition {
    /// The number of bags.
    #[must_use]
    pub fn bag_count(&self) -> usize {
        self.starts.len() - 1
    }

    /// Whether there is no bag.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bag_count() == 0
    }

    /// The blocks of bag `index`.
    ///
    /// # Panics
    ///
    /// Panics when `index` is not below [`bag_count`](Self::bag_count).
    #[must_use]
    pub fn bag(&self, index: usize) -> &[BlockId] {
        &self.blocks[self.starts[index]..self.starts[index + 1]]
    }

    /// Every bag, in order.
    pub fn bags(&self) -> impl ExactSizeIterator<Item = &[BlockId]> + '_ {
        (0..self.bag_count()).map(|index| self.bag(index))
    }

    /// The size of the largest bag, less one; `0` without a bag.
    #[must_use]
    pub fn width(&self) -> usize {
        self.bags()
            .map(<[BlockId]>::len)
            .max()
            .unwrap_or(1)
            .saturating_sub(1)
    }
}

/// A node still to walk, or a change to the stack of held blocks.
#[derive(Clone, Copy)]
enum Step {
    Walk(SplNodeId),
    Hold(BlockId),
    Pop,
}

/// The path decomposition of `decomposition`.
pub(super) fn decompose(decomposition: &SplDecomposition) -> PathDecomposition {
    let bound = decomposition
        .blocks
        .iter()
        .map(|block| block.index() + 1)
        .max()
        .unwrap_or(0);
    let mut held = vec![false; bound];
    // The blocks every bag holds, each with whether this entry added it.
    let mut stack: Vec<(BlockId, bool)> = Vec::new();
    let mut path = PathDecomposition {
        blocks: Vec::new(),
        starts: vec![0],
    };
    let mut work = vec![Step::Walk(decomposition.root)];
    while let Some(step) = work.pop() {
        let id = match step {
            Step::Walk(id) => id,
            Step::Hold(block) => {
                hold(&mut stack, &mut held, block);
                continue;
            }
            Step::Pop => {
                let (block, added) = stack.pop().expect("a pop follows its push");
                if added {
                    held[block.index()] = false;
                }
                continue;
            }
        };
        match *decomposition.node(id) {
            SplNode::Block(block) => {
                let own = (!held[block.index()]).then_some(block);
                path.push_bag(&stack, own);
            }
            SplNode::Series { children } => {
                // Each statement but the last holds the entry of the next one.
                let children = decomposition.children(children);
                for (position, &child) in children.iter().enumerate().rev() {
                    let next = children.get(position + 1);
                    match next.and_then(|&next| entry(decomposition, next)) {
                        Some(next) => {
                            work.push(Step::Pop);
                            work.push(Step::Walk(child));
                            work.push(Step::Hold(next));
                        }
                        None => work.push(Step::Walk(child)),
                    }
                }
            }
            SplNode::Parallel { branch, arms } => {
                // The branch with the next statement, for the arms that fall
                // straight through.
                hold(&mut stack, &mut held, branch);
                path.push_bag(&stack, None);
                work.push(Step::Pop);
                work.extend(
                    decomposition
                        .children(arms)
                        .iter()
                        .rev()
                        .map(|&arm| Step::Walk(arm)),
                );
            }
            SplNode::Loop { header, body } => {
                work.push(Step::Pop);
                work.push(Step::Walk(body));
                hold(&mut stack, &mut held, header);
            }
            SplNode::Break | SplNode::Continue => {}
        }
    }
    path
}

/// Put `block` on the stack of held blocks.
fn hold(stack: &mut Vec<(BlockId, bool)>, held: &mut [bool], block: BlockId) {
    let added = !held[block.index()];
    held[block.index()] = true;
    stack.push((block, added));
}

impl PathDecomposition {
    /// Add a bag of the held blocks and `own`.
    fn push_bag(&mut self, stack: &[(BlockId, bool)], own: Option<BlockId>) {
        let held = stack.iter().filter(|entry| entry.1).map(|entry| entry.0);
        self.blocks.extend(held.chain(own));
        self.starts.push(self.blocks.len());
    }
}

/// The block a node is entered at, `None` for one that only passes edges on.
fn entry(decomposition: &SplDecomposition, mut id: SplNodeId) -> Option<BlockId> {
    loop {
        match *decomposition.node(id) {
            SplNode::Block(block) => return Some(block),
            SplNode::Parallel { branch, .. } => return Some(branch),
            SplNode::Loop { header, .. } => return Some(header),
            SplNode::Series { children } => id = *decomposition.children(children).first()?,
            SplNode::Break | SplNode::Continue => return None,
        }
    }
}
