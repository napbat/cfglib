//! Traversals of a [`Cfg`] from its entry block.

extern crate alloc;
use alloc::vec::Vec;

use crate::block::BlockId;
use crate::graph::traverse::{
    breadth_first, depth_first_postorder, depth_first_preorder, reverse_postorder,
};
use crate::graph::view::Direction;

use super::Cfg;

impl<I, E> Cfg<I, E> {
    /// Depth-first preorder traversal starting from the entry block.
    #[must_use]
    pub fn depth_first_preorder(&self) -> Vec<BlockId> {
        depth_first_preorder(self, self.entry(), Direction::Forward)
    }

    /// Depth-first postorder traversal starting from the entry block.
    #[must_use]
    pub fn depth_first_postorder(&self) -> Vec<BlockId> {
        depth_first_postorder(self, self.entry(), Direction::Forward)
    }

    /// Breadth-first traversal starting from the entry block.
    #[must_use]
    pub fn breadth_first(&self) -> Vec<BlockId> {
        breadth_first(self, self.entry(), Direction::Forward)
    }

    /// Reverse postorder starting from the entry block.
    #[must_use]
    pub fn reverse_postorder(&self) -> Vec<BlockId> {
        reverse_postorder(self, self.entry(), Direction::Forward)
    }
}

#[cfg(test)]
mod tests {
    extern crate alloc;
    use alloc::vec;

    use crate::cfg::Cfg;
    use crate::edge::EdgeKind;
    use crate::test_util::ff;

    #[test]
    fn traversal_methods_walk_forward_from_the_entry() {
        let mut cfg = Cfg::new();
        let middle = cfg.new_block();
        let last = cfg.new_block();
        cfg.block_mut(cfg.entry()).push(ff("entry"));
        cfg.block_mut(middle).push(ff("middle"));
        cfg.block_mut(last).push(ff("last"));
        cfg.add_edge(cfg.entry(), middle, EdgeKind::Fallthrough);
        cfg.add_edge(middle, last, EdgeKind::Fallthrough);

        assert_eq!(cfg.depth_first_preorder(), vec![cfg.entry(), middle, last]);
        assert_eq!(cfg.depth_first_postorder(), vec![last, middle, cfg.entry()]);
        assert_eq!(cfg.reverse_postorder(), vec![cfg.entry(), middle, last]);
        assert_eq!(cfg.breadth_first(), vec![cfg.entry(), middle, last]);
    }
}
