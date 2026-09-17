//! Basic block — a contiguous sequence of instructions with a single
//! entry point and a single exit point.

extern crate alloc;
use alloc::string::String;
use alloc::vec::Vec;

use crate::graph::store::{Id, IdTag};

/// Tag marking an identity that addresses a basic block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BlockTag;

impl IdTag for BlockTag {
    const PREFIX: &'static str = "bb";
}

/// Opaque identifier for a basic block within a [`Cfg`](crate::Cfg).
///
/// A block identity is its slot in the CFG's store. Slots are never reused,
/// so an identity stays valid until [`Cfg::compact`](crate::Cfg::compact)
/// renumbers the graph and reports the change as a
/// [`Renumbering`](crate::Renumbering).
pub type BlockId = Id<BlockTag>;

/// A basic block containing a linear sequence of instructions.
///
/// The block does not carry its own identity: that is the slot the CFG minted
/// it in, which [`Cfg::block_ids`](crate::Cfg::block_ids) yields and
/// [`Cfg::block`](crate::Cfg::block) resolves.
///
/// Predication (ARM IT blocks, GPU wave predication, CMOV sequences) is not
/// block state: instructions declare their guards through
/// [`Predicated`](crate::Predicated), and
/// [`lift_predicated`](crate::lift_predicated) regionizes them.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BasicBlock<I> {
    /// Instructions in program order.
    pub(crate) instructions: Vec<I>,
    /// Optional human-readable label (e.g. from a `label` instruction).
    pub(crate) label: Option<String>,
}

impl<I> BasicBlock<I> {
    /// Create an empty block.
    pub(crate) const fn new() -> Self {
        Self {
            instructions: Vec::new(),
            label: None,
        }
    }

    /// The instructions inside this block.
    #[inline]
    #[must_use]
    pub fn instructions(&self) -> &[I] {
        &self.instructions
    }

    /// Mutable access to the instruction vector.
    ///
    /// Blocks impose no invariants on their instruction list, so full `Vec`
    /// control (insert, remove, drain) is available directly.
    #[inline]
    pub fn instructions_mut(&mut self) -> &mut Vec<I> {
        &mut self.instructions
    }

    /// Optional label for this block.
    #[inline]
    #[must_use]
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    /// Returns `true` if the block contains no instructions.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.instructions.is_empty()
    }

    /// Append an instruction to the end of the block.
    #[inline]
    pub fn push(&mut self, inst: I) {
        self.instructions.push(inst);
    }

    /// Set or replace the block's human-readable label.
    #[inline]
    pub fn set_label(&mut self, label: impl Into<String>) {
        self.label = Some(label.into());
    }
}

impl<I> Default for BasicBlock<I> {
    fn default() -> Self {
        Self::new()
    }
}
