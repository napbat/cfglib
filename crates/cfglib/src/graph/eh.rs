//! Exception handling (EH) modeling.
//!
//! [`EhModel`] classifies every block by its role in exception handling —
//! landing pad, cleanup, catch dispatch, resume point — and lists the
//! exception-control edges. It derives only what the [`Cfg`](crate::Cfg)
//! does not store: the handlers of a block, their cleanup records, and the
//! kind of an edge stay on the `Cfg` ([`Cfg::handlers_at`](crate::Cfg::handlers_at),
//! [`Cfg::cleanup`](crate::Cfg::cleanup), [`Cfg::edge`](crate::Cfg::edge)).

extern crate alloc;
use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use crate::block::BlockId;
use crate::edge::EdgeId;
use crate::region::HandlerKind;

/// Classification of a block's role in exception handling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EhBlockKind {
    /// Normal code — not part of an EH construct.
    Normal,
    /// A landing pad — first block of an exception handler.
    LandingPad,
    /// A cleanup block — executes during stack unwinding (finally).
    Cleanup,
    /// A catch dispatch — selects among multiple handlers.
    CatchSwitch,
    /// A block that rethrows the active exception or continues execution
    /// after it: the source of an
    /// [`ExceptionResume`](crate::EdgeKind::ExceptionResume) or
    /// [`ExceptionContinue`](crate::EdgeKind::ExceptionContinue) edge.
    Resume,
}

impl From<HandlerKind> for EhBlockKind {
    fn from(kind: HandlerKind) -> Self {
        match kind {
            HandlerKind::Catch | HandlerKind::CatchAll => Self::LandingPad,
            HandlerKind::Finally | HandlerKind::Fault => Self::Cleanup,
            HandlerKind::Filter { .. } => Self::CatchSwitch,
        }
    }
}

/// EH model for a CFG.
///
/// Every block-keyed table is a dense vector indexed by block, sized by the
/// source CFG's block bound: the classification is total over blocks anyway,
/// and the rest answer in one load instead of a tree descent.
#[derive(Debug, Clone)]
pub struct EhModel {
    block_kinds: Vec<EhBlockKind>,
    eh_edges: Vec<EdgeId>,
    protected_by: Vec<BTreeSet<BlockId>>,
}

mod build;

static NO_PROTECTED_BLOCKS: BTreeSet<BlockId> = BTreeSet::new();

impl EhModel {
    /// The block bound this model was computed over.
    ///
    /// Every block-keyed query answers for an index below it.
    #[must_use]
    pub fn block_bound(&self) -> usize {
        self.block_kinds.len()
    }

    /// The exception-handling role of one block.
    ///
    /// A block outside the model — one added after it was computed — reads as
    /// [`EhBlockKind::Normal`].
    #[must_use]
    pub fn block_kind(&self, block: BlockId) -> EhBlockKind {
        self.block_kinds
            .get(block.index())
            .copied()
            .unwrap_or(EhBlockKind::Normal)
    }

    /// Every block and its exception-handling role, in block order.
    pub fn block_kinds(&self) -> impl Iterator<Item = (BlockId, EhBlockKind)> + '_ {
        self.block_kinds
            .iter()
            .enumerate()
            .map(|(index, &kind)| (BlockId::from_index(index), kind))
    }

    /// The blocks whose role is `kind`, in block order.
    pub fn blocks_of(&self, kind: EhBlockKind) -> impl Iterator<Item = BlockId> + '_ {
        self.block_kinds()
            .filter(move |&(_, candidate)| candidate == kind)
            .map(|(block, _)| block)
    }

    /// Every exception-control edge — handler, unwind, leave, resume, and
    /// continue — in edge identity order. Read the kind and the
    /// caller-owned payload of each from the [`Cfg`](crate::Cfg) it was
    /// computed over.
    #[must_use]
    pub fn eh_edges(&self) -> &[EdgeId] {
        &self.eh_edges
    }

    /// The blocks `handler_entry` protects, empty when it protects none.
    #[must_use]
    pub fn protected_by(&self, handler_entry: BlockId) -> &BTreeSet<BlockId> {
        self.protected_by
            .get(handler_entry.index())
            .unwrap_or(&NO_PROTECTED_BLOCKS)
    }
}

#[cfg(test)]
mod tests;
