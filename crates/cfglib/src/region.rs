//! Exception-handler region model.
//!
//! Regions represent protected areas of code (try blocks) and their
//! associated handlers (catch, finally, fault, filter). The model follows
//! the CLR / JVM exception metadata; a Windows SEH `__except` is a
//! [`HandlerKind::Filter`] and a `__finally` a [`HandlerKind::Finally`].
//!
//! These regions are declared by the consumer. The single-entry single-exit
//! regions that the shape of the graph has are a different thing:
//! [`ProgramStructureTree`](crate::ProgramStructureTree) computes those.
//!
//! Regions are **optional metadata** on a [`Cfg`] — GPU
//! shaders and simple ISAs simply leave the region list empty. A region's
//! identity is its position in the [`Cfg`]'s region list:
//! [`Cfg::add_region`] returns it, and [`Cfg::regions`] yields it beside each
//! region.
//!
//! # Granularity contract
//!
//! Protection is **block-granular**: every instruction of a protected
//! block may unwind. A frontend needing statement-granular protection
//! (source `try` bodies) splits blocks at the protected boundaries
//! ([`Cfg::split_block`](crate::Cfg::split_block)) so block granularity is
//! exact — the same convention the source-CFG lowering uses for every
//! other boundary.
//!
//! # Block references
//!
//! The metadata names blocks in two ways. A region's protected set and a
//! handler's known body are *sets*: a block that is removed simply drops out
//! of them at the next [`Cfg::compact`](crate::Cfg::compact). A handler
//! entry, a filter block, a cleanup exit block, and a continuation target
//! are *single references* with no stand-in, so the structural passes
//! ([`simplify_mapped`](crate::simplify_mapped),
//! [`merge_blocks_mapped`](crate::merge_blocks_mapped),
//! [`remove_empty_blocks_mapped`](crate::remove_empty_blocks_mapped),
//! [`remove_unreachable_mapped`](crate::remove_unreachable_mapped),
//! [`contract_edge_mapped`](crate::contract_edge_mapped)) never remove or merge those blocks
//! away. [`Cfg::split_block`](crate::Cfg::split_block) keeps both kinds true:
//! the new tail joins every set that held the original.
//! [`Cfg::check_regions`] checks that every single reference names a live
//! block and every region and handler reference resolves.
//!
//! # Cleanup and filter data
//!
//! Handler-body completeness is modeled through [`HandlerBody`]: source and
//! normalized native frontends can provide an exact block set, while
//! table-driven formats that encode only a handler entry can say that the
//! remaining extent is unknown.
//!
//! A source lowering may put the protected body in blocks of its own, run
//! every non-local transfer (`break`, `continue`, `return`, `goto`) and every
//! unwind through one shared cleanup block, and register a filtered `catch`
//! (C# `when`) as a plain [`HandlerKind::Catch`] because the predicate sits in
//! the pad rather than in a funclet of its own. `Finally` continuations that
//! depend on the entry reason (normal / exceptional / break-out), and handler
//! filters keyed by consumer types instead of [`HandlerKind::Filter`]'s block
//! reference, are answered by two records:
//!
//! - [`Cleanup`] records what a cleanup handler does once its body ends: the
//!   block it exits from and the [`Continuation`]s it selects among, each
//!   tagged with the [`CompletionReason`] that entered the cleanup. A cleanup
//!   lowered once (rather than duplicated per route) otherwise leaves a fan of
//!   indistinguishable out-edges; with the record, an analysis reads
//!   cleanup-then-continue structure instead. Continuations are library-typed
//!   ([`BlockId`] plus a reason), so the [`Cfg`] owns them:
//!   [`Cfg::add_continuation`], [`Cfg::set_cleanup_exit`], [`Cfg::cleanup`].
//! - [`HandlerMetadata`] carries consumer-owned handler data — for example a
//!   filter predicate identity or a CLR catch-type token — beside the CFG.
//!   It is a side table rather than a field or a type parameter, so no
//!   consumer type escapes into [`Cfg`]'s signature.

extern crate alloc;
use alloc::collections::BTreeSet;
use alloc::vec;
use alloc::vec::Vec;

mod error;
mod metadata;
mod recover;

pub use error::RegionError;
pub use metadata::HandlerMetadata;
pub use recover::{
    ExclusiveExtent, ExtentIssue, ExtentPromotionStatus, promote_exclusive_extents,
    promote_handler_extents, recover_exclusive_extents,
};

use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::identity::define_dense_id;

define_dense_id! {
    /// Opaque identifier for a region within a [`Cfg`]: its position in the
    /// region list.
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub struct RegionId(pub(crate) u32);
    display = "region";
    /// Create a `RegionId` from a dense zero-based index.
    ///
    /// # Panics
    ///
    /// Panics when `index` exceeds `u32::MAX`.
    from_index = "region index exceeds u32::MAX";
}

/// A protected region (try block) and its handlers.
///
/// A region has no identity field: [`Cfg::add_region`] assigns its
/// [`RegionId`], which is its position in the region list.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Region {
    /// Blocks covered by the protected (try) region.
    pub protected_blocks: BTreeSet<BlockId>,
    /// Exception handlers attached to this region.
    pub handlers: Vec<Handler>,
    /// Parent region (for nested try/catch).
    pub parent: Option<RegionId>,
}

/// An exception handler attached to a [`Region`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Handler {
    /// Entry block of the handler.
    pub entry: BlockId,
    /// Whether the complete handler body is known, and its blocks when it is.
    pub body: HandlerBody,
    /// The handler classification.
    pub kind: HandlerKind,
}

impl Handler {
    /// The blocks the handler names: its entry, its filter block when it is a
    /// [`HandlerKind::Filter`], then the blocks of its known extent (which
    /// hold the entry again). A handler of unknown extent names no extent
    /// blocks.
    pub fn blocks(&self) -> impl Iterator<Item = BlockId> + '_ {
        core::iter::once(self.entry)
            .chain(self.kind.filter_block())
            .chain(self.body.blocks().into_iter().flatten().copied())
    }
}

/// Completeness of an exception handler's block extent.
///
/// Exception tables commonly identify the protected range and handler entry
/// without encoding where the handler body ends. Keeping that state distinct
/// from an empty body prevents consumers from treating a guessed extent as
/// lossless metadata.
///
/// With the `serde` feature, this serializes as an externally tagged enum
/// (`{"Known":[…]}` / `"Unknown"`); snapshots that stored a handler body as
/// a plain block sequence predate this type and no longer deserialize.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum HandlerBody {
    /// Only the handler entry is known; the complete body extent is not.
    Unknown,
    /// The complete set of blocks belonging to the handler, which includes
    /// the handler's own entry. [`ControlTree`](crate::ControlTree) treats a
    /// set without it as a frontend bug.
    Known(BTreeSet<BlockId>),
}

impl HandlerBody {
    /// Construct a handler body whose complete extent is unavailable.
    #[inline]
    #[must_use]
    pub const fn unknown() -> Self {
        Self::Unknown
    }

    /// Construct a complete handler body from its blocks.
    #[must_use]
    pub fn known(blocks: impl IntoIterator<Item = BlockId>) -> Self {
        Self::Known(blocks.into_iter().collect())
    }

    /// Return the complete block set, or `None` when its extent is unknown.
    #[inline]
    #[must_use]
    pub const fn blocks(&self) -> Option<&BTreeSet<BlockId>> {
        match self {
            Self::Unknown => None,
            Self::Known(blocks) => Some(blocks),
        }
    }

    /// Whether the complete block extent is known.
    #[inline]
    #[must_use]
    pub const fn is_known(&self) -> bool {
        matches!(self, Self::Known(_))
    }
}

impl From<BTreeSet<BlockId>> for HandlerBody {
    fn from(blocks: BTreeSet<BlockId>) -> Self {
        Self::Known(blocks)
    }
}

/// Classification of an exception handler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum HandlerKind {
    /// Catch handler — catches a specific exception type.
    ///
    /// Frontends retain their type token in a [`HandlerMetadata`] side table.
    Catch,
    /// Catch-all handler — catches any exception.
    CatchAll,
    /// Finally handler — always executed.
    Finally,
    /// Fault handler — executed on exception only (CLR).
    Fault,
    /// Filter handler — a user-defined predicate determines whether
    /// this handler catches the exception (a CLR filter clause, a Windows
    /// SEH `__except`).
    Filter {
        /// Block containing the filter predicate.
        filter_block: BlockId,
    },
}

impl HandlerKind {
    /// The name of the kind in the CFG text form: `catch`, `catch_all`,
    /// `finally`, `fault` or `filter`. A filter's block is not part of its
    /// name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Catch => "catch",
            Self::CatchAll => "catch_all",
            Self::Finally => "finally",
            Self::Fault => "fault",
            Self::Filter { .. } => "filter",
        }
    }

    /// The kind named `name`, the inverse of [`name`](Self::name). A
    /// `filter` takes its predicate block from `filter_block`, so it is
    /// `None` without one; every other kind ignores `filter_block`.
    #[must_use]
    pub fn from_name(name: &str, filter_block: Option<BlockId>) -> Option<Self> {
        match name {
            "catch" => Some(Self::Catch),
            "catch_all" => Some(Self::CatchAll),
            "finally" => Some(Self::Finally),
            "fault" => Some(Self::Fault),
            "filter" => filter_block.map(|filter_block| Self::Filter { filter_block }),
            _ => None,
        }
    }

    /// The block holding the filter predicate, for a
    /// [`Filter`](Self::Filter) handler.
    #[inline]
    #[must_use]
    pub const fn filter_block(self) -> Option<BlockId> {
        match self {
            Self::Filter { filter_block } => Some(filter_block),
            Self::Catch | Self::CatchAll | Self::Finally | Self::Fault => None,
        }
    }
}

/// A reference to one [`Handler`] of one [`Region`] — the region's id plus
/// the handler's position in [`Region::handlers`].
///
/// Handlers have no identity of their own because they are stored inline, so
/// this pair is what the side data ([`Cleanup`] records, [`HandlerMetadata`]
/// payloads) is keyed by.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HandlerRef {
    region: RegionId,
    handler: u32,
}

impl HandlerRef {
    /// Reference the handler at `index` of `region`.
    ///
    /// # Panics
    ///
    /// Panics when `index` exceeds `u32::MAX`.
    #[must_use]
    pub fn new(region: RegionId, index: usize) -> Self {
        Self {
            region,
            handler: u32::try_from(index).expect("handler index exceeds u32::MAX"),
        }
    }

    /// The region owning the handler.
    #[inline]
    #[must_use]
    pub fn region(self) -> RegionId {
        self.region
    }

    /// The handler's position in [`Region::handlers`].
    #[inline]
    #[must_use]
    pub const fn index(self) -> usize {
        self.handler as usize
    }
}

impl core::fmt::Display for HandlerRef {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}.handler{}", self.region, self.handler)
    }
}

/// Why control entered a cleanup handler — the reason that selects which of
/// its [`Continuation`]s runs once the cleanup body ends.
///
/// A `finally` lowered as a single block (rather than duplicated once per
/// route) is entered by every way out of the protected region, so the reason
/// is what tells those routes apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CompletionReason {
    /// The protected body, or a handler, ran to its end.
    Normal,
    /// A transfer left the region for another block of the same callable —
    /// `break`, `continue`, `goto`, a switch exit. Which one is the
    /// [`Continuation::target`] block.
    Transfer,
    /// A `return` left the callable.
    Return,
    /// An exception is propagating through the cleanup, to a handler further
    /// out or out of the callable.
    Unwind,
}

/// One recorded route out of a cleanup handler: the reason control entered
/// it, and where control goes once its body ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Continuation {
    /// The completion reason that entered the cleanup on this route.
    pub reason: CompletionReason,
    /// The block control continues at — the next cleanup of a chain, a loop's
    /// break target, the enclosing handler, the callable's exit.
    pub target: BlockId,
}

/// What a cleanup handler does once its body ends: the block it exits from
/// and every route it can continue along.
///
/// A frontend records routes as it lowers them (a route exists long before
/// the block it continues into does), and attaches them to the handler once
/// the region is registered — [`Cfg::add_continuation`] and
/// [`Cfg::set_cleanup_exit`]. The [`Cfg`] keys the record by its handler.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Cleanup {
    /// The block the cleanup body ends in, from which every continuation
    /// edge leaves. `None` until the body is lowered, and afterwards when the
    /// cleanup itself diverges (a `return` inside the `finally`): then the
    /// recorded continuations are unreachable, which is what the source says.
    pub exit_block: Option<BlockId>,
    /// The recorded routes, in first-recorded order, each distinct.
    pub continuations: Vec<Continuation>,
}

impl Cleanup {
    /// The blocks control continues at when the cleanup was entered for
    /// `reason`, in recorded order.
    pub fn targets_for(&self, reason: CompletionReason) -> impl Iterator<Item = BlockId> + '_ {
        self.continuations
            .iter()
            .filter(move |continuation| continuation.reason == reason)
            .map(|continuation| continuation.target)
    }
}

/// Precomputed innermost-protecting-region lookup.
///
/// It answers, in O(1) per block after one O(regions × blocks) build, which
/// region protects a block most deeply: the latest-added covering region
/// wins, which is the innermost one for a frontend that registers an outer
/// region before the regions nested in it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegionIndex {
    innermost: Vec<Option<RegionId>>,
}

impl RegionIndex {
    /// Compute the index from a CFG's current regions.
    ///
    /// The index is a snapshot: recompute after adding regions or blocks.
    #[must_use]
    pub fn compute<I, E>(cfg: &Cfg<I, E>) -> Self {
        let mut innermost = vec![None; cfg.block_bound()];
        for (id, region) in cfg.regions() {
            for &block in &region.protected_blocks {
                if let Some(slot) = innermost.get_mut(block.index()) {
                    *slot = Some(id);
                }
            }
        }
        Self { innermost }
    }

    /// The innermost region protecting `block`, if any.
    #[must_use]
    pub fn protecting(&self, block: BlockId) -> Option<RegionId> {
        self.innermost.get(block.index()).copied().flatten()
    }
}

#[cfg(test)]
mod tests;
