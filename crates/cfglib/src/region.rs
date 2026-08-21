//! Exception-handler region model.
//!
//! Regions represent protected areas of code (try blocks) and their
//! associated handlers (catch, finally, fault, filter). This model
//! is inspired by the CLR / JVM exception metadata and Echo's
//! `ExceptionHandlerRegion`.
//!
//! Regions are **optional metadata** on a [`Cfg`] — GPU
//! shaders and simple ISAs simply leave the region list empty.
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
//! # v2 (consumer-gated, now driven by a real consumer)
//!
//! Two needs were deliberately left undesigned until the first source
//! language with `try` landed: `Finally` continuations that depend on the
//! entry reason (normal / exceptional / break-out), and handler filters keyed
//! by consumer types instead of [`HandlerKind::Filter`]'s block reference.
//! That consumer arrived — a source lowering that puts the protected body in
//! blocks of its own, runs every non-local transfer (`break`, `continue`,
//! `return`, `goto`) and every unwind through one shared cleanup block, and
//! registers a filtered `catch` (C# `when`) as a plain [`HandlerKind::Catch`]
//! because the predicate sits in the pad rather than in a funclet of its own.
//! Both shapes are answered here, and both are **additive**: [`Region`] and
//! [`Handler`] are unchanged, so existing frontends keep compiling.
//!
//! - [`Cleanup`] records what a cleanup handler does once its body ends: the
//!   block it resumes from and the [`Continuation`]s it selects among, each
//!   tagged with the [`CompletionReason`] that entered the cleanup. A cleanup
//!   lowered once (rather than duplicated per route) otherwise leaves a fan of
//!   indistinguishable out-edges; with the record, [`EhModel`](crate::EhModel)
//!   and any analysis over it read cleanup-then-continue structure instead.
//!   Continuations are library-typed ([`BlockId`] plus a reason), so the
//!   [`Cfg`] owns them: [`Cfg::add_continuation`], [`Cfg::set_cleanup_resume`],
//!   [`Cfg::cleanup`].
//! - [`HandlerFilters`] carries the filter *predicate identity* a frontend
//!   already has (a syntax node, an interned expression, a type id) beside the
//!   CFG. It is a side table rather than a field or a type parameter, so no
//!   consumer type escapes into [`Cfg`]'s signature.

extern crate alloc;
use alloc::collections::BTreeSet;
use alloc::vec;
use alloc::vec::Vec;

use crate::block::BlockId;
use crate::cfg::Cfg;

/// Opaque identifier for a region within a [`Cfg`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RegionId(pub(crate) u32);

impl RegionId {
    pub(crate) fn from_index(index: usize) -> Self {
        Self(u32::try_from(index).expect("region index exceeds u32::MAX"))
    }

    /// Create a `RegionId` from a raw index.
    #[inline]
    #[must_use]
    pub fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    /// Returns the raw index.
    #[inline]
    #[must_use]
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

impl core::fmt::Display for RegionId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "region{}", self.0)
    }
}

/// A protected region (try block) and its handlers.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Region {
    /// Region identity.
    pub id: RegionId,
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
    /// All blocks in the handler body.
    pub body: BTreeSet<BlockId>,
    /// The handler classification.
    pub kind: HandlerKind,
}

/// Classification of an exception handler.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum HandlerKind {
    /// Catch handler — catches a specific exception type.
    Catch,
    /// Catch-all handler — catches any exception.
    CatchAll,
    /// Finally handler — always executed.
    Finally,
    /// Fault handler — executed on exception only (CLR).
    Fault,
    /// Filter handler — a user-defined predicate determines whether
    /// this handler catches the exception.
    Filter {
        /// Block containing the filter predicate.
        filter_block: BlockId,
    },
}

/// A reference to one [`Handler`] of one [`Region`] — the region's id plus
/// the handler's position in [`Region::handlers`].
///
/// Handlers have no identity of their own because they are stored inline, so
/// this pair is what the v2 side data ([`Cleanup`] records,
/// [`HandlerFilters`] payloads) is keyed by.
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
    pub fn index(self) -> usize {
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
    /// [`Continuation::resume`] block.
    Transfer,
    /// A `return` left the callable.
    Return,
    /// An exception is propagating through the cleanup, to a handler further
    /// out or out of the callable.
    Unwind,
}

/// One recorded route out of a cleanup handler: the reason control entered
/// it, and where control resumes once its body ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Continuation {
    /// The completion reason that entered the cleanup on this route.
    pub reason: CompletionReason,
    /// The block control resumes at — the next cleanup of a chain, a loop's
    /// break target, the enclosing handler, the callable's exit.
    pub resume: BlockId,
}

/// What a cleanup handler does once its body ends: where it resumes from and
/// every route it can continue along.
///
/// A frontend records routes as it lowers them (a route exists long before
/// the block it continues into does), and attaches them to the handler once
/// the region is registered — [`Cfg::add_continuation`] and
/// [`Cfg::set_cleanup_resume`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Cleanup {
    /// The handler this record belongs to — normally a
    /// [`HandlerKind::Finally`] or [`HandlerKind::Fault`] one.
    pub handler: HandlerRef,
    /// The block the cleanup body ends in, from which every continuation
    /// edge leaves. `None` until the body is lowered, and afterwards when the
    /// cleanup itself diverges (a `return` inside the `finally`): then the
    /// recorded continuations are unreachable, which is what the source says.
    pub resume_from: Option<BlockId>,
    /// The recorded routes, in first-recorded order, each distinct.
    pub continuations: Vec<Continuation>,
}

impl Cleanup {
    /// The blocks control resumes at when the cleanup was entered for
    /// `reason`, in recorded order.
    pub fn resumes_for(&self, reason: CompletionReason) -> impl Iterator<Item = BlockId> + '_ {
        self.continuations
            .iter()
            .filter(move |continuation| continuation.reason == reason)
            .map(|continuation| continuation.resume)
    }
}

/// Consumer-typed filter payloads for handlers, keyed by [`HandlerRef`].
///
/// [`HandlerKind::Filter`] names a *block* holding the predicate, which fits
/// bytecode where a filter is its own funclet. Source lowerings frequently
/// have no such block: C#'s `catch (E e) when (cond)` evaluates its predicate
/// in the pad itself, so the handler registers as [`HandlerKind::Catch`] and
/// the only useful description of the filter is whatever identifies the
/// predicate in the frontend's own world — a syntax-node id, an interned
/// expression, a caught type, a bitmask. That payload is therefore consumer
/// DATA rather than a library type, and it is orthogonal to
/// [`HandlerKind`]: any handler may carry one.
///
/// It is a **side table** and not a field or a type parameter on
/// [`Handler`] on purpose. A `Handler<F>` would force `Region<F>` and
/// `Cfg<I, F>`, infecting every algorithm signature in the crate, and a new
/// field would break every existing struct-literal construction. A separate
/// table keeps [`Cfg`] object-simple, keeps existing frontends compiling, and
/// is dropped by consumers that have no filters at all.
///
/// Storage is a sorted `Vec` of pairs rather than a map: `serde` renders it
/// as a sequence, so it survives formats that only admit string map keys,
/// and lookups stay a binary search.
///
/// # Examples
///
/// ```
/// use cfglib::{HandlerFilters, HandlerRef, RegionId};
///
/// // `catch (IOException e) when (e.Retryable)` — a plain `Catch` handler
/// // whose predicate the frontend identifies by its own expression handle.
/// let handler = HandlerRef::new(RegionId::from_raw(0), 0);
/// let mut filters = HandlerFilters::new();
/// assert!(filters.set(handler, "expr#17").is_none());
///
/// assert_eq!(filters.get(handler), Some(&"expr#17"));
/// assert_eq!(filters.len(), 1);
/// // Re-registering returns the payload it replaced.
/// assert_eq!(filters.set(handler, "expr#18"), Some("expr#17"));
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HandlerFilters<F> {
    /// Sorted by key, so lookups binary-search and iteration is stable.
    entries: Vec<(HandlerRef, F)>,
}

impl<F> HandlerFilters<F> {
    /// An empty table.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Attach `filter` to `handler`, returning the payload it replaced.
    pub fn set(&mut self, handler: HandlerRef, filter: F) -> Option<F> {
        match self.entries.binary_search_by_key(&handler, |(key, _)| *key) {
            Ok(at) => Some(core::mem::replace(&mut self.entries[at].1, filter)),
            Err(at) => {
                self.entries.insert(at, (handler, filter));
                None
            }
        }
    }

    /// The filter payload attached to `handler`, if any.
    #[must_use]
    pub fn get(&self, handler: HandlerRef) -> Option<&F> {
        let at = self
            .entries
            .binary_search_by_key(&handler, |(key, _)| *key)
            .ok()?;
        Some(&self.entries[at].1)
    }

    /// Mutable access to the filter payload attached to `handler`.
    pub fn get_mut(&mut self, handler: HandlerRef) -> Option<&mut F> {
        let at = self
            .entries
            .binary_search_by_key(&handler, |(key, _)| *key)
            .ok()?;
        Some(&mut self.entries[at].1)
    }

    /// Detach and return the filter payload attached to `handler`.
    pub fn remove(&mut self, handler: HandlerRef) -> Option<F> {
        let at = self
            .entries
            .binary_search_by_key(&handler, |(key, _)| *key)
            .ok()?;
        Some(self.entries.remove(at).1)
    }

    /// The number of handlers carrying a filter.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether no handler carries a filter.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate over every `(handler, filter)` pair in handler order.
    pub fn iter(&self) -> impl Iterator<Item = (HandlerRef, &F)> {
        self.entries.iter().map(|(key, filter)| (*key, filter))
    }
}

impl<F> Default for HandlerFilters<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Precomputed innermost-protecting-region lookup.
///
/// [`Cfg::protecting_region`](crate::Cfg::protecting_region) scans the
/// region list backwards per query; over many blocks (AST lifting,
/// statement-level unwind edges) that is quadratic. This index answers the
/// same question in O(1) per block after one O(regions × blocks) build,
/// with identical innermost semantics (the latest-added covering region
/// wins).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegionIndex {
    innermost: Vec<Option<RegionId>>,
}

impl RegionIndex {
    /// Build the index from a CFG's current regions.
    ///
    /// The index is a snapshot: rebuild after adding regions or blocks.
    #[must_use]
    pub fn build<I>(cfg: &Cfg<I>) -> Self {
        let mut innermost = vec![None; cfg.num_blocks()];
        for region in cfg.regions() {
            for &block in &region.protected_blocks {
                if let Some(slot) = innermost.get_mut(block.index()) {
                    *slot = Some(region.id);
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
mod tests {
    use super::*;
    use crate::block::BlockId;
    use crate::cfg::Cfg;
    use crate::test_util::MockInst;
    use alloc::collections::BTreeSet;

    fn block_set(ids: &[u32]) -> BTreeSet<BlockId> {
        ids.iter().map(|&i| BlockId::from_raw(i)).collect()
    }

    #[test]
    fn add_region_assigns_sequential_ids() {
        let mut cfg: Cfg<MockInst> = Cfg::new();
        let r0 = cfg.add_region(Region {
            id: RegionId(999), // should be overwritten
            protected_blocks: block_set(&[0]),
            handlers: alloc::vec![],
            parent: None,
        });
        let r1 = cfg.add_region(Region {
            id: RegionId(999),
            protected_blocks: block_set(&[0]),
            handlers: alloc::vec![],
            parent: Some(r0),
        });
        assert_eq!(r0.index(), 0);
        assert_eq!(r1.index(), 1);
        assert_eq!(cfg.regions().len(), 2);
        assert_eq!(cfg.regions()[1].parent, Some(r0));
    }

    #[test]
    fn protecting_region_finds_innermost() {
        let mut cfg: Cfg<MockInst> = Cfg::new();
        let _b1 = cfg.new_block();
        let _b2 = cfg.new_block();
        // Outer region protects blocks 0,1,2.
        let outer = cfg.add_region(Region {
            id: RegionId(0),
            protected_blocks: block_set(&[0, 1, 2]),
            handlers: alloc::vec![],
            parent: None,
        });
        // Inner region protects block 1 only.
        let inner = cfg.add_region(Region {
            id: RegionId(0),
            protected_blocks: block_set(&[1]),
            handlers: alloc::vec![],
            parent: Some(outer),
        });

        // Block 1 should find the inner (last-added) region.
        let r = cfg.protecting_region(BlockId::from_raw(1)).unwrap();
        assert_eq!(r.id, inner);

        // Block 0 should find the outer region.
        let r = cfg.protecting_region(BlockId::from_raw(0)).unwrap();
        assert_eq!(r.id, outer);

        // Block that's not in any region.
        let b3 = cfg.new_block();
        assert!(cfg.protecting_region(b3).is_none());

        // The precomputed index agrees with the linear scan everywhere.
        let index = RegionIndex::build(&cfg);
        for block in [0, 1, 2] {
            let block = BlockId::from_raw(block);
            assert_eq!(
                index.protecting(block),
                cfg.protecting_region(block).map(|r| r.id)
            );
        }
        assert_eq!(index.protecting(BlockId::from_raw(1)), Some(inner));
        // b3 belongs to no region.
        assert!(index.protecting(b3).is_none());
    }

    /// A CFG with one protected block, one catch pad, and one cleanup — the
    /// shape a source `try`/`catch`/`finally` lowers to.
    fn try_catch_finally() -> (Cfg<MockInst>, RegionId, [BlockId; 4]) {
        let mut cfg: Cfg<MockInst> = Cfg::new();
        let pad = cfg.new_block();
        let cleanup = cfg.new_block();
        let after = cfg.new_block();
        let exit = cfg.new_block();
        let region = cfg.add_region(Region {
            id: RegionId(0),
            protected_blocks: block_set(&[0]),
            handlers: alloc::vec![
                Handler {
                    entry: pad,
                    body: [pad].into_iter().collect(),
                    kind: HandlerKind::Catch,
                },
                Handler {
                    entry: cleanup,
                    body: [cleanup].into_iter().collect(),
                    kind: HandlerKind::Finally,
                },
            ],
            parent: None,
        });
        (cfg, region, [pad, cleanup, after, exit])
    }

    #[test]
    fn handler_ref_names_one_handler_of_one_region() {
        let (cfg, region, [pad, cleanup, _, _]) = try_catch_finally();
        let catch = HandlerRef::new(region, 0);
        let finally = HandlerRef::new(region, 1);

        assert_eq!(catch.region(), region);
        assert_eq!(finally.index(), 1);
        assert!(catch < finally, "handlers order by position in the region");
        assert_eq!(
            cfg.regions()[region.index()].handlers[catch.index()].entry,
            pad
        );
        assert_eq!(
            cfg.regions()[region.index()].handlers[finally.index()].entry,
            cleanup
        );
        assert_eq!(alloc::format!("{finally}"), "region0.handler1");
    }

    #[test]
    fn a_cleanup_records_one_route_per_distinct_destination() {
        let (mut cfg, region, [_, cleanup, after, exit]) = try_catch_finally();
        let finally = HandlerRef::new(region, 1);
        assert!(
            cfg.cleanup(finally).is_none(),
            "no record until one is made"
        );

        cfg.set_cleanup_resume(finally, cleanup);
        for continuation in [
            Continuation {
                reason: CompletionReason::Normal,
                resume: after,
            },
            Continuation {
                reason: CompletionReason::Return,
                resume: exit,
            },
            // A second `return` route to the same block is the same route.
            Continuation {
                reason: CompletionReason::Return,
                resume: exit,
            },
            // Same destination, different reason: a distinct route.
            Continuation {
                reason: CompletionReason::Unwind,
                resume: exit,
            },
        ] {
            cfg.add_continuation(finally, continuation);
        }

        let record = cfg.cleanup(finally).expect("recorded");
        assert_eq!(record.handler, finally);
        assert_eq!(record.resume_from, Some(cleanup));
        assert_eq!(
            record
                .continuations
                .iter()
                .map(|route| (route.reason, route.resume))
                .collect::<Vec<_>>(),
            alloc::vec![
                (CompletionReason::Normal, after),
                (CompletionReason::Return, exit),
                (CompletionReason::Unwind, exit),
            ],
            "distinct routes, in first-recorded order"
        );

        // The reason selects the route — the point of the record.
        assert_eq!(
            record
                .resumes_for(CompletionReason::Normal)
                .collect::<Vec<_>>(),
            alloc::vec![after]
        );
        assert_eq!(
            record
                .resumes_for(CompletionReason::Return)
                .collect::<Vec<_>>(),
            alloc::vec![exit]
        );
        assert_eq!(record.resumes_for(CompletionReason::Transfer).next(), None);
        assert_eq!(cfg.cleanups().len(), 1);
    }

    #[test]
    fn a_diverging_cleanup_has_routes_but_no_resume_block() {
        let (mut cfg, region, [_, _, after, _]) = try_catch_finally();
        let finally = HandlerRef::new(region, 1);
        cfg.add_continuation(
            finally,
            Continuation {
                reason: CompletionReason::Normal,
                resume: after,
            },
        );

        // The cleanup body ends in a `return`, so nothing leaves it: the
        // recorded route stays, unreachable, which is what the source says.
        let record = cfg.cleanup(finally).expect("recorded");
        assert_eq!(record.resume_from, None);
        assert_eq!(record.continuations.len(), 1);
    }

    #[test]
    fn cleanups_are_per_handler() {
        let (mut cfg, region, [_, _, after, exit]) = try_catch_finally();
        let catch = HandlerRef::new(region, 0);
        let finally = HandlerRef::new(region, 1);
        cfg.add_continuation(
            catch,
            Continuation {
                reason: CompletionReason::Normal,
                resume: after,
            },
        );
        cfg.add_continuation(
            finally,
            Continuation {
                reason: CompletionReason::Unwind,
                resume: exit,
            },
        );

        assert_eq!(cfg.cleanups().len(), 2);
        assert_eq!(
            cfg.cleanup(catch)
                .map(|record| record.continuations[0].resume),
            Some(after)
        );
        assert_eq!(
            cfg.cleanup(finally)
                .map(|record| record.continuations[0].reason),
            Some(CompletionReason::Unwind)
        );
    }

    #[test]
    fn handler_filters_attach_consumer_payloads_to_any_handler_kind() {
        let (cfg, region, _) = try_catch_finally();
        // The filtered `catch` registers as a plain `Catch` — its predicate
        // lives in the pad — so the payload is the frontend's own handle.
        let catch = HandlerRef::new(region, 0);
        let finally = HandlerRef::new(region, 1);
        assert_eq!(cfg.regions()[0].handlers[0].kind, HandlerKind::Catch);

        let mut filters: HandlerFilters<&str> = HandlerFilters::new();
        assert!(filters.is_empty());
        assert!(filters.set(finally, "unreachable").is_none());
        assert!(filters.set(catch, "e.Retryable").is_none());

        assert_eq!(filters.len(), 2);
        assert_eq!(filters.get(catch), Some(&"e.Retryable"));
        assert_eq!(
            filters.set(catch, "e.Retryable && !e.Fatal"),
            Some("e.Retryable")
        );
        if let Some(filter) = filters.get_mut(catch) {
            *filter = "e.Fatal";
        }
        assert_eq!(filters.get(catch), Some(&"e.Fatal"));
        // Iteration is in handler order regardless of insertion order.
        assert_eq!(
            filters.iter().map(|(key, _)| key).collect::<Vec<_>>(),
            alloc::vec![catch, finally]
        );

        assert_eq!(filters.remove(finally), Some("unreachable"));
        assert_eq!(filters.remove(finally), None);
        assert_eq!(filters.len(), 1);
        assert_eq!(
            HandlerFilters::<u32>::default(),
            HandlerFilters::<u32>::new()
        );
    }

    #[test]
    fn the_v2_additions_leave_v1_construction_alone() {
        // Exactly the v1 shape: a region built by struct literal, handlers
        // built by struct literal, nothing else supplied.
        let mut cfg: Cfg<MockInst> = Cfg::new();
        let pad = cfg.new_block();
        let id = cfg.add_region(Region {
            id: RegionId(0),
            protected_blocks: block_set(&[0]),
            handlers: alloc::vec![Handler {
                entry: pad,
                body: [pad].into_iter().collect(),
                kind: HandlerKind::Finally,
            }],
            parent: None,
        });

        assert_eq!(cfg.regions().len(), 1);
        assert_eq!(
            cfg.protecting_region(BlockId::from_raw(0)).map(|r| r.id),
            Some(id)
        );
        // A frontend that records no routes carries no records.
        assert_eq!(cfg.cleanups().len(), 0);
        assert!(cfg.cleanup(HandlerRef::new(id, 0)).is_none());
    }

    #[test]
    fn handler_kind_variants() {
        let h = Handler {
            entry: BlockId::from_raw(1),
            body: block_set(&[1, 2]),
            kind: HandlerKind::Finally,
        };
        assert_eq!(h.kind, HandlerKind::Finally);

        let h2 = Handler {
            entry: BlockId::from_raw(3),
            body: block_set(&[3]),
            kind: HandlerKind::Filter {
                filter_block: BlockId::from_raw(4),
            },
        };
        assert!(matches!(h2.kind, HandlerKind::Filter { .. }));
    }
}
