//! Exception-region accessors of [`Cfg`].
//!
//! A region's identity is its position in the region list, and a handler's
//! is its region plus its position in [`Region::handlers`]. Region ids are
//! never reused: [`Cfg::remove_region`] leaves an inert tombstone. Lookups
//! by an id that names no region or handler panic, like
//! [`Cfg::block`](Cfg::block); [`Cfg::contains_region`] and
//! [`Cfg::contains_handler`] answer first when the id comes from elsewhere.

extern crate alloc;

use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use crate::block::BlockId;
use crate::graph::traverse::reachable;
use crate::graph::view::Direction;
use crate::region::{Handler, HandlerRef, Region, RegionError, RegionId};

use super::Cfg;
use super::anchors::BlockRefMut;

impl<I, E> Cfg<I, E> {
    /// Every exception-handler region with its identity, in identity order.
    pub fn regions(
        &self,
    ) -> impl DoubleEndedIterator<Item = (RegionId, &Region)> + ExactSizeIterator + '_ {
        self.regions
            .iter()
            .enumerate()
            .map(|(index, region)| (RegionId::from_index(index), region))
    }

    /// The number of regions, removed ones included: every id below it names
    /// a region.
    #[inline]
    #[must_use]
    pub fn region_count(&self) -> usize {
        self.regions.len()
    }

    /// Whether `id` names a region of this CFG.
    #[inline]
    #[must_use]
    pub fn contains_region(&self, id: RegionId) -> bool {
        id.index() < self.regions.len()
    }

    /// Look up one exception region by its identity.
    ///
    /// # Panics
    ///
    /// Panics if `id` names no region of this CFG.
    #[must_use]
    pub fn region(&self, id: RegionId) -> &Region {
        self.regions
            .get(id.index())
            .unwrap_or_else(|| missing_region(id))
    }

    /// Mutable access to one exception region.
    ///
    /// The region's references are not checked as they change; run
    /// [`check_regions`](Self::check_regions) after an edit that may break
    /// one.
    ///
    /// # Panics
    ///
    /// Panics if `id` names no region of this CFG.
    pub fn region_mut(&mut self, id: RegionId) -> &mut Region {
        self.regions
            .get_mut(id.index())
            .unwrap_or_else(|| missing_region(id))
    }

    /// Whether `handler` names a handler of a region of this CFG.
    #[must_use]
    pub fn contains_handler(&self, handler: HandlerRef) -> bool {
        self.regions
            .get(handler.region().index())
            .is_some_and(|region| handler.index() < region.handlers.len())
    }

    /// Look up one handler by its region-and-position identity.
    ///
    /// # Panics
    ///
    /// Panics if `handler` names no handler of this CFG.
    #[must_use]
    pub fn handler(&self, handler: HandlerRef) -> &Handler {
        self.regions
            .get(handler.region().index())
            .and_then(|region| region.handlers.get(handler.index()))
            .unwrap_or_else(|| missing_handler(handler))
    }

    /// Mutable access to one handler.
    ///
    /// The handler's references are not checked as they change; run
    /// [`check_regions`](Self::check_regions) after an edit that may break
    /// one.
    ///
    /// # Panics
    ///
    /// Panics if `handler` names no handler of this CFG.
    pub fn handler_mut(&mut self, handler: HandlerRef) -> &mut Handler {
        self.regions
            .get_mut(handler.region().index())
            .and_then(|region| region.handlers.get_mut(handler.index()))
            .unwrap_or_else(|| missing_handler(handler))
    }

    /// Every handler with its identity, in region order and then in handler
    /// order.
    pub fn handlers(&self) -> impl Iterator<Item = (HandlerRef, &Handler)> + '_ {
        self.regions().flat_map(|(id, region)| {
            region
                .handlers
                .iter()
                .enumerate()
                .map(move |(index, handler)| (HandlerRef::new(id, index), handler))
        })
    }

    /// The handlers entered at `entry`, in the order of
    /// [`handlers`](Self::handlers). Several regions may share one pad.
    pub fn handlers_at(&self, entry: BlockId) -> impl Iterator<Item = (HandlerRef, &Handler)> + '_ {
        self.handlers()
            .filter(move |(_, handler)| handler.entry == entry)
    }

    /// Removes one region, leaving an inert tombstone (empty protected
    /// set, no handlers) so later region identities stay stable.
    /// Structuring and fidelity reporting ignore inert regions.
    ///
    /// The cleanup records of the region's handlers go with it: a cleanup
    /// describes a handler, and a handler that no longer exists cannot keep
    /// the blocks it exited from alive.
    ///
    /// # Panics
    ///
    /// Panics if `id` names no region of this CFG.
    pub fn remove_region(&mut self, id: RegionId) -> Region {
        let slot = self.region_mut(id);
        let removed = core::mem::replace(
            slot,
            Region {
                protected_blocks: BTreeSet::new(),
                handlers: Vec::new(),
                parent: None,
            },
        );
        self.cleanups.retain(|handler| handler.region() != id);
        removed
    }

    /// Removes every region none of whose handlers can be entered — each
    /// handler entry is unreachable from the graph entry, so nothing in
    /// the protected range raises into it. Frontends that model
    /// exceptional flow exactly leave such regions behind when a source
    /// construct's body provably cannot throw (a `try`/`finally` around
    /// arithmetic); dropping them states the same semantics with plain
    /// sequential flow. Returns the number of removed regions.
    ///
    /// Run this before [`remove_unreachable`](crate::remove_unreachable) to
    /// drop the dead handler blocks too: that pass keeps every block a
    /// surviving region names.
    ///
    /// # Examples
    ///
    /// ```
    /// use cfglib::{Cfg, Handler, HandlerBody, HandlerKind, Region, RegionIndex};
    ///
    /// let mut cfg = Cfg::<u32>::new();
    /// let body = cfg.entry();
    /// let handler = cfg.new_block();
    /// // The handler has no incoming edge: nothing in `body` can raise into it.
    /// cfg.add_region(Region {
    ///     protected_blocks: [body].into_iter().collect(),
    ///     handlers: vec![Handler {
    ///         entry: handler,
    ///         body: HandlerBody::known([handler]),
    ///         kind: HandlerKind::CatchAll,
    ///     }],
    ///     parent: None,
    /// })?;
    ///
    /// assert_eq!(cfg.remove_unreachable_regions(), 1);
    /// assert!(RegionIndex::compute(&cfg).protecting(body).is_none());
    /// # Ok::<(), cfglib::RegionError>(())
    /// ```
    pub fn remove_unreachable_regions(&mut self) -> usize {
        let reachable = reachable(self, [self.entry()], Direction::Forward);
        let doomed: Vec<RegionId> = self
            .regions()
            .filter(|(_, region)| {
                !region.handlers.is_empty()
                    && region.handlers.iter().all(|handler| {
                        !reachable
                            .get(handler.entry.index())
                            .copied()
                            .unwrap_or(false)
                    })
            })
            .map(|(id, _)| id)
            .collect();
        let removed = doomed.len();
        for id in doomed {
            self.remove_region(id);
        }
        removed
    }

    /// Add a region and return its id, the next position of the region list.
    ///
    /// # Errors
    ///
    /// Returns [`RegionError::ParentMissing`] when the parent names no region
    /// of this CFG, and [`RegionError::BlockNotLive`] when a handler entry or
    /// filter block names no live block. The CFG is unchanged then.
    pub fn add_region(&mut self, region: Region) -> Result<RegionId, RegionError> {
        if let Some(parent) = region.parent
            && !self.contains_region(parent)
        {
            return Err(RegionError::ParentMissing(parent));
        }
        if let Some(block) = region
            .handlers
            .iter()
            .flat_map(|handler| core::iter::once(handler.entry).chain(handler.kind.filter_block()))
            .find(|&block| !self.contains_block(block))
        {
            return Err(RegionError::BlockNotLive(block));
        }
        let id = RegionId::from_index(self.regions.len());
        self.regions.push(region);
        Ok(id)
    }

    /// Check every exception-region invariant: each region's parent is a
    /// region, each cleanup record belongs to a handler and lists each route
    /// once, and every single block reference (handler entry, filter block,
    /// cleanup exit block, continuation target) names a live block.
    ///
    /// The mutators keep these true for the references they are handed;
    /// [`region_mut`](Self::region_mut), [`handler_mut`](Self::handler_mut)
    /// and [`remove_block`](Self::remove_block) can break them.
    ///
    /// # Errors
    ///
    /// Returns the first broken invariant: parents in region order, then
    /// cleanup records in handler order, then single block references in the
    /// order of the metadata.
    pub fn check_regions(&self) -> Result<(), RegionError> {
        self.region_errors().next().map_or(Ok(()), Err)
    }

    /// Every broken exception-region invariant, in the order
    /// [`check_regions`](Self::check_regions) reports the first of them.
    /// [`verify`](crate::verify) lists them all.
    pub(crate) fn region_errors(&self) -> impl Iterator<Item = RegionError> + '_ {
        let parents = self.regions.iter().filter_map(|region| {
            region
                .parent
                .filter(|&parent| !self.contains_region(parent))
                .map(RegionError::ParentMissing)
        });
        let cleanups = self.cleanups.iter().flat_map(|(handler, cleanup)| {
            let missing =
                (!self.contains_handler(handler)).then_some(RegionError::HandlerMissing(handler));
            let mut routes = BTreeSet::new();
            let repeated = cleanup
                .continuations
                .iter()
                .filter(move |&&route| !routes.insert(route))
                .map(move |&route| RegionError::ContinuationRepeated(handler, route));
            missing.into_iter().chain(repeated)
        });
        let blocks = self.dangling_anchors().map(RegionError::BlockNotLive);
        parents.chain(cleanups).chain(blocks)
    }

    /// Make `tail`, the second half of a split `head`, belong wherever `head`
    /// does: every set holding `head` gains `tail`, and a cleanup that exits
    /// from `head` now exits from `tail`, where `head`'s code ends.
    pub(crate) fn extend_exception_metadata_to_tail(&mut self, head: BlockId, tail: BlockId) {
        self.for_each_block_ref_mut(|reference| match reference {
            BlockRefMut::Set(blocks) => {
                if blocks.contains(&head) {
                    blocks.insert(tail);
                }
            }
            BlockRefMut::Anchor(block, role) => {
                if role.ends_a_body() && *block == head {
                    *block = tail;
                }
            }
        });
    }
}

#[cold]
#[track_caller]
fn missing_region(id: RegionId) -> ! {
    panic!("{id} is not a region of this CFG")
}

#[cold]
#[track_caller]
fn missing_handler(handler: HandlerRef) -> ! {
    panic!("{handler} is not a handler of this CFG")
}
