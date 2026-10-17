//! Exception-region accessors of [`Cfg`].

extern crate alloc;

use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use crate::block::BlockId;
use crate::graph::traverse::{TraversalDirection, reachable};
use crate::region::{Handler, HandlerBody, HandlerRef, Region, RegionId};

use super::Cfg;

impl<I, E> Cfg<I, E> {
    /// All exception-handler regions.
    #[inline]
    #[must_use]
    pub fn regions(&self) -> &[Region] {
        &self.regions
    }

    /// Returns one exception region by its stable identity.
    #[must_use]
    pub fn region(&self, id: RegionId) -> Option<&Region> {
        self.regions.get(id.index())
    }

    /// Returns mutable access to one exception region by its stable identity.
    ///
    /// The region identity itself remains owned by this CFG; callers should not
    /// replace [`Region::id`] with a different value.
    pub fn region_mut(&mut self, id: RegionId) -> Option<&mut Region> {
        self.regions.get_mut(id.index())
    }

    /// Returns one handler by its stable region-and-position identity.
    #[must_use]
    pub fn handler(&self, handler: HandlerRef) -> Option<&Handler> {
        self.region(handler.region())?.handlers.get(handler.index())
    }

    /// Returns mutable access to one handler by its stable identity.
    pub fn handler_mut(&mut self, handler: HandlerRef) -> Option<&mut Handler> {
        self.region_mut(handler.region())?
            .handlers
            .get_mut(handler.index())
    }

    /// Removes one region, leaving an inert tombstone (empty protected
    /// set, no handlers) so later region identities stay stable.
    /// Structuring and fidelity reporting ignore inert regions.
    ///
    /// The cleanup records of the region's handlers go with it: a cleanup
    /// describes a handler, and a handler that no longer exists cannot keep
    /// the blocks it resumed from alive.
    pub fn remove_region(&mut self, id: RegionId) -> Option<Region> {
        let slot = self.regions.get_mut(id.index())?;
        let removed = core::mem::replace(
            slot,
            Region {
                id,
                protected_blocks: BTreeSet::new(),
                handlers: Vec::new(),
                parent: None,
            },
        );
        self.cleanups
            .retain(|cleanup| cleanup.handler.region() != id);
        Some(removed)
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
    /// use cfglib::{Cfg, Handler, HandlerBody, HandlerKind, Region, RegionId};
    ///
    /// let mut cfg = Cfg::<u32>::new();
    /// let body = cfg.entry();
    /// let handler = cfg.new_block();
    /// // The handler has no incoming edge: nothing in `body` can raise into it.
    /// cfg.add_region(Region {
    ///     id: RegionId::from_raw(0),
    ///     protected_blocks: [body].into_iter().collect(),
    ///     handlers: vec![Handler {
    ///         entry: handler,
    ///         body: HandlerBody::known([handler]),
    ///         kind: HandlerKind::CatchAll,
    ///     }],
    ///     parent: None,
    /// });
    ///
    /// assert_eq!(cfg.remove_unreachable_regions(), 1);
    /// assert!(cfg.protecting_region(body).is_none());
    /// ```
    pub fn remove_unreachable_regions(&mut self) -> usize {
        let reachable = reachable(self, [self.entry()], TraversalDirection::Outgoing);
        let doomed: Vec<RegionId> = self
            .regions
            .iter()
            .filter(|region| {
                !region.handlers.is_empty()
                    && region.handlers.iter().all(|handler| {
                        !reachable
                            .get(handler.entry.index())
                            .copied()
                            .unwrap_or(false)
                    })
            })
            .map(|region| region.id)
            .collect();
        let removed = doomed.len();
        for id in doomed {
            self.remove_region(id);
        }
        removed
    }

    /// Add a region and return its id.
    pub fn add_region(&mut self, mut region: Region) -> RegionId {
        let id = RegionId::from_index(self.regions.len());
        region.id = id;
        self.regions.push(region);
        id
    }

    /// Returns the innermost region that protects `block`, if any.
    #[must_use]
    pub fn protecting_region(&self, block: BlockId) -> Option<&Region> {
        // Return the deepest (last-added) region whose protected set
        // contains this block.
        self.regions
            .iter()
            .rev()
            .find(|r| r.protected_blocks.contains(&block))
    }

    /// Make `tail`, the second half of a split `head`, belong wherever `head`
    /// does. See [`split_block_with_payload`](Self::split_block_with_payload).
    pub(crate) fn extend_exception_metadata_to_tail(&mut self, head: BlockId, tail: BlockId) {
        for region in &mut self.regions {
            if region.protected_blocks.contains(&head) {
                region.protected_blocks.insert(tail);
            }
            for handler in &mut region.handlers {
                if let HandlerBody::Known(blocks) = &mut handler.body
                    && blocks.contains(&head)
                {
                    blocks.insert(tail);
                }
            }
        }
        for cleanup in &mut self.cleanups {
            if cleanup.resume_from == Some(head) {
                cleanup.resume_from = Some(tail);
            }
        }
    }
}
