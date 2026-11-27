//! Cleanup-record and continuation accessors of [`Cfg`].

use crate::block::BlockId;
use crate::region::{Cleanup, Continuation, HandlerMetadata, HandlerRef, RegionError};

use super::Cfg;

impl<I, E> Cfg<I, E> {
    /// Every recorded [`Cleanup`], keyed by its handler, in handler order.
    #[inline]
    #[must_use]
    pub fn cleanups(&self) -> &HandlerMetadata<Cleanup> {
        &self.cleanups
    }

    /// The cleanup record of `handler`, if it has one.
    #[must_use]
    pub fn cleanup(&self, handler: HandlerRef) -> Option<&Cleanup> {
        self.cleanups.metadata(handler)
    }

    /// Record one route out of a cleanup handler: where control continues
    /// once the cleanup body ends, and the reason that entered it.
    ///
    /// The first call for a handler creates its record. Recording the same
    /// `(reason, target)` pair twice is a no-op, so a lowering that walks
    /// several transfers into one cleanup keeps one route per distinct
    /// destination, in first-recorded order.
    ///
    /// # Examples
    ///
    /// A `try { ... } finally { ... }` whose body both falls out normally and
    /// `return`s: one cleanup block, two routes, told apart by reason.
    ///
    /// ```
    /// use cfglib::{
    ///     Cfg, CompletionReason, Continuation, Handler, HandlerBody, HandlerKind, HandlerRef,
    ///     Region,
    /// };
    ///
    /// let mut cfg = Cfg::<&'static str>::new();
    /// let cleanup_block = cfg.new_block();
    /// let after = cfg.new_block();
    /// let exit = cfg.new_block();
    ///
    /// let region = cfg.add_region(Region {
    ///     protected_blocks: [cfg.entry()].into_iter().collect(),
    ///     handlers: vec![Handler {
    ///         entry: cleanup_block,
    ///         body: HandlerBody::known([cleanup_block]),
    ///         kind: HandlerKind::Finally,
    ///     }],
    ///     parent: None,
    /// })?;
    ///
    /// let handler = HandlerRef::new(region, 0);
    /// cfg.set_cleanup_exit(handler, cleanup_block)?;
    /// cfg.add_continuation(handler, Continuation {
    ///     reason: CompletionReason::Normal,
    ///     target: after,
    /// })?;
    /// cfg.add_continuation(handler, Continuation {
    ///     reason: CompletionReason::Return,
    ///     target: exit,
    /// })?;
    ///
    /// // Both routes leave the same block, and each one is identifiable.
    /// let recorded = cfg.cleanup(handler).unwrap();
    /// assert_eq!(recorded.exit_block, Some(cleanup_block));
    /// assert_eq!(
    ///     recorded.targets_for(CompletionReason::Return).collect::<Vec<_>>(),
    ///     vec![exit]
    /// );
    /// # Ok::<(), cfglib::RegionError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`RegionError::HandlerMissing`] if `handler` names no handler
    /// of this CFG — register the region first, then attach its routes — and
    /// [`RegionError::BlockNotLive`] if the target names no live block. The
    /// CFG is unchanged then.
    pub fn add_continuation(
        &mut self,
        handler: HandlerRef,
        continuation: Continuation,
    ) -> Result<(), RegionError> {
        self.check_cleanup_reference(handler, continuation.target)?;
        let cleanup = self
            .cleanups
            .metadata_or_insert_with(handler, Cleanup::default);
        if !cleanup.continuations.contains(&continuation) {
            cleanup.continuations.push(continuation);
        }
        Ok(())
    }

    /// Record the block a cleanup handler's body ends in — the block every
    /// continuation edge leaves from.
    ///
    /// A cleanup that diverges never reaches one, so leaving it unset is the
    /// honest description of a `finally` that returns.
    ///
    /// # Errors
    ///
    /// Returns [`RegionError::HandlerMissing`] if `handler` names no handler
    /// of this CFG, and [`RegionError::BlockNotLive`] if `exit_block` names no
    /// live block. The CFG is unchanged then.
    pub fn set_cleanup_exit(
        &mut self,
        handler: HandlerRef,
        exit_block: BlockId,
    ) -> Result<(), RegionError> {
        self.check_cleanup_reference(handler, exit_block)?;
        self.cleanups
            .metadata_or_insert_with(handler, Cleanup::default)
            .exit_block = Some(exit_block);
        Ok(())
    }

    /// A cleanup record may be attached only to a handler of this CFG, and
    /// may name only live blocks.
    fn check_cleanup_reference(
        &self,
        handler: HandlerRef,
        block: BlockId,
    ) -> Result<(), RegionError> {
        if !self.contains_handler(handler) {
            return Err(RegionError::HandlerMissing(handler));
        }
        if !self.contains_block(block) {
            return Err(RegionError::BlockNotLive(block));
        }
        Ok(())
    }
}
