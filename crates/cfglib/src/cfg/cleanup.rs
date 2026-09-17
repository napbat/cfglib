//! Cleanup-record and continuation accessors of [`Cfg`].

extern crate alloc;

use alloc::vec::Vec;

use crate::block::BlockId;
use crate::region::{Cleanup, Continuation, HandlerRef};

use super::Cfg;

impl<I, E> Cfg<I, E> {
    /// Every recorded [`Cleanup`], in the order the handlers first recorded
    /// one.
    #[inline]
    #[must_use]
    pub fn cleanups(&self) -> &[Cleanup] {
        &self.cleanups
    }

    /// The cleanup record of `handler`, if it has one.
    #[must_use]
    pub fn cleanup(&self, handler: HandlerRef) -> Option<&Cleanup> {
        self.cleanups
            .iter()
            .find(|cleanup| cleanup.handler == handler)
    }

    /// Record one route out of a cleanup handler: where control resumes once
    /// the cleanup body ends, and the reason that entered it.
    ///
    /// The first call for a handler creates its record. Recording the same
    /// `(reason, resume)` pair twice is a no-op, so a lowering that walks
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
    ///     Cfg, CompletionReason, Continuation, EhModel, Handler, HandlerBody, HandlerKind,
    ///     HandlerRef, Region, RegionId,
    /// };
    ///
    /// let mut cfg = Cfg::<&'static str>::new();
    /// let cleanup_block = cfg.new_block();
    /// let after = cfg.new_block();
    /// let exit = cfg.new_block();
    ///
    /// let region = cfg.add_region(Region {
    ///     id: RegionId::from_raw(0), // overwritten by `add_region`
    ///     protected_blocks: [cfg.entry()].into_iter().collect(),
    ///     handlers: vec![Handler {
    ///         entry: cleanup_block,
    ///         body: HandlerBody::known([cleanup_block]),
    ///         kind: HandlerKind::Finally,
    ///     }],
    ///     parent: None,
    /// });
    ///
    /// let handler = HandlerRef::new(region, 0);
    /// cfg.set_cleanup_resume(handler, cleanup_block);
    /// cfg.add_continuation(handler, Continuation {
    ///     reason: CompletionReason::Normal,
    ///     resume: after,
    /// });
    /// cfg.add_continuation(handler, Continuation {
    ///     reason: CompletionReason::Return,
    ///     resume: exit,
    /// });
    ///
    /// // Both routes leave the same block, and each one is identifiable.
    /// let model = EhModel::compute(&cfg);
    /// let recorded = model.cleanup(cleanup_block).unwrap();
    /// assert_eq!(recorded.resume_from, Some(cleanup_block));
    /// assert_eq!(
    ///     recorded.resumes_for(CompletionReason::Return).collect::<Vec<_>>(),
    ///     vec![exit]
    /// );
    /// ```
    ///
    /// # Panics
    ///
    /// Panics (debug) if `handler` does not refer to a handler of a region in
    /// this CFG — register the region first, then attach its routes.
    pub fn add_continuation(&mut self, handler: HandlerRef, continuation: Continuation) {
        debug_assert!(
            self.handler_exists(handler),
            "handler does not exist in this CFG"
        );
        let cleanup = self.cleanup_entry(handler);
        if !cleanup.continuations.contains(&continuation) {
            cleanup.continuations.push(continuation);
        }
    }

    /// Record the block a cleanup handler's body ends in — the block every
    /// continuation edge leaves from.
    ///
    /// A cleanup that diverges never reaches one, so leaving it unset is the
    /// honest description of a `finally` that returns.
    ///
    /// # Panics
    ///
    /// Panics (debug) if `handler` does not refer to a handler of a region in
    /// this CFG, or if `resume_from` does not refer to a block in it.
    pub fn set_cleanup_resume(&mut self, handler: HandlerRef, resume_from: BlockId) {
        debug_assert!(
            self.handler_exists(handler),
            "handler does not exist in this CFG"
        );
        debug_assert!(
            self.contains_block(resume_from),
            "resume block does not exist in this CFG"
        );
        self.cleanup_entry(handler).resume_from = Some(resume_from);
    }

    /// The cleanup record of `handler`, created empty when it is the first
    /// route recorded for it.
    fn cleanup_entry(&mut self, handler: HandlerRef) -> &mut Cleanup {
        let existing = self
            .cleanups
            .iter()
            .position(|cleanup| cleanup.handler == handler);
        let at = existing.unwrap_or_else(|| {
            self.cleanups.push(Cleanup {
                handler,
                resume_from: None,
                continuations: Vec::new(),
            });
            self.cleanups.len() - 1
        });
        &mut self.cleanups[at]
    }

    /// Whether `handler` refers to a handler of a region in this CFG.
    fn handler_exists(&self, handler: HandlerRef) -> bool {
        self.regions
            .get(handler.region().index())
            .is_some_and(|region| handler.index() < region.handlers.len())
    }
}
