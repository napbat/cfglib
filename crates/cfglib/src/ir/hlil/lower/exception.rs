//! Statement lowering of `try`: the protected body, each handler, the finally
//! body, and the region that lists them.
//!
//! The parts of a `try` lower one after another, each as a body that
//! [`Lowerer::lower_body`] runs on its own stack. A [`TryScope`] carries what
//! the later parts need, and a [`TryStage`] says which part is running.

extern crate alloc;

use alloc::boxed::Box;
use alloc::collections::BTreeSet;
use alloc::format;
use alloc::vec::{self, Vec};

use crate::block::BlockId;
use crate::graph::view::DenseId;
use crate::region::{self, RegionId};

use super::super::flat_builder::{FlatBuilder, LowerEdge, Lowered, Occurrence};
use super::super::name::Name;
use super::super::{Dialect, Entity, FunctionView, Handler, HandlerKind, LowerError};
use super::control::{Step, Steps};
use super::{Lowerer, Sources, TryFrame};

/// A `try` whose parts are being lowered.
pub(super) struct TryScope<'a, H: Dialect, S: FunctionView<H>> {
    stmt: S::Stmt,
    finally_body: &'a [S::Stmt],
    /// The blocks that the protected body allocated are
    /// `created[protected_start..protected_end]`. The end is known once the
    /// body is lowered.
    protected_start: usize,
    protected_end: usize,
    /// The protected blocks that may throw, which unwind edges leave from.
    frame: TryFrame,
    /// The edges into what follows the `try`.
    normal: Sources<'a, H::Constant>,
    /// The handlers that are not lowered yet.
    handlers: vec::IntoIter<Handler<'a, H, S::Stmt, S::Var>>,
    /// The handlers lowered so far, which the region lists.
    lowered: Vec<region::Handler>,
}

/// The part of a `try` that is being lowered.
pub(super) enum TryStage<'a, H: Dialect, S: FunctionView<H>> {
    /// The protected body.
    Protected,
    /// The filter funclet of a handler, which has the blocks of its own.
    Filter {
        /// The body of the handler, which follows the filter.
        body: &'a [S::Stmt],
        entry: BlockId,
        filter_block: BlockId,
        /// The open block and pending edges of the handler entry, which the
        /// funclet leaves untouched.
        saved_current: Option<BlockId>,
        saved_pending: Sources<'a, H::Constant>,
    },
    /// The body of a handler.
    Handler {
        entry: BlockId,
        kind: region::HandlerKind,
        /// Where the blocks of the body start in `created`.
        body_start: usize,
    },
    /// The finally body.
    Finally {
        entry: BlockId,
        /// Where the blocks of the body start in `created`.
        start: usize,
    },
}

impl<'a, H, S, B> Lowerer<'a, H, S, B>
where
    H: Dialect,
    S: FunctionView<H>,
    B: FlatBuilder<H>,
{
    /// Starts the protected body of a `try`.
    pub(super) fn lower_try(
        &mut self,
        stmt: S::Stmt,
        body: &'a [S::Stmt],
        finally_body: &'a [S::Stmt],
        steps: &mut Steps<'a, H, S, B>,
    ) {
        self.seal(LowerEdge::Fallthrough);
        let protected_start = self.created.len();
        self.try_frames.push(TryFrame {
            may_throw_blocks: BTreeSet::new(),
        });
        let scope = Box::new(TryScope {
            stmt,
            finally_body,
            protected_start,
            protected_end: protected_start,
            frame: TryFrame {
                may_throw_blocks: BTreeSet::new(),
            },
            normal: Vec::new(),
            handlers: Vec::new().into_iter(),
            lowered: Vec::new(),
        });
        Self::enter(
            steps,
            body,
            Step::Try {
                scope,
                stage: TryStage::Protected,
            },
        );
    }

    /// Continues a `try` once the body of `stage` is lowered.
    pub(super) fn resume_try(
        &mut self,
        mut scope: Box<TryScope<'a, H, S>>,
        stage: TryStage<'a, H, S>,
        steps: &mut Steps<'a, H, S, B>,
    ) -> Result<(), LowerError<B::Error>> {
        match stage {
            TryStage::Protected => {
                scope.frame = self
                    .try_frames
                    .pop()
                    .expect("the try frame pushed by the try");
                scope.protected_end = self.created.len();
                scope.normal = self.take_continuation();
                let source = self.source;
                scope.handlers = source.handlers(scope.stmt).collect::<Vec<_>>().into_iter();
                self.next_handler(scope, steps)
            }
            TryStage::Filter {
                body,
                entry,
                filter_block,
                saved_current,
                saved_pending,
            } => {
                if self.current.is_some() || !self.pending.is_empty() {
                    return Err(LowerError::Unsupported(
                        "a filter body must end in a terminator".into(),
                    ));
                }
                self.current = saved_current;
                self.pending = saved_pending;
                let kind = region::HandlerKind::Filter { filter_block };
                self.lower_handler_body(scope, entry, kind, body, steps);
                Ok(())
            }
            TryStage::Handler {
                entry,
                kind,
                body_start,
            } => {
                scope.normal.extend(self.take_continuation());
                let blocks = self.blocks_from(body_start, entry);
                scope.lowered.push(region::Handler {
                    entry,
                    body: region::HandlerBody::Known(blocks),
                    kind,
                });
                self.next_handler(scope, steps)
            }
            TryStage::Finally { entry, start } => {
                let blocks = self.blocks_from(start, entry);
                scope.lowered.push(region::Handler {
                    entry,
                    body: region::HandlerBody::Known(blocks),
                    kind: region::HandlerKind::Finally,
                });
                self.register_region(*scope);
                Ok(())
            }
        }
    }

    /// The blocks allocated since `start`, and `entry`.
    fn blocks_from(&self, start: usize, entry: BlockId) -> BTreeSet<BlockId> {
        let mut blocks: BTreeSet<BlockId> = self.created[start..].iter().copied().collect();
        blocks.insert(entry);
        blocks
    }

    /// Lowers the next handler of the `try`, or the finally body once every
    /// handler is lowered.
    fn next_handler(
        &mut self,
        mut scope: Box<TryScope<'a, H, S>>,
        steps: &mut Steps<'a, H, S, B>,
    ) -> Result<(), LowerError<B::Error>> {
        let Some(handler) = scope.handlers.next() else {
            self.lower_finally(scope, steps);
            return Ok(());
        };
        let entry = self.new_block();
        for &block in &scope.frame.may_throw_blocks {
            self.builder.add_edge(block, entry, LowerEdge::Unwind);
        }
        self.current = Some(entry);
        self.pending.clear();

        if let Some(binding) = handler.binding {
            let ty = self.source.declared_type(binding).ok_or_else(|| {
                LowerError::Unsupported(format!(
                    "handler binding {} has no declared type",
                    Name::variable(binding)
                ))
            })?;
            let delivered = Occurrence {
                variable: self.variables[binding.index()],
                ty: ty.clone(),
            };
            self.emit(
                Lowered::CaughtException,
                &[],
                &[delivered],
                Entity::Statement(scope.stmt),
            )?;
        }

        let kind = match handler.kind {
            HandlerKind::Catch => region::HandlerKind::Catch,
            HandlerKind::CatchAll => region::HandlerKind::CatchAll,
            HandlerKind::Fault => region::HandlerKind::Fault,
            HandlerKind::Filter { filter_body } => {
                if filter_body.is_empty() {
                    return Err(LowerError::Unsupported(
                        "a filter handler needs a non-empty filter body".into(),
                    ));
                }
                // The funclet is runtime-invoked: lower it into blocks of its
                // own with no sequential continuation.
                let saved_current = self.current.take();
                let saved_pending = core::mem::take(&mut self.pending);
                let filter_block = self.new_block();
                self.current = Some(filter_block);
                let stage = TryStage::Filter {
                    body: handler.body,
                    entry,
                    filter_block,
                    saved_current,
                    saved_pending,
                };
                Self::enter(steps, filter_body, Step::Try { scope, stage });
                return Ok(());
            }
        };
        self.lower_handler_body(scope, entry, kind, handler.body, steps);
        Ok(())
    }

    /// Starts the body of the handler that enters at `entry`.
    fn lower_handler_body(
        &mut self,
        scope: Box<TryScope<'a, H, S>>,
        entry: BlockId,
        kind: region::HandlerKind,
        body: &'a [S::Stmt],
        steps: &mut Steps<'a, H, S, B>,
    ) {
        let stage = TryStage::Handler {
            entry,
            kind,
            body_start: self.created.len(),
        };
        Self::enter(steps, body, Step::Try { scope, stage });
    }

    /// Starts the finally body, which every normal and unwinding path of the
    /// `try` enters, or registers the region when there is none.
    fn lower_finally(
        &mut self,
        mut scope: Box<TryScope<'a, H, S>>,
        steps: &mut Steps<'a, H, S, B>,
    ) {
        if scope.finally_body.is_empty() {
            self.pending = core::mem::take(&mut scope.normal);
            self.register_region(*scope);
            return;
        }
        let entry = self.new_block();
        for &block in &scope.frame.may_throw_blocks {
            self.builder.add_edge(block, entry, LowerEdge::Unwind);
        }
        for (from, role) in core::mem::take(&mut scope.normal) {
            self.builder.add_edge(from, entry, role);
        }
        self.current = Some(entry);
        let stage = TryStage::Finally {
            entry,
            start: self.created.len(),
        };
        let finally_body = scope.finally_body;
        Self::enter(steps, finally_body, Step::Try { scope, stage });
    }

    /// Registers the region that protects the blocks of the `try` body, if it
    /// allocated any.
    fn register_region(&mut self, scope: TryScope<'a, H, S>) {
        let protected_blocks: BTreeSet<BlockId> = self.created
            [scope.protected_start..scope.protected_end]
            .iter()
            .copied()
            .collect();
        if !protected_blocks.is_empty() {
            self.builder.add_region(region::Region {
                id: RegionId::from_raw(0),
                protected_blocks,
                handlers: scope.lowered,
                parent: None,
            });
        }
    }
}
