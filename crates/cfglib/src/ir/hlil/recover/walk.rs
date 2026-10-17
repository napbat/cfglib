//! The statement walk of the rebuild.
//!
//! The walk keeps its own stacks, so the nesting depth of a function costs no
//! call frames. A body that is being rebuilt and a construct that waits for
//! its child bodies are frames of one explicit stack. The rebuilt statements
//! of every open body sit on the statement stack, and a construct hands the
//! builder the slices that hold its children, then replaces them with its own
//! statement. The builder therefore sees its calls in the order of the
//! recursive formulation: operands before their statement, the children of a
//! construct before the construct, and the bodies of a construct in source
//! order.

extern crate alloc;

use alloc::vec::Vec;

use crate::graph::view::DenseId;

use super::super::shape::{Handler, HandlerKind, SwitchArm};
use super::region::Copies;
use super::{Fallible, FunctionBuilder, FunctionView, Rebuilder, RecoverDialect, StmtShape};

/// One step of the rebuild that waits for the frames above it.
pub(super) enum Frame<'a, H, S, B>
where
    H: RecoverDialect,
    S: FunctionView<H>,
    B: FunctionBuilder<H>,
{
    /// The statements `ids[index..]` of one body, with the variable copies
    /// among the statements before `index`.
    Body {
        ids: &'a [S::Stmt],
        index: usize,
        copies: Copies<S::Var>,
    },
    /// A construct that waits for its child bodies.
    Open(Open<'a, H, S, B>),
}

/// A construct whose child bodies are being rebuilt.
pub(super) struct Open<'a, H, S, B>
where
    H: RecoverDialect,
    S: FunctionView<H>,
    B: FunctionBuilder<H>,
{
    kind: Kind<'a, H, S, B>,
    /// The height of the statement stack where its first body begins.
    first: usize,
    /// The number of its bodies that have been started, and one more once
    /// every body has finished.
    started: usize,
}

/// One handler of a try that is being rebuilt, with its binding already
/// resolved.
pub(super) struct OpenHandler<'a, H, S, B>
where
    H: RecoverDialect,
    S: FunctionView<H>,
    B: FunctionBuilder<H>,
{
    handler: Handler<'a, H, S::Stmt, S::Var>,
    binding: Option<B::Var>,
}

/// What a construct needs, besides its bodies, to become a statement. The
/// bodies are listed in the order that they rebuild.
pub(super) enum Kind<'a, H, S, B>
where
    H: RecoverDialect,
    S: FunctionView<H>,
    B: FunctionBuilder<H>,
{
    If {
        id: S::Stmt,
        condition: B::Expr,
        then_body: &'a [S::Stmt],
        else_body: &'a [S::Stmt],
    },
    While {
        id: S::Stmt,
        condition: B::Expr,
        body: &'a [S::Stmt],
    },
    /// A post-tested loop, whose condition rebuilds after its body.
    DoWhile {
        id: S::Stmt,
        body: &'a [S::Stmt],
        condition: S::Expr,
    },
    Loop {
        id: S::Stmt,
        body: &'a [S::Stmt],
    },
    /// The initializer, the update, then the body. The condition rebuilds
    /// between the initializer and the update.
    For {
        id: S::Stmt,
        initializer: &'a [S::Stmt],
        condition: Option<S::Expr>,
        rebuilt_condition: Option<B::Expr>,
        update: &'a [S::Stmt],
        body: &'a [S::Stmt],
    },
    Switch {
        id: S::Stmt,
        scrutinee: B::Expr,
        arms: Vec<SwitchArm<'a, H, S::Stmt>>,
        default_body: &'a [S::Stmt],
    },
    Try {
        id: S::Stmt,
        /// The try body, the filter body and the body of each handler, and
        /// the finally body.
        bodies: Vec<&'a [S::Stmt]>,
        handlers: Vec<OpenHandler<'a, H, S, B>>,
    },
    Labeled {
        id: S::Stmt,
        label: &'a str,
        body: &'a [S::Stmt],
    },
    /// A region statement of the source, whose operands wait on the
    /// expression stack from `operands_start`.
    Region {
        id: S::Stmt,
        op: &'a H::Op,
        operands_start: usize,
        body: &'a [S::Stmt],
    },
    /// An initializer and a `while` that recover as a `for`.
    Counted {
        loop_id: S::Stmt,
        initializer: B::Stmt,
        condition: B::Expr,
        /// The trailing assignment of the loop body, which becomes the
        /// update: its statement and the two sides of the assignment.
        update: (S::Stmt, S::Expr, S::Expr),
        body: &'a [S::Stmt],
    },
    /// An enter and the `try` that guards its body, which recover as a
    /// region.
    Guarded {
        enter_id: S::Stmt,
        guarded_id: S::Stmt,
        /// The statements of the cleanup handler, which the region
        /// regenerates.
        cleanup: &'a [S::Stmt],
        subject: B::Expr,
        region_op: H::Op,
        body: &'a [S::Stmt],
    },
}

impl<'a, H, S, B> Rebuilder<'a, H, S, B>
where
    H: RecoverDialect,
    S: FunctionView<H>,
    B: FunctionBuilder<H>,
{
    /// Rebuilds the statements of `ids` onto the statement stack, where they
    /// stay.
    pub(super) fn rebuild_root(&mut self, ids: &'a [S::Stmt]) -> Fallible<(), B::Error> {
        self.frames.push(Frame::Body {
            ids,
            index: 0,
            copies: Copies::new(),
        });
        while let Some(frame) = self.frames.pop() {
            match frame {
                Frame::Body { ids, index, copies } => self.step_body(ids, index, copies)?,
                Frame::Open(mut open) => {
                    if let Some(body) = self.next_body(&mut open)? {
                        self.frames.push(Frame::Open(open));
                        self.frames.push(Frame::Body {
                            ids: body,
                            index: 0,
                            copies: Copies::new(),
                        });
                    } else {
                        self.finish(open)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Rebuilds the statements of one body from `index` on, up to the first
    /// construct that has bodies of its own. That construct becomes a frame
    /// above the rest of this body.
    fn step_body(
        &mut self,
        ids: &'a [S::Stmt],
        mut index: usize,
        mut copies: Copies<S::Var>,
    ) -> Fallible<(), B::Error> {
        while index < ids.len() {
            let id = ids[index];
            if self.suppress_region_exit(id)? {
                index += 1;
                continue;
            }
            self.track_region_aliases(id)?;
            let paired = if index + 1 < ids.len() {
                match self.try_region(ids, index, &mut copies)? {
                    Some(open) => Some(open),
                    None => self.try_for(ids, index)?,
                }
            } else {
                None
            };
            let opened = if paired.is_some() {
                index += 2;
                paired
            } else {
                index += 1;
                self.open_statement(id)?
            };
            if let Some(open) = opened {
                self.frames.push(Frame::Body { ids, index, copies });
                self.frames.push(Frame::Open(open));
                return Ok(());
            }
        }
        Ok(())
    }

    /// A construct whose first body starts on top of the statement stack.
    pub(super) fn open(&self, kind: Kind<'a, H, S, B>) -> Open<'a, H, S, B> {
        Open {
            kind,
            first: self.statement_stack.len(),
            started: 0,
        }
    }

    /// Rebuilds one statement onto the statement stack, or returns the
    /// construct of a statement that has bodies to rebuild first.
    fn open_statement(&mut self, id: S::Stmt) -> Fallible<Option<Open<'a, H, S, B>>, B::Error> {
        let source: &'a S = self.source;
        let rebuilt = match self.stmt(id)? {
            StmtShape::Expression(expression) => {
                let expression = self.rebuild_expression(expression)?;
                self.builder.expression_statement(expression)
            }
            StmtShape::Assign { target, value } => self.rebuild_assign(target, value)?,
            StmtShape::If {
                condition,
                then_body,
                else_body,
            } => {
                if let Some(select) = self.try_select(id, condition, then_body, else_body)? {
                    self.statement_stack.push(select);
                    return Ok(None);
                }
                let condition = self.rebuild_expression(condition)?;
                return Ok(Some(self.open(Kind::If {
                    id,
                    condition,
                    then_body,
                    else_body,
                })));
            }
            StmtShape::While { condition, body } => {
                let condition = self.rebuild_expression(condition)?;
                return Ok(Some(self.open(Kind::While {
                    id,
                    condition,
                    body,
                })));
            }
            StmtShape::DoWhile { body, condition } => {
                return Ok(Some(self.open(Kind::DoWhile {
                    id,
                    body,
                    condition,
                })));
            }
            StmtShape::Loop { body } => return Ok(Some(self.open(Kind::Loop { id, body }))),
            StmtShape::For {
                initializer,
                condition,
                update,
                body,
            } => {
                return Ok(Some(self.open(Kind::For {
                    id,
                    initializer,
                    condition,
                    rebuilt_condition: None,
                    update,
                    body,
                })));
            }
            StmtShape::Switch {
                scrutinee,
                default_body,
            } => {
                let scrutinee = self.rebuild_expression(scrutinee)?;
                return Ok(Some(self.open(Kind::Switch {
                    id,
                    scrutinee,
                    arms: source.switch_arms(id).collect(),
                    default_body,
                })));
            }
            StmtShape::Break { label } => self.builder.break_(label),
            StmtShape::Continue { label } => self.builder.continue_(label),
            StmtShape::Return { values } => {
                let start = self.rebuild_operands(values)?;
                let rebuilt = self.builder.return_(&self.expression_stack[start..]);
                self.expression_stack.truncate(start);
                rebuilt
            }
            StmtShape::Labeled { label, body } => {
                return Ok(Some(self.open(Kind::Labeled { id, label, body })));
            }
            StmtShape::Goto { label } => self.builder.goto(label),
            StmtShape::Try { body, finally_body } => {
                return self.open_try(id, body, finally_body).map(Some);
            }
            StmtShape::Region { op, operands, body } => {
                let operands_start = self.rebuild_operands(operands)?;
                return Ok(Some(self.open(Kind::Region {
                    id,
                    op,
                    operands_start,
                    body,
                })));
            }
        };
        self.statements[id.index()] = Some(rebuilt);
        self.statement_stack.push(rebuilt);
        Ok(None)
    }

    /// The construct of a `try`, with the binding of every handler resolved.
    fn open_try(
        &mut self,
        id: S::Stmt,
        body: &'a [S::Stmt],
        finally_body: &'a [S::Stmt],
    ) -> Fallible<Open<'a, H, S, B>, B::Error> {
        let source: &'a S = self.source;
        let mut bodies = Vec::from([body]);
        let mut handlers = Vec::new();
        for handler in source.handlers(id) {
            if let HandlerKind::Filter { filter_body } = handler.kind {
                bodies.push(filter_body);
            }
            bodies.push(handler.body);
            let binding = handler
                .binding
                .map(|binding| self.variable(binding))
                .transpose()?;
            handlers.push(OpenHandler { handler, binding });
        }
        bodies.push(finally_body);
        Ok(self.open(Kind::Try {
            id,
            bodies,
            handlers,
        }))
    }

    /// Rebuilds an assignment of source expressions.
    pub(super) fn rebuild_assign(
        &mut self,
        target: S::Expr,
        value: S::Expr,
    ) -> Fallible<B::Stmt, B::Error> {
        let target = self.rebuild_expression(target)?;
        let value = self.rebuild_expression(value)?;
        Ok(self.builder.assign(target, value))
    }

    /// Starts the next body of `open`, or returns `None` when every body has
    /// finished. A body that starts after the first records where the one
    /// before it ended.
    fn next_body(
        &mut self,
        open: &mut Open<'a, H, S, B>,
    ) -> Fallible<Option<&'a [S::Stmt]>, B::Error> {
        let stage = open.started;
        open.started += 1;
        if stage > 0 {
            self.ends.push(self.statement_stack.len());
        }
        Ok(match &mut open.kind {
            Kind::If {
                then_body,
                else_body,
                ..
            } => match stage {
                0 => Some(*then_body),
                1 => Some(*else_body),
                _ => None,
            },
            Kind::While { body, .. }
            | Kind::DoWhile { body, .. }
            | Kind::Loop { body, .. }
            | Kind::Labeled { body, .. }
            | Kind::Region { body, .. }
            | Kind::Counted { body, .. }
            | Kind::Guarded { body, .. } => (stage == 0).then_some(*body),
            Kind::For {
                initializer,
                condition,
                rebuilt_condition,
                update,
                body,
                ..
            } => match stage {
                0 => Some(*initializer),
                1 => {
                    *rebuilt_condition = condition
                        .map(|condition| self.rebuild_expression(condition))
                        .transpose()?;
                    Some(*update)
                }
                2 => Some(*body),
                _ => None,
            },
            Kind::Switch {
                arms, default_body, ..
            } => match arms.get(stage) {
                Some(arm) => Some(arm.body),
                None => (stage == arms.len()).then_some(*default_body),
            },
            Kind::Try { bodies, .. } => bodies.get(stage).copied(),
        })
    }

    /// Turns a construct whose bodies have all finished into its statement,
    /// in place of them on the statement stack. The statement is also what
    /// the source statement of the construct maps to.
    fn finish(&mut self, open: Open<'a, H, S, B>) -> Fallible<(), B::Error> {
        let Open {
            kind,
            first,
            started,
        } = open;
        // The final call of `next_body` recorded the end of the last body.
        let base = self.ends.len() - (started - 1);
        let (id, rebuilt) = match kind {
            Kind::If { id, condition, .. } => {
                let (then_end, else_end) = (self.ends[base], self.ends[base + 1]);
                let stack = &self.statement_stack;
                let rebuilt = self.builder.if_(
                    condition,
                    &stack[first..then_end],
                    &stack[then_end..else_end],
                );
                (id, rebuilt)
            }
            Kind::While { id, condition, .. } => {
                let stack = &self.statement_stack;
                (id, self.builder.while_(condition, &stack[first..]))
            }
            Kind::DoWhile { id, condition, .. } => {
                let condition = self.rebuild_expression(condition)?;
                let stack = &self.statement_stack;
                (id, self.builder.do_while(&stack[first..], condition))
            }
            Kind::Loop { id, .. } => (id, self.builder.loop_(&self.statement_stack[first..])),
            Kind::For {
                id,
                rebuilt_condition,
                ..
            } => {
                let (initializer_end, update_end) = (self.ends[base], self.ends[base + 1]);
                let stack = &self.statement_stack;
                let rebuilt = self.builder.for_(
                    &stack[first..initializer_end],
                    rebuilt_condition,
                    &stack[initializer_end..update_end],
                    &stack[update_end..],
                );
                (id, rebuilt)
            }
            Kind::Switch {
                id,
                scrutinee,
                arms,
                ..
            } => (id, self.finish_switch(scrutinee, &arms, first, base)),
            Kind::Try { id, handlers, .. } => (id, self.finish_try(&handlers, first, base)),
            Kind::Labeled { id, label, .. } => {
                let stack = &self.statement_stack;
                (id, self.builder.labeled(label, &stack[first..]))
            }
            Kind::Region {
                id,
                op,
                operands_start,
                ..
            } => {
                let rebuilt = self.builder.region(
                    op,
                    &self.expression_stack[operands_start..],
                    &self.statement_stack[first..],
                );
                self.expression_stack.truncate(operands_start);
                (id, rebuilt)
            }
            Kind::Counted {
                loop_id,
                initializer,
                condition,
                update: (update_id, update_target, update_value),
                ..
            } => {
                let update = self.rebuild_assign(update_target, update_value)?;
                self.statements[update_id.index()] = Some(update);
                let stack = &self.statement_stack;
                let counted =
                    self.builder
                        .for_(&[initializer], Some(condition), &[update], &stack[first..]);
                self.for_loops += 1;
                (loop_id, counted)
            }
            Kind::Guarded {
                enter_id,
                guarded_id,
                cleanup,
                subject,
                region_op,
                ..
            } => {
                let region = self.finish_guarded(guarded_id, cleanup, subject, &region_op, first);
                (enter_id, region)
            }
        };
        self.statements[id.index()] = Some(rebuilt);
        self.ends.truncate(base);
        self.statement_stack.truncate(first);
        self.statement_stack.push(rebuilt);
        Ok(())
    }

    /// The switch whose arm bodies sit on the statement stack from `first`,
    /// the end of each after `base` on the stack of ends, and then the
    /// default body.
    fn finish_switch(
        &mut self,
        scrutinee: B::Expr,
        arms: &[SwitchArm<'a, H, S::Stmt>],
        first: usize,
        base: usize,
    ) -> B::Stmt {
        let ends = &self.ends[base..];
        let stack = &self.statement_stack;
        let mut start = first;
        let mut rebuilt_arms: Vec<SwitchArm<'_, H, B::Stmt>> = Vec::with_capacity(arms.len());
        for (arm, &end) in arms.iter().zip(ends) {
            rebuilt_arms.push(SwitchArm {
                values: arm.values,
                body: &stack[start..end],
            });
            start = end;
        }
        let default_body = &stack[start..ends[arms.len()]];
        self.builder.switch(scrutinee, &rebuilt_arms, default_body)
    }

    /// The try whose bodies sit on the statement stack from `first`, the end
    /// of each after `base` on the stack of ends: the try body, the filter
    /// body and the body of each handler, and the finally body.
    fn finish_try(
        &mut self,
        handlers: &[OpenHandler<'a, H, S, B>],
        first: usize,
        base: usize,
    ) -> B::Stmt {
        let stack = &self.statement_stack;
        let mut ends = self.ends[base..].iter().copied();
        let mut start = first;
        let mut take_body = || {
            let end = ends.next().expect("every body of a try has finished");
            let body = &stack[start..end];
            start = end;
            body
        };
        let body = take_body();
        let rebuilt_handlers: Vec<Handler<'_, H, B::Stmt, B::Var>> = handlers
            .iter()
            .map(|open| {
                let kind = match open.handler.kind {
                    HandlerKind::Catch => HandlerKind::Catch,
                    HandlerKind::CatchAll => HandlerKind::CatchAll,
                    HandlerKind::Fault => HandlerKind::Fault,
                    HandlerKind::Filter { .. } => HandlerKind::Filter {
                        filter_body: take_body(),
                    },
                };
                Handler {
                    kind,
                    binding: open.binding,
                    caught_types: open.handler.caught_types,
                    body: take_body(),
                }
            })
            .collect();
        let finally_body = take_body();
        self.builder.try_(body, &rebuilt_handlers, finally_body)
    }

    /// The region that replaces a guarded body: it re-owns the guarding
    /// `try`, its cleanup, and every exit that was suppressed, besides the
    /// enter that the caller maps.
    fn finish_guarded(
        &mut self,
        guarded_id: S::Stmt,
        cleanup: &[S::Stmt],
        subject: B::Expr,
        region_op: &H::Op,
        first: usize,
    ) -> B::Stmt {
        let active = self
            .active_regions
            .pop()
            .expect("the region pushed on opening is still active");
        let stack = &self.statement_stack;
        let region = self.builder.region(region_op, &[subject], &stack[first..]);
        self.statements[guarded_id.index()] = Some(region);
        for &statement in cleanup {
            self.statements[statement.index()] = Some(region);
        }
        for exit in active.exits {
            self.statements[exit.index()] = Some(region);
        }
        self.regions += 1;
        region
    }
}
