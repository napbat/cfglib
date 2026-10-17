//! Statement lowering: blocks, joins, loops, switches, and regions.
//!
//! Control flow lowers through lazy join materialization: ending a construct
//! leaves its continuation as pending edges, and the next emitted
//! instruction materializes the join block and wires them. A path that
//! terminates (return, goto, break) simply leaves nothing pending.

extern crate alloc;

use alloc::collections::BTreeSet;
use alloc::format;
use alloc::vec::Vec;

use crate::block::BlockId;
use crate::graph::view::DenseId;
use crate::region::{self, RegionId};

use super::super::flat_builder::{FlatBuilder, LowerEdge, Lowered, Occurrence};
use super::super::name::Name;
use super::super::transfer::Untargeted;
use super::super::view::labeled_loop;
use super::super::{
    Dialect, Entity, ExprShape, FunctionView, Handler, HandlerKind, LowerError, StmtShape,
};
use super::{Frame, FrameKind, Lowerer, Sources, TryFrame};

impl<'a, H, S, B> Lowerer<'a, H, S, B>
where
    H: Dialect,
    S: FunctionView<H>,
    B: FlatBuilder<H>,
{
    pub(super) fn lower_body(&mut self, body: &'a [S::Stmt]) -> Result<(), LowerError<B::Error>> {
        for &stmt in body {
            self.lower_statement(stmt)?;
        }
        Ok(())
    }

    fn lower_statement(&mut self, stmt: S::Stmt) -> Result<(), LowerError<B::Error>> {
        match self.source.stmt(stmt) {
            StmtShape::Expression(expr) => self.lower_expression_statement(expr),
            StmtShape::Assign { target, value } => self.lower_assign(stmt, target, value),
            StmtShape::If {
                condition,
                then_body,
                else_body,
            } => self.lower_if(stmt, condition, then_body, else_body),
            StmtShape::While { .. }
            | StmtShape::DoWhile { .. }
            | StmtShape::Loop { .. }
            | StmtShape::For { .. } => self.lower_loop_statement(stmt, None),
            StmtShape::Switch {
                scrutinee,
                default_body,
            } => self.lower_switch(stmt, scrutinee, default_body),
            StmtShape::Break { label } => self.lower_break(stmt, label),
            StmtShape::Continue { label } => self.lower_continue(stmt, label),
            StmtShape::Return { values } => {
                let uses = self.flatten_all(values)?;
                self.emit(Lowered::Return, &uses, &[], Entity::Statement(stmt))?;
                self.current = None;
                Ok(())
            }
            StmtShape::Labeled { label, body } => self.lower_labeled(label, body),
            StmtShape::Goto { label } => {
                self.lower_goto(label);
                Ok(())
            }
            StmtShape::Try { body, finally_body } => self.lower_try(stmt, body, finally_body),
            StmtShape::Region { op, operands, body } => self.lower_region(stmt, op, operands, body),
        }
    }

    fn lower_expression_statement(&mut self, expr: S::Expr) -> Result<(), LowerError<B::Error>> {
        if let ExprShape::Operation { op, operands } = self.source.expr(expr) {
            let uses = self.flatten_all(operands)?;
            self.emit(Lowered::Operation(op), &uses, &[], Entity::Expression(expr))?;
        }
        // A bare variable or constant evaluated for effect lowers to nothing.
        Ok(())
    }

    fn lower_assign(
        &mut self,
        stmt: S::Stmt,
        target: S::Expr,
        value: S::Expr,
    ) -> Result<(), LowerError<B::Error>> {
        let origin = Entity::Statement(stmt);
        match self.source.expr(target) {
            ExprShape::Variable(variable) => {
                let target = self.read(target, variable);
                self.flatten_into(&target, value, origin)
            }
            ExprShape::Operation { op, operands } => {
                let mut uses = self.flatten_all(operands)?;
                uses.push(self.flatten(value)?);
                self.emit(Lowered::Store { location: op }, &uses, &[], origin)
            }
            ExprShape::Constant(_) => Err(LowerError::Unsupported(
                "assignment into a constant place".into(),
            )),
        }
    }

    /// Flattens `condition` and ends the open block with a branch on it.
    fn emit_branch(
        &mut self,
        stmt: S::Stmt,
        condition: S::Expr,
    ) -> Result<BlockId, LowerError<B::Error>> {
        let condition = self.flatten(condition)?;
        self.emit(Lowered::Branch, &[condition], &[], Entity::Statement(stmt))?;
        Ok(self
            .current
            .take()
            .expect("the branch instruction opened a block"))
    }

    /// The open continuation as join sources: pending edges plus the open
    /// block's sequential fall-out.
    fn take_continuation(&mut self) -> Sources<'a, H::Constant> {
        let mut sources = core::mem::take(&mut self.pending);
        if let Some(block) = self.current.take() {
            sources.push((block, LowerEdge::Fallthrough));
        }
        sources
    }

    /// The open continuation as explicit-transfer sources (breaks, gotos).
    fn take_transfer(&mut self) -> Sources<'a, H::Constant> {
        let mut sources = core::mem::take(&mut self.pending);
        if let Some(block) = self.current.take() {
            sources.push((block, LowerEdge::Jump));
        }
        sources
    }

    fn lower_if(
        &mut self,
        stmt: S::Stmt,
        condition: S::Expr,
        then_body: &'a [S::Stmt],
        else_body: &'a [S::Stmt],
    ) -> Result<(), LowerError<B::Error>> {
        let branch = self.emit_branch(stmt, condition)?;

        self.pending = alloc::vec![(branch, LowerEdge::True)];
        self.lower_body(then_body)?;
        let mut join = self.take_continuation();

        self.pending = alloc::vec![(branch, LowerEdge::False)];
        self.lower_body(else_body)?;
        join.extend(self.take_continuation());

        self.pending = join;
        Ok(())
    }

    /// Registers a loop's label: forward gotos already flowed into the entry
    /// through the pending list; later gotos resolve to `entry`.
    fn register_loop_label(&mut self, label: Option<&'a str>, entry: BlockId) {
        if let Some(label) = label {
            self.labels.insert(label, entry);
        }
    }

    fn adopt_forward_gotos(&mut self, label: Option<&'a str>) {
        if let Some(label) = label
            && let Some(waiting) = self.forward_gotos.remove(label)
        {
            self.pending.extend(waiting);
        }
    }

    fn push_loop_frame(&mut self, label: Option<&'a str>, continue_target: BlockId) {
        self.frames.push(Frame {
            label,
            kind: FrameKind::Loop { continue_target },
            break_sources: Vec::new(),
        });
    }

    fn pop_frame(&mut self) -> Frame<'a, H::Constant> {
        self.frames
            .pop()
            .expect("the frame pushed by the construct")
    }

    pub(super) fn lower_while(
        &mut self,
        stmt: S::Stmt,
        condition: S::Expr,
        body: &'a [S::Stmt],
        label: Option<&'a str>,
    ) -> Result<(), LowerError<B::Error>> {
        self.seal(LowerEdge::Fallthrough);
        self.adopt_forward_gotos(label);
        let header = self.block();
        self.emit_branch(stmt, condition)?;
        self.register_loop_label(label, header);

        self.push_loop_frame(label, header);
        self.pending = alloc::vec![(header, LowerEdge::True)];
        self.lower_body(body)?;
        self.connect_to(header);
        let frame = self.pop_frame();

        self.pending = frame.break_sources;
        self.pending.push((header, LowerEdge::False));
        Ok(())
    }

    pub(super) fn lower_do_while(
        &mut self,
        stmt: S::Stmt,
        condition: S::Expr,
        body: &'a [S::Stmt],
        label: Option<&'a str>,
    ) -> Result<(), LowerError<B::Error>> {
        self.seal(LowerEdge::Fallthrough);
        self.adopt_forward_gotos(label);
        // The latch evaluates the condition; continues transfer to it.
        let latch = self.new_block();
        let mark = self.materialized.len();

        self.push_loop_frame(label, latch);
        self.lower_body(body)?;
        self.connect_to(latch);
        let frame = self.pop_frame();
        // The loop is entered where the body begins, or at the latch when the
        // body materialized no block of its own.
        let entry = self.materialized.get(mark).copied().unwrap_or(latch);
        self.register_loop_label(label, entry);

        self.current = Some(latch);
        self.emit_branch(stmt, condition)?;
        self.builder.add_edge(latch, entry, LowerEdge::True);

        self.pending = frame.break_sources;
        self.pending.push((latch, LowerEdge::False));
        Ok(())
    }

    pub(super) fn lower_loop(
        &mut self,
        body: &'a [S::Stmt],
        label: Option<&'a str>,
    ) -> Result<(), LowerError<B::Error>> {
        if body.is_empty() {
            return Err(LowerError::Unsupported(
                "an endless loop with an empty body has no flat form".into(),
            ));
        }
        self.seal(LowerEdge::Fallthrough);
        self.adopt_forward_gotos(label);
        let header = self.block();
        self.register_loop_label(label, header);

        self.push_loop_frame(label, header);
        self.lower_body(body)?;
        self.connect_to(header);
        let frame = self.pop_frame();

        self.pending = frame.break_sources;
        Ok(())
    }

    pub(super) fn lower_for(
        &mut self,
        stmt: S::Stmt,
        initializer: &'a [S::Stmt],
        condition: Option<S::Expr>,
        update: &'a [S::Stmt],
        body: &'a [S::Stmt],
        label: Option<&'a str>,
    ) -> Result<(), LowerError<B::Error>> {
        self.adopt_forward_gotos(label);
        let mark = self.materialized.len();
        self.lower_body(initializer)?;
        self.seal(LowerEdge::Fallthrough);
        let head = self.block();

        let mut exit_sources = Vec::new();
        if let Some(condition) = condition {
            self.emit_branch(stmt, condition)?;
            self.pending = alloc::vec![(head, LowerEdge::True)];
            exit_sources.push((head, LowerEdge::False));
        }
        // The head was materialized after the mark, so the mark names a block.
        self.register_loop_label(label, self.materialized[mark]);

        // Continues transfer to the update section when one exists.
        let update_block = if update.is_empty() {
            None
        } else {
            Some(self.new_block())
        };
        self.push_loop_frame(label, update_block.unwrap_or(head));
        self.lower_body(body)?;
        if let Some(update_entry) = update_block {
            self.connect_to(update_entry);
            self.current = Some(update_entry);
            self.lower_body(update)?;
        }
        self.connect_to(head);
        let frame = self.pop_frame();

        self.pending = frame.break_sources;
        self.pending.extend(exit_sources);
        Ok(())
    }

    fn lower_switch(
        &mut self,
        stmt: S::Stmt,
        scrutinee: S::Expr,
        default_body: &'a [S::Stmt],
    ) -> Result<(), LowerError<B::Error>> {
        let scrutinee = self.flatten(scrutinee)?;
        self.emit(Lowered::Switch, &[scrutinee], &[], Entity::Statement(stmt))?;
        let dispatch = self
            .current
            .take()
            .expect("the dispatch instruction opened a block");

        self.frames.push(Frame {
            label: None,
            kind: FrameKind::Switch,
            break_sources: Vec::new(),
        });
        let source = self.source;
        let mut exit_sources = Vec::new();
        for arm in source.switch_arms(stmt) {
            self.pending = arm
                .values
                .iter()
                .map(|value| (dispatch, LowerEdge::Case(value)))
                .collect();
            self.lower_body(arm.body)?;
            exit_sources.extend(self.take_continuation());
        }
        self.pending = alloc::vec![(dispatch, LowerEdge::Default)];
        self.lower_body(default_body)?;
        exit_sources.extend(self.take_continuation());
        let frame = self.pop_frame();

        exit_sources.extend(frame.break_sources);
        self.pending = exit_sources;
        Ok(())
    }

    fn lower_break(
        &mut self,
        stmt: S::Stmt,
        label: Option<&'a str>,
    ) -> Result<(), LowerError<B::Error>> {
        let sources = self.take_transfer();
        if sources.is_empty() {
            return Ok(());
        }
        let Some(frame) = self.frames.iter_mut().rev().find(|frame| match label {
            Some(label) => frame.label == Some(label),
            None => !matches!(frame.kind, FrameKind::Block),
        }) else {
            let cause = label.map_or(Untargeted::Break, Untargeted::BreakUnenclosed);
            return Err(LowerError::Unsupported(cause.message(stmt)));
        };
        frame.break_sources.extend(sources);
        Ok(())
    }

    fn lower_continue(
        &mut self,
        stmt: S::Stmt,
        label: Option<&'a str>,
    ) -> Result<(), LowerError<B::Error>> {
        let sources = self.take_transfer();
        if sources.is_empty() {
            return Ok(());
        }
        let Some(target) = self.frames.iter().rev().find_map(|frame| match frame.kind {
            FrameKind::Loop { continue_target } if label.is_none() || frame.label == label => {
                Some(continue_target)
            }
            _ => None,
        }) else {
            let cause = match label {
                None => Untargeted::Continue,
                Some(label) if self.frames.iter().any(|frame| frame.label == Some(label)) => {
                    Untargeted::ContinueBlock(label)
                }
                Some(label) => Untargeted::ContinueUnenclosed(label),
            };
            return Err(LowerError::Unsupported(cause.message(stmt)));
        };
        for (from, role) in sources {
            self.builder.add_edge(from, target, role);
        }
        Ok(())
    }

    /// Lowers the loop `stmt`, which `label` names when the loop is the only
    /// statement of a labeled group.
    fn lower_loop_statement(
        &mut self,
        stmt: S::Stmt,
        label: Option<&'a str>,
    ) -> Result<(), LowerError<B::Error>> {
        match self.source.stmt(stmt) {
            StmtShape::While { condition, body } => self.lower_while(stmt, condition, body, label),
            StmtShape::DoWhile { body, condition } => {
                self.lower_do_while(stmt, condition, body, label)
            }
            StmtShape::Loop { body } => self.lower_loop(body, label),
            StmtShape::For {
                initializer,
                condition,
                update,
                body,
            } => self.lower_for(stmt, initializer, condition, update, body, label),
            _ => unreachable!("only a loop statement is lowered as a loop"),
        }
    }

    fn lower_labeled(
        &mut self,
        label: &'a str,
        body: &'a [S::Stmt],
    ) -> Result<(), LowerError<B::Error>> {
        if let Some(only) = labeled_loop::<H, S>(self.source, body) {
            return self.lower_loop_statement(only, Some(label));
        }
        self.seal(LowerEdge::Fallthrough);
        self.adopt_forward_gotos(Some(label));
        let target = self.block();
        self.labels.insert(label, target);
        self.frames.push(Frame {
            label: Some(label),
            kind: FrameKind::Block,
            break_sources: Vec::new(),
        });
        self.lower_body(body)?;
        let frame = self.pop_frame();
        let mut sources = self.take_continuation();
        sources.extend(frame.break_sources);
        self.pending = sources;
        Ok(())
    }

    fn lower_goto(&mut self, label: &'a str) {
        let sources = self.take_transfer();
        if sources.is_empty() {
            return;
        }
        if let Some(&target) = self.labels.get(label) {
            for (from, role) in sources {
                self.builder.add_edge(from, target, role);
            }
        } else {
            self.forward_gotos.entry(label).or_default().extend(sources);
        }
    }

    fn lower_try(
        &mut self,
        stmt: S::Stmt,
        body: &'a [S::Stmt],
        finally_body: &'a [S::Stmt],
    ) -> Result<(), LowerError<B::Error>> {
        self.seal(LowerEdge::Fallthrough);
        let protected_start = self.created.len();
        self.try_frames.push(TryFrame {
            may_throw_blocks: BTreeSet::new(),
        });
        self.lower_body(body)?;
        let try_frame = self.try_frames.pop().expect("the try frame pushed above");
        let protected_end = self.created.len();
        let mut normal = self.take_continuation();

        let source = self.source;
        let mut region_handlers = Vec::new();
        for handler in source.handlers(stmt) {
            region_handlers.push(self.lower_handler(stmt, &handler, &try_frame, &mut normal)?);
        }

        if finally_body.is_empty() {
            self.pending = normal;
        } else {
            let finally_entry = self.new_block();
            for &block in &try_frame.may_throw_blocks {
                self.builder
                    .add_edge(block, finally_entry, LowerEdge::Unwind);
            }
            for (from, role) in normal {
                self.builder.add_edge(from, finally_entry, role);
            }
            self.current = Some(finally_entry);
            let finally_start = self.created.len();
            self.lower_body(finally_body)?;
            let mut blocks: BTreeSet<BlockId> =
                self.created[finally_start..].iter().copied().collect();
            blocks.insert(finally_entry);
            region_handlers.push(region::Handler {
                entry: finally_entry,
                body: region::HandlerBody::Known(blocks),
                kind: region::HandlerKind::Finally,
            });
        }

        let protected_blocks: BTreeSet<BlockId> = self.created[protected_start..protected_end]
            .iter()
            .copied()
            .collect();
        if !protected_blocks.is_empty() {
            self.builder.add_region(region::Region {
                id: RegionId::from_raw(0),
                protected_blocks,
                handlers: region_handlers,
                parent: None,
            });
        }
        Ok(())
    }

    fn lower_handler(
        &mut self,
        stmt: S::Stmt,
        handler: &Handler<'a, H, S::Stmt, S::Var>,
        try_frame: &TryFrame,
        normal: &mut Sources<'a, H::Constant>,
    ) -> Result<region::Handler, LowerError<B::Error>> {
        let entry = self.new_block();
        for &block in &try_frame.may_throw_blocks {
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
                Entity::Statement(stmt),
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
                self.lower_body(filter_body)?;
                if self.current.is_some() || !self.pending.is_empty() {
                    return Err(LowerError::Unsupported(
                        "a filter body must end in a terminator".into(),
                    ));
                }
                self.current = saved_current;
                self.pending = saved_pending;
                region::HandlerKind::Filter { filter_block }
            }
        };

        let body_start = self.created.len();
        self.lower_body(handler.body)?;
        normal.extend(self.take_continuation());
        let mut blocks: BTreeSet<BlockId> = self.created[body_start..].iter().copied().collect();
        blocks.insert(entry);
        Ok(region::Handler {
            entry,
            body: region::HandlerBody::Known(blocks),
            kind,
        })
    }

    fn lower_region(
        &mut self,
        stmt: S::Stmt,
        op: &'a H::Op,
        operands: &[S::Expr],
        body: &'a [S::Stmt],
    ) -> Result<(), LowerError<B::Error>> {
        let origin = Entity::Statement(stmt);
        // The operands are evaluated once, on entry, and the exit releases
        // what they held then. A tree flattens into temporaries that nothing
        // else writes, but a bare variable read would alias a variable that
        // the body may reassign, so it is copied first.
        let mut uses = Vec::with_capacity(operands.len());
        for &operand in operands {
            self.flatten_onto(operand, &mut uses)?;
            if matches!(self.source.expr(operand), ExprShape::Variable(_)) {
                let read = uses.pop().expect("flattening leaves the operand's value");
                let held = Occurrence {
                    variable: self.temporary(&read.ty),
                    ty: read.ty.clone(),
                };
                self.emit(
                    Lowered::Copy,
                    &[read],
                    core::slice::from_ref(&held),
                    Entity::Expression(operand),
                )?;
                uses.push(held);
            }
        }
        self.emit(Lowered::RegionEnter(op), &uses, &[], origin)?;
        self.lower_body(body)?;
        self.emit(Lowered::RegionExit(op), &uses, &[], origin)
    }
}
