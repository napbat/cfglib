//! Statement lowering: blocks, joins, loops, switches, and regions.
//!
//! Control flow lowers through lazy join materialization: ending a
//! construct leaves its continuation as pending edges, and the next emitted
//! instruction materializes the join block and wires them. A path that
//! terminates (return, goto, break) simply leaves nothing pending.

extern crate alloc;

use alloc::collections::BTreeSet;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use crate::block::BlockId;
use crate::ir::mlil;
use crate::region;

use super::super::statement::HandlerKind;
use super::super::{
    EntityId, Error, ExpressionId, ExpressionKind, Result, StatementId, StatementKind, SwitchArm,
};
use super::{Frame, FrameKind, LowerDialect, Lowerer, TryFrame};

impl<D: LowerDialect + mlil::VerifyDialect> Lowerer<'_, D> {
    pub(super) fn lower_body(&mut self, body: &[StatementId]) -> Result<()> {
        for &statement in body {
            self.lower_statement(statement)?;
        }
        Ok(())
    }

    fn lower_statement(&mut self, id: StatementId) -> Result<()> {
        let statement = self.statement(id)?;
        let origin = EntityId::Statement(id);
        match statement.kind().clone() {
            StatementKind::Expression(expression) => self.lower_expression_statement(expression),
            StatementKind::Assign { target, value } => self.lower_assign(origin, target, value),
            StatementKind::If {
                condition,
                then_body,
                else_body,
            } => self.lower_if(origin, condition, &then_body, &else_body),
            StatementKind::While { condition, body } => {
                self.lower_while(origin, condition, &body, None)
            }
            StatementKind::DoWhile { body, condition } => {
                self.lower_do_while(origin, condition, &body, None)
            }
            StatementKind::Loop { body } => self.lower_loop(&body, None),
            StatementKind::For {
                initializer,
                condition,
                update,
                body,
            } => self.lower_for(origin, &initializer, condition, &update, &body, None),
            StatementKind::Switch {
                scrutinee,
                cases,
                default_body,
            } => self.lower_switch(origin, scrutinee, &cases, &default_body),
            StatementKind::Break { label } => self.lower_break(id, label.as_deref()),
            StatementKind::Continue { label } => self.lower_continue(id, label.as_deref()),
            StatementKind::Return { values } => {
                let uses = self.flatten_all(&values)?;
                self.emit(D::return_operation(), uses, Vec::new(), origin)?;
                self.current = None;
                Ok(())
            }
            StatementKind::Labeled { label, body } => self.lower_labeled(&label, &body),
            StatementKind::Goto { label } => self.lower_goto(&label),
            StatementKind::Try {
                body,
                handlers,
                finally_body,
            } => self.lower_try(origin, &body, &handlers, &finally_body),
            StatementKind::Region {
                operation,
                operands,
                body,
            } => self.lower_region(origin, &operation, &operands, &body),
        }
    }

    fn lower_expression_statement(&mut self, expression: ExpressionId) -> Result<()> {
        let node = self.expression(expression)?;
        if let ExpressionKind::Operation {
            operation,
            operands,
        } = node.kind().clone()
        {
            let uses = self.flatten_all(&operands)?;
            self.emit(
                D::lower_operation(&operation),
                uses,
                Vec::new(),
                EntityId::Expression(expression),
            )?;
        }
        // A bare variable or constant evaluated for effect lowers to nothing.
        Ok(())
    }

    fn lower_assign(
        &mut self,
        origin: EntityId,
        target: ExpressionId,
        value: ExpressionId,
    ) -> Result<()> {
        let target_node = self.expression(target)?;
        match target_node.kind().clone() {
            ExpressionKind::Variable(variable) => {
                let target_type = target_node.value_type().clone();
                self.flatten_into(
                    mlil::VariableId::from_raw(variable.raw()),
                    target_type,
                    value,
                    origin,
                )
            }
            ExpressionKind::Operation {
                operation,
                operands,
            } => {
                let mut uses = self.flatten_all(&operands)?;
                let (value_variable, value_type) = self.flatten(value)?;
                uses.push(mlil::TypedVariable::new(value_variable, value_type));
                self.emit(D::store_operation(&operation), uses, Vec::new(), origin)?;
                Ok(())
            }
            ExpressionKind::Constant(_) => Err(Error::UnsupportedLower(
                "assignment into a constant place".into(),
            )),
        }
    }

    /// The open continuation as join sources: pending edges plus the open
    /// block's sequential fall-out.
    fn take_continuation(&mut self) -> Vec<(BlockId, <D as mlil::Dialect>::Edge)> {
        let mut sources = core::mem::take(&mut self.pending);
        if let Some(block) = self.current.take() {
            sources.push((block, D::fallthrough_edge()));
        }
        sources
    }

    /// The open continuation as explicit-transfer sources (breaks, gotos).
    fn take_transfer(&mut self) -> Vec<(BlockId, <D as mlil::Dialect>::Edge)> {
        let mut sources = core::mem::take(&mut self.pending);
        if let Some(block) = self.current.take() {
            sources.push((block, D::jump_edge()));
        }
        sources
    }

    fn lower_if(
        &mut self,
        origin: EntityId,
        condition: ExpressionId,
        then_body: &[StatementId],
        else_body: &[StatementId],
    ) -> Result<()> {
        let (variable, value_type) = self.flatten(condition)?;
        self.emit(
            D::branch_operation(),
            alloc::vec![mlil::TypedVariable::new(variable, value_type)],
            Vec::new(),
            origin,
        )?;
        let branch = self
            .current
            .take()
            .expect("the branch instruction opened a block");

        self.pending = alloc::vec![(branch, D::true_edge())];
        self.lower_body(then_body)?;
        let mut join = self.take_continuation();

        self.pending = alloc::vec![(branch, D::false_edge())];
        self.lower_body(else_body)?;
        join.extend(self.take_continuation());

        self.pending = join;
        Ok(())
    }

    /// Registers a loop's label: forward gotos already flowed into the
    /// entry through the pending list; later gotos resolve to `entry`.
    fn register_loop_label(&mut self, label: Option<&str>, entry: BlockId) {
        if let Some(label) = label {
            self.labels.insert(String::from(label), entry);
        }
    }

    fn adopt_forward_gotos(&mut self, label: Option<&str>) {
        if let Some(label) = label
            && let Some(waiting) = self.forward_gotos.remove(label)
        {
            self.pending.extend(waiting);
        }
    }

    pub(super) fn lower_while(
        &mut self,
        origin: EntityId,
        condition: ExpressionId,
        body: &[StatementId],
        label: Option<String>,
    ) -> Result<()> {
        self.seal(D::fallthrough_edge());
        self.adopt_forward_gotos(label.as_deref());
        let header = self.block()?;
        let (variable, value_type) = self.flatten(condition)?;
        self.emit(
            D::branch_operation(),
            alloc::vec![mlil::TypedVariable::new(variable, value_type)],
            Vec::new(),
            origin,
        )?;
        self.current = None;
        self.register_loop_label(label.as_deref(), header);

        self.frames.push(Frame {
            label,
            kind: FrameKind::Loop {
                continue_target: header,
            },
            break_sources: Vec::new(),
        });
        self.pending = alloc::vec![(header, D::true_edge())];
        self.lower_body(body)?;
        self.connect_to(header)?;
        let frame = self.frames.pop().expect("the loop frame pushed above");

        self.pending = frame.break_sources;
        self.pending.push((header, D::false_edge()));
        Ok(())
    }

    pub(super) fn lower_do_while(
        &mut self,
        origin: EntityId,
        condition: ExpressionId,
        body: &[StatementId],
        label: Option<String>,
    ) -> Result<()> {
        self.seal(D::fallthrough_edge());
        self.adopt_forward_gotos(label.as_deref());
        // The latch evaluates the condition; continues transfer to it.
        let latch = self.builder.new_block("");
        let body_start = self.builder.block_count();

        self.frames.push(Frame {
            label,
            kind: FrameKind::Loop {
                continue_target: latch,
            },
            break_sources: Vec::new(),
        });
        self.lower_body(body)?;
        self.connect_to(latch)?;
        let frame = self.frames.pop().expect("the loop frame pushed above");
        let entry = if self.builder.block_count() > body_start {
            BlockId::from_index(body_start)
        } else {
            latch
        };
        self.register_loop_label(frame.label.as_deref(), entry);

        self.current = Some(latch);
        let (variable, value_type) = self.flatten(condition)?;
        self.emit(
            D::branch_operation(),
            alloc::vec![mlil::TypedVariable::new(variable, value_type)],
            Vec::new(),
            origin,
        )?;
        self.builder.add_edge(latch, entry, D::true_edge(), None)?;
        self.current = None;

        self.pending = frame.break_sources;
        self.pending.push((latch, D::false_edge()));
        Ok(())
    }

    pub(super) fn lower_loop(&mut self, body: &[StatementId], label: Option<String>) -> Result<()> {
        if body.is_empty() {
            return Err(Error::UnsupportedLower(
                "an endless loop with an empty body has no flat form".into(),
            ));
        }
        self.seal(D::fallthrough_edge());
        self.adopt_forward_gotos(label.as_deref());
        let header = self.block()?;
        self.register_loop_label(label.as_deref(), header);

        self.frames.push(Frame {
            label,
            kind: FrameKind::Loop {
                continue_target: header,
            },
            break_sources: Vec::new(),
        });
        self.lower_body(body)?;
        self.connect_to(header)?;
        let frame = self.frames.pop().expect("the loop frame pushed above");

        self.pending = frame.break_sources;
        Ok(())
    }

    pub(super) fn lower_for(
        &mut self,
        origin: EntityId,
        initializer: &[StatementId],
        condition: Option<ExpressionId>,
        update: &[StatementId],
        body: &[StatementId],
        label: Option<String>,
    ) -> Result<()> {
        self.adopt_forward_gotos(label.as_deref());
        let first = self.builder.block_count();
        self.lower_body(initializer)?;
        self.seal(D::fallthrough_edge());
        let head = self.block()?;

        let mut exit_sources = Vec::new();
        if let Some(condition) = condition {
            let (variable, value_type) = self.flatten(condition)?;
            self.emit(
                D::branch_operation(),
                alloc::vec![mlil::TypedVariable::new(variable, value_type)],
                Vec::new(),
                origin,
            )?;
            self.current = None;
            self.pending = alloc::vec![(head, D::true_edge())];
            exit_sources.push((head, D::false_edge()));
        }
        let entry = if self.builder.block_count() > first {
            BlockId::from_index(first)
        } else {
            head
        };
        self.register_loop_label(label.as_deref(), entry);

        // Continues transfer to the update section when one exists.
        let update_block = if update.is_empty() {
            None
        } else {
            Some(self.builder.new_block(""))
        };
        self.frames.push(Frame {
            label,
            kind: FrameKind::Loop {
                continue_target: update_block.unwrap_or(head),
            },
            break_sources: Vec::new(),
        });
        self.lower_body(body)?;
        if let Some(update_entry) = update_block {
            self.connect_to(update_entry)?;
            self.current = Some(update_entry);
            self.lower_body(update)?;
        }
        self.connect_to(head)?;
        let frame = self.frames.pop().expect("the loop frame pushed above");

        self.pending = frame.break_sources;
        self.pending.extend(exit_sources);
        Ok(())
    }

    fn lower_switch(
        &mut self,
        origin: EntityId,
        scrutinee: ExpressionId,
        cases: &[SwitchArm<D>],
        default_body: &[StatementId],
    ) -> Result<()> {
        let (variable, value_type) = self.flatten(scrutinee)?;
        self.emit(
            D::switch_operation(),
            alloc::vec![mlil::TypedVariable::new(variable, value_type)],
            Vec::new(),
            origin,
        )?;
        let dispatch = self
            .current
            .take()
            .expect("the dispatch instruction opened a block");

        self.frames.push(Frame {
            label: None,
            kind: FrameKind::Switch,
            break_sources: Vec::new(),
        });
        let mut exit_sources = Vec::new();
        for arm in cases {
            self.pending = arm
                .values
                .iter()
                .map(|value| (dispatch, D::case_edge(value)))
                .collect();
            self.lower_body(&arm.body)?;
            exit_sources.extend(self.take_continuation());
        }
        self.pending = alloc::vec![(dispatch, D::default_edge())];
        self.lower_body(default_body)?;
        exit_sources.extend(self.take_continuation());
        let frame = self.frames.pop().expect("the switch frame pushed above");

        exit_sources.extend(frame.break_sources);
        self.pending = exit_sources;
        Ok(())
    }

    fn lower_break(&mut self, id: StatementId, label: Option<&str>) -> Result<()> {
        let sources = self.take_transfer();
        if sources.is_empty() {
            return Ok(());
        }
        let frame = self
            .frames
            .iter_mut()
            .rev()
            .find(|frame| match label {
                Some(label) => frame.label.as_deref() == Some(label),
                None => !matches!(frame.kind, FrameKind::Block),
            })
            .ok_or_else(|| Error::UnsupportedLower(format_transfer_error("break", id, label)))?;
        frame.break_sources.extend(sources);
        Ok(())
    }

    fn lower_continue(&mut self, id: StatementId, label: Option<&str>) -> Result<()> {
        let sources = self.take_transfer();
        if sources.is_empty() {
            return Ok(());
        }
        let target = self
            .frames
            .iter()
            .rev()
            .find_map(|frame| match (&frame.kind, label) {
                (FrameKind::Loop { continue_target }, None) => Some(*continue_target),
                (FrameKind::Loop { continue_target }, Some(label))
                    if frame.label.as_deref() == Some(label) =>
                {
                    Some(*continue_target)
                }
                _ => None,
            })
            .ok_or_else(|| Error::UnsupportedLower(format_transfer_error("continue", id, label)))?;
        for (from, edge) in sources {
            self.builder.add_edge(from, target, edge, None)?;
        }
        Ok(())
    }

    fn lower_labeled(&mut self, label: &str, body: &[StatementId]) -> Result<()> {
        // A label wrapping exactly one loop names that loop directly, so
        // labeled breaks and continues resolve on the loop's own frame.
        if let [only] = body {
            let statement = self.statement(*only)?;
            let loop_origin = EntityId::Statement(*only);
            match statement.kind().clone() {
                StatementKind::While { condition, body } => {
                    return self.lower_while(
                        loop_origin,
                        condition,
                        &body,
                        Some(String::from(label)),
                    );
                }
                StatementKind::DoWhile { body, condition } => {
                    return self.lower_do_while(
                        loop_origin,
                        condition,
                        &body,
                        Some(String::from(label)),
                    );
                }
                StatementKind::Loop { body } => {
                    return self.lower_loop(&body, Some(String::from(label)));
                }
                StatementKind::For {
                    initializer,
                    condition,
                    update,
                    body,
                } => {
                    return self.lower_for(
                        loop_origin,
                        &initializer,
                        condition,
                        &update,
                        &body,
                        Some(String::from(label)),
                    );
                }
                _ => {}
            }
        }
        self.seal(D::fallthrough_edge());
        self.adopt_forward_gotos(Some(label));
        let target = self.block()?;
        self.labels.insert(String::from(label), target);
        self.frames.push(Frame {
            label: Some(String::from(label)),
            kind: FrameKind::Block,
            break_sources: Vec::new(),
        });
        self.lower_body(body)?;
        let frame = self.frames.pop().expect("the labeled frame pushed above");
        let mut sources = self.take_continuation();
        sources.extend(frame.break_sources);
        self.pending = sources;
        Ok(())
    }

    fn lower_goto(&mut self, label: &str) -> Result<()> {
        let sources = self.take_transfer();
        if sources.is_empty() {
            return Ok(());
        }
        if let Some(&target) = self.labels.get(label) {
            for (from, edge) in sources {
                self.builder.add_edge(from, target, edge, None)?;
            }
        } else {
            self.forward_gotos
                .entry(String::from(label))
                .or_default()
                .extend(sources);
        }
        Ok(())
    }

    fn lower_try(
        &mut self,
        origin: EntityId,
        body: &[StatementId],
        handlers: &[super::super::Handler<D>],
        finally_body: &[StatementId],
    ) -> Result<()> {
        self.seal(D::fallthrough_edge());
        let protected_start = self.builder.block_count();
        self.try_frames.push(TryFrame {
            may_throw_blocks: BTreeSet::new(),
        });
        self.lower_body(body)?;
        let try_frame = self.try_frames.pop().expect("the try frame pushed above");
        let protected_end = self.builder.block_count();
        let mut normal = self.take_continuation();

        let mut region_handlers = Vec::new();
        for handler in handlers {
            region_handlers.push(self.lower_handler(origin, handler, &try_frame, &mut normal)?);
        }

        if finally_body.is_empty() {
            self.pending = normal;
        } else {
            let finally_entry = self.builder.new_block("");
            for &block in &try_frame.may_throw_blocks {
                self.builder
                    .add_edge(block, finally_entry, D::unwind_edge(), None)?;
            }
            for (from, edge) in normal.drain(..) {
                self.builder.add_edge(from, finally_entry, edge, None)?;
            }
            self.current = Some(finally_entry);
            let finally_start = self.builder.block_count();
            self.lower_body(finally_body)?;
            let mut blocks: BTreeSet<BlockId> = (finally_start..self.builder.block_count())
                .map(BlockId::from_index)
                .collect();
            blocks.insert(finally_entry);
            region_handlers.push(region::Handler {
                entry: finally_entry,
                body: region::HandlerBody::Known(blocks),
                kind: region::HandlerKind::Finally,
            });
        }

        let protected_blocks: BTreeSet<BlockId> = (protected_start..protected_end)
            .map(BlockId::from_index)
            .collect();
        if !protected_blocks.is_empty() {
            self.builder.add_region(region::Region {
                id: crate::RegionId::from_raw(0),
                protected_blocks,
                handlers: region_handlers,
                parent: None,
            })?;
        }
        Ok(())
    }

    fn lower_handler(
        &mut self,
        origin: EntityId,
        handler: &super::super::Handler<D>,
        try_frame: &TryFrame,
        normal: &mut Vec<(BlockId, <D as mlil::Dialect>::Edge)>,
    ) -> Result<region::Handler> {
        let entry = self.builder.new_block("");
        for &block in &try_frame.may_throw_blocks {
            self.builder
                .add_edge(block, entry, D::unwind_edge(), None)?;
        }
        self.current = Some(entry);
        self.pending.clear();

        if let Some(binding) = handler.binding {
            let Some(operation) = D::caught_exception_operation() else {
                return Err(Error::UnsupportedLower(
                    "a handler binding needs a caught-exception operation".into(),
                ));
            };
            let variable = self.source.variable(binding).ok_or_else(|| {
                Error::UnsupportedLower(format!("handler binds missing variable {binding}"))
            })?;
            let value_type = variable.declared_type.clone().ok_or_else(|| {
                Error::UnsupportedLower(format!("handler binding {binding} has no declared type"))
            })?;
            self.emit(
                operation,
                Vec::new(),
                alloc::vec![mlil::TypedVariable::new(
                    mlil::VariableId::from_raw(binding.raw()),
                    value_type,
                )],
                origin,
            )?;
        }

        let kind = match &handler.kind {
            HandlerKind::Catch => region::HandlerKind::Catch,
            HandlerKind::CatchAll => region::HandlerKind::CatchAll,
            HandlerKind::Fault => region::HandlerKind::Fault,
            HandlerKind::Filter { filter_body } => {
                if filter_body.is_empty() {
                    return Err(Error::UnsupportedLower(
                        "a filter handler needs a non-empty filter body".into(),
                    ));
                }
                // The funclet is runtime-invoked: lower it into blocks of
                // its own with no sequential continuation.
                let saved_current = self.current.take();
                let saved_pending = core::mem::take(&mut self.pending);
                let filter_block = self.builder.new_block("");
                self.current = Some(filter_block);
                self.lower_body(filter_body)?;
                if self.current.is_some() || !self.pending.is_empty() {
                    return Err(Error::UnsupportedLower(
                        "a filter body must end in a terminator".into(),
                    ));
                }
                self.current = saved_current;
                self.pending = saved_pending;
                region::HandlerKind::Filter { filter_block }
            }
        };

        let body_start = self.builder.block_count();
        self.lower_body(&handler.body)?;
        normal.extend(self.take_continuation());
        let mut blocks: BTreeSet<BlockId> = (body_start..self.builder.block_count())
            .map(BlockId::from_index)
            .collect();
        blocks.insert(entry);
        Ok(region::Handler {
            entry,
            body: region::HandlerBody::Known(blocks),
            kind,
        })
    }

    fn lower_region(
        &mut self,
        origin: EntityId,
        operation: &<D as super::Dialect>::Operation,
        operands: &[ExpressionId],
        body: &[StatementId],
    ) -> Result<()> {
        let Some((enter, exit)) = D::region_operations(operation) else {
            return Err(Error::UnsupportedLower(
                "a region statement needs enter/exit operations".into(),
            ));
        };
        let uses = self.flatten_all(operands)?;
        self.emit(enter, uses.clone(), Vec::new(), origin)?;
        self.lower_body(body)?;
        self.emit(exit, uses, Vec::new(), origin)?;
        Ok(())
    }
}

fn format_transfer_error(keyword: &str, id: StatementId, label: Option<&str>) -> String {
    match label {
        Some(label) => format!("{id} {keyword}s unenclosing label {label}"),
        None => format!("{id} {keyword}s outside any enclosing construct"),
    }
}
