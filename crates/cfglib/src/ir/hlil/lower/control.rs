//! Statement lowering: blocks, joins, loops, switches, and regions.
//!
//! Control flow lowers through lazy join materialization: ending a construct
//! leaves its continuation as pending edges, and the next emitted
//! instruction materializes the join block and wires them. A path that
//! terminates (return, goto, break) simply leaves nothing pending.
//!
//! The depth of the nesting costs no call frames. A construct lowers
//! the part that precedes its bodies, then leaves a [`Step`] for the part
//! that follows them, and [`Lowerer::lower_body`] runs the bodies and the
//! steps on a stack of its own.

extern crate alloc;

use alloc::boxed::Box;
use alloc::vec::{self, Vec};
use core::slice;

use crate::block::BlockId;

use super::super::flat_builder::{FlatBuilder, LowerEdge, Lowered, Occurrence};
use super::super::transfer::Untargeted;
use super::super::view::labeled_loop;
use super::super::{Dialect, Entity, ExprShape, FunctionView, LowerError, StmtShape, SwitchArm};
use super::exception::{TryScope, TryStage};
use super::{Frame, FrameKind, Lowerer, Origin, Sources, Value};

/// What lowering has still to do: the rest of a body, or the part of a
/// construct that follows one of its bodies.
pub(super) enum Step<'a, H: Dialect, S: FunctionView<H>, B: FlatBuilder<H>> {
    /// The rest of a body.
    Body(slice::Iter<'a, S::Stmt>),
    /// A conditional whose `then` body is lowered; its `else` body is next.
    IfThen {
        branch: BlockId,
        else_body: &'a [S::Stmt],
    },
    /// A conditional whose `else` body is lowered; the two arms join.
    IfElse { join: Sources<'a, H::Constant> },
    /// A `while` loop whose body is lowered.
    While { header: BlockId },
    /// A `do`-`while` loop whose body is lowered; its latch is next.
    DoWhile {
        stmt: S::Stmt,
        condition: S::Expr,
        label: Option<&'a str>,
        latch: BlockId,
        /// How many blocks were materialized before the body.
        mark: usize,
    },
    /// An endless loop whose body is lowered.
    Loop { header: BlockId },
    /// A `for` loop whose initializer is lowered; its head and body are next.
    ForInitializer(ForTail<'a, H, S>),
    /// A `for` loop whose body is lowered; its update is next, if it has one.
    ForBody {
        head: BlockId,
        update_block: Option<BlockId>,
        update: &'a [S::Stmt],
        exit_sources: Sources<'a, H::Constant>,
    },
    /// A `for` loop whose update is lowered.
    ForUpdate {
        head: BlockId,
        exit_sources: Sources<'a, H::Constant>,
    },
    /// A switch whose arm, or whose default body, is lowered.
    Switch(SwitchScope<'a, H, S>),
    /// A labeled group that is not a loop, whose body is lowered.
    Labeled,
    /// A region statement whose body is lowered; its exit is next.
    Region {
        op: &'a H::Op,
        uses: Vec<Value<H, B>>,
        origin: Origin<H, S>,
    },
    /// A `try` whose part named by the stage is lowered.
    Try {
        scope: Box<TryScope<'a, H, S>>,
        stage: TryStage<'a, H, S>,
    },
}

/// The stack of what lowering has still to do, the next on top.
pub(super) type Steps<'a, H, S, B> = Vec<Step<'a, H, S, B>>;

/// The parts of a `for` loop that follow its initializer.
pub(super) struct ForTail<'a, H: Dialect, S: FunctionView<H>> {
    stmt: S::Stmt,
    condition: Option<S::Expr>,
    update: &'a [S::Stmt],
    body: &'a [S::Stmt],
    label: Option<&'a str>,
    /// How many blocks were materialized before the initializer.
    mark: usize,
}

/// A switch whose arms are being lowered.
pub(super) struct SwitchScope<'a, H: Dialect, S: FunctionView<H>> {
    dispatch: BlockId,
    /// The arms that are not lowered yet.
    arms: vec::IntoIter<SwitchArm<'a, H, S::Stmt>>,
    /// The default body, until it is lowered.
    default_body: Option<&'a [S::Stmt]>,
    /// The edges that leave the arms for what follows the switch.
    exit_sources: Sources<'a, H::Constant>,
}

impl<'a, H, S, B> Lowerer<'a, H, S, B>
where
    H: Dialect,
    S: FunctionView<H>,
    B: FlatBuilder<H>,
{
    /// Lowers `body` and everything nested in it.
    ///
    /// The depth of the nesting costs no call frames: the part of a construct
    /// that follows its bodies waits in a [`Step`] instead.
    pub(super) fn lower_body(&mut self, body: &'a [S::Stmt]) -> Result<(), LowerError<B::Error>> {
        let mut steps = alloc::vec![Step::Body(body.iter())];
        while let Some(step) = steps.pop() {
            match step {
                Step::Body(mut statements) => {
                    if let Some(&stmt) = statements.next() {
                        steps.push(Step::Body(statements));
                        self.lower_statement(stmt, &mut steps)?;
                    }
                }
                Step::IfThen { branch, else_body } => {
                    let join = self.take_continuation();
                    self.pending = alloc::vec![(branch, LowerEdge::False)];
                    Self::enter(&mut steps, else_body, Step::IfElse { join });
                }
                Step::IfElse { mut join } => {
                    join.extend(self.take_continuation());
                    self.pending = join;
                }
                Step::While { header } => {
                    self.connect_to(header);
                    let frame = self.pop_frame();
                    self.pending = frame.break_sources;
                    self.pending.push((header, LowerEdge::False));
                }
                Step::DoWhile {
                    stmt,
                    condition,
                    label,
                    latch,
                    mark,
                } => self.lower_do_while_latch(stmt, condition, label, latch, mark)?,
                Step::Loop { header } => {
                    self.connect_to(header);
                    let frame = self.pop_frame();
                    self.pending = frame.break_sources;
                }
                Step::ForInitializer(tail) => self.lower_for_head(&tail, &mut steps)?,
                Step::ForBody {
                    head,
                    update_block,
                    update,
                    exit_sources,
                } => {
                    if let Some(update_entry) = update_block {
                        self.connect_to(update_entry);
                        self.current = Some(update_entry);
                        let step = Step::ForUpdate { head, exit_sources };
                        Self::enter(&mut steps, update, step);
                    } else {
                        self.finish_for(head, exit_sources);
                    }
                }
                Step::ForUpdate { head, exit_sources } => self.finish_for(head, exit_sources),
                Step::Switch(mut switch) => {
                    switch.exit_sources.extend(self.take_continuation());
                    self.lower_switch_part(switch, &mut steps);
                }
                Step::Labeled => {
                    let frame = self.pop_frame();
                    let mut sources = self.take_continuation();
                    sources.extend(frame.break_sources);
                    self.pending = sources;
                }
                Step::Region { op, uses, origin } => {
                    self.emit(Lowered::RegionExit(op), &uses, &[], origin)?;
                }
                Step::Try { scope, stage } => self.resume_try(scope, stage, &mut steps)?,
            }
        }
        Ok(())
    }

    /// Runs `body`, then `after`.
    pub(super) fn enter(
        steps: &mut Steps<'a, H, S, B>,
        body: &'a [S::Stmt],
        after: Step<'a, H, S, B>,
    ) {
        steps.push(after);
        steps.push(Step::Body(body.iter()));
    }

    /// Lowers the part of `stmt` that precedes its bodies, and schedules the
    /// bodies and the rest.
    fn lower_statement(
        &mut self,
        stmt: S::Stmt,
        steps: &mut Steps<'a, H, S, B>,
    ) -> Result<(), LowerError<B::Error>> {
        match self.source.stmt(stmt) {
            StmtShape::Expression(expr) => self.lower_expression_statement(expr),
            StmtShape::Assign { target, value } => self.lower_assign(stmt, target, value),
            StmtShape::If {
                condition,
                then_body,
                else_body,
            } => self.lower_if(stmt, condition, then_body, else_body, steps),
            StmtShape::While { .. }
            | StmtShape::DoWhile { .. }
            | StmtShape::Loop { .. }
            | StmtShape::For { .. } => self.lower_loop_statement(stmt, None, steps),
            StmtShape::Switch {
                scrutinee,
                default_body,
            } => self.lower_switch(stmt, scrutinee, default_body, steps),
            StmtShape::Break { label } => self.lower_break(stmt, label),
            StmtShape::Continue { label } => self.lower_continue(stmt, label),
            StmtShape::Return { values } => {
                let uses = self.flatten_all(values)?;
                self.emit(Lowered::Return, &uses, &[], Entity::Statement(stmt))?;
                self.current = None;
                Ok(())
            }
            StmtShape::Labeled { label, body } => self.lower_labeled(label, body, steps),
            StmtShape::Goto { label } => {
                self.lower_goto(label);
                Ok(())
            }
            StmtShape::Try { body, finally_body } => {
                self.lower_try(stmt, body, finally_body, steps);
                Ok(())
            }
            StmtShape::Region { op, operands, body } => {
                self.lower_region(stmt, op, operands, body, steps)
            }
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
    pub(super) fn take_continuation(&mut self) -> Sources<'a, H::Constant> {
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
        steps: &mut Steps<'a, H, S, B>,
    ) -> Result<(), LowerError<B::Error>> {
        let branch = self.emit_branch(stmt, condition)?;
        self.pending = alloc::vec![(branch, LowerEdge::True)];
        Self::enter(steps, then_body, Step::IfThen { branch, else_body });
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

    pub(super) fn pop_frame(&mut self) -> Frame<'a, H::Constant> {
        self.frames
            .pop()
            .expect("the frame pushed by the construct")
    }

    fn lower_while(
        &mut self,
        stmt: S::Stmt,
        condition: S::Expr,
        body: &'a [S::Stmt],
        label: Option<&'a str>,
        steps: &mut Steps<'a, H, S, B>,
    ) -> Result<(), LowerError<B::Error>> {
        self.seal(LowerEdge::Fallthrough);
        self.adopt_forward_gotos(label);
        let header = self.block();
        self.emit_branch(stmt, condition)?;
        self.register_loop_label(label, header);

        self.push_loop_frame(label, header);
        self.pending = alloc::vec![(header, LowerEdge::True)];
        Self::enter(steps, body, Step::While { header });
        Ok(())
    }

    fn lower_do_while(
        &mut self,
        stmt: S::Stmt,
        condition: S::Expr,
        body: &'a [S::Stmt],
        label: Option<&'a str>,
        steps: &mut Steps<'a, H, S, B>,
    ) {
        self.seal(LowerEdge::Fallthrough);
        self.adopt_forward_gotos(label);
        // The latch evaluates the condition; continues transfer to it.
        let latch = self.new_block();
        let mark = self.materialized.len();

        self.push_loop_frame(label, latch);
        let after = Step::DoWhile {
            stmt,
            condition,
            label,
            latch,
            mark,
        };
        Self::enter(steps, body, after);
    }

    /// Ends a `do`-`while` loop whose body is lowered, with the latch that
    /// evaluates its condition.
    fn lower_do_while_latch(
        &mut self,
        stmt: S::Stmt,
        condition: S::Expr,
        label: Option<&'a str>,
        latch: BlockId,
        mark: usize,
    ) -> Result<(), LowerError<B::Error>> {
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

    fn lower_loop(
        &mut self,
        body: &'a [S::Stmt],
        label: Option<&'a str>,
        steps: &mut Steps<'a, H, S, B>,
    ) {
        self.seal(LowerEdge::Fallthrough);
        self.adopt_forward_gotos(label);
        let header = self.block();
        self.register_loop_label(label, header);

        self.push_loop_frame(label, header);
        Self::enter(steps, body, Step::Loop { header });
    }

    /// Opens the head of a `for` loop whose initializer is lowered, and
    /// schedules its body.
    fn lower_for_head(
        &mut self,
        tail: &ForTail<'a, H, S>,
        steps: &mut Steps<'a, H, S, B>,
    ) -> Result<(), LowerError<B::Error>> {
        let ForTail {
            stmt,
            condition,
            update,
            body,
            label,
            mark,
        } = *tail;
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
        let after = Step::ForBody {
            head,
            update_block,
            update,
            exit_sources,
        };
        Self::enter(steps, body, after);
        Ok(())
    }

    /// Ends a `for` loop whose body and update are lowered.
    fn finish_for(&mut self, head: BlockId, exit_sources: Sources<'a, H::Constant>) {
        self.connect_to(head);
        let frame = self.pop_frame();
        self.pending = frame.break_sources;
        self.pending.extend(exit_sources);
    }

    fn lower_switch(
        &mut self,
        stmt: S::Stmt,
        scrutinee: S::Expr,
        default_body: &'a [S::Stmt],
        steps: &mut Steps<'a, H, S, B>,
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
        let switch = SwitchScope {
            dispatch,
            arms: source.switch_arms(stmt).collect::<Vec<_>>().into_iter(),
            default_body: Some(default_body),
            exit_sources: Vec::new(),
        };
        self.lower_switch_part(switch, steps);
        Ok(())
    }

    /// Starts the next arm of `switch`, then its default body, and ends the
    /// switch once both are lowered.
    fn lower_switch_part(
        &mut self,
        mut switch: SwitchScope<'a, H, S>,
        steps: &mut Steps<'a, H, S, B>,
    ) {
        let body = if let Some(arm) = switch.arms.next() {
            self.pending = arm
                .values
                .iter()
                .map(|value| (switch.dispatch, LowerEdge::Case(value)))
                .collect();
            arm.body
        } else if let Some(default_body) = switch.default_body.take() {
            self.pending = alloc::vec![(switch.dispatch, LowerEdge::Default)];
            default_body
        } else {
            let frame = self.pop_frame();
            switch.exit_sources.extend(frame.break_sources);
            self.pending = switch.exit_sources;
            return;
        };
        Self::enter(steps, body, Step::Switch(switch));
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

    /// Lowers the part of the loop `stmt` that precedes its bodies, which
    /// `label` names when the loop is the only statement of a labeled group.
    fn lower_loop_statement(
        &mut self,
        stmt: S::Stmt,
        label: Option<&'a str>,
        steps: &mut Steps<'a, H, S, B>,
    ) -> Result<(), LowerError<B::Error>> {
        match self.source.stmt(stmt) {
            StmtShape::While { condition, body } => {
                self.lower_while(stmt, condition, body, label, steps)
            }
            StmtShape::DoWhile { body, condition } => {
                self.lower_do_while(stmt, condition, body, label, steps);
                Ok(())
            }
            StmtShape::Loop { body } => {
                self.lower_loop(body, label, steps);
                Ok(())
            }
            StmtShape::For {
                initializer,
                condition,
                update,
                body,
            } => {
                self.adopt_forward_gotos(label);
                let tail = ForTail {
                    stmt,
                    condition,
                    update,
                    body,
                    label,
                    mark: self.materialized.len(),
                };
                Self::enter(steps, initializer, Step::ForInitializer(tail));
                Ok(())
            }
            _ => unreachable!("only a loop statement is lowered as a loop"),
        }
    }

    fn lower_labeled(
        &mut self,
        label: &'a str,
        body: &'a [S::Stmt],
        steps: &mut Steps<'a, H, S, B>,
    ) -> Result<(), LowerError<B::Error>> {
        if let Some(only) = labeled_loop::<H, S>(self.source, body) {
            return self.lower_loop_statement(only, Some(label), steps);
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
        Self::enter(steps, body, Step::Labeled);
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

    fn lower_region(
        &mut self,
        stmt: S::Stmt,
        op: &'a H::Op,
        operands: &[S::Expr],
        body: &'a [S::Stmt],
        steps: &mut Steps<'a, H, S, B>,
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
        Self::enter(steps, body, Step::Region { op, uses, origin });
        Ok(())
    }
}
