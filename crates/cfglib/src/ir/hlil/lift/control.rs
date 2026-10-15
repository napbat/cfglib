//! Statement translation over the control tree.
//!
//! The tree names flat instructions only through block spans. A run of
//! consecutive block leaves in one body translates as a single instruction
//! list, together with a return, branch, or dispatch that follows it, so that
//! expression inlining sees the whole linear region instead of one fragment
//! per block.

extern crate alloc;

use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use crate::block::BlockId;
use crate::dataflow::ProgramPoint;
use crate::graph::view::DenseId;
use crate::region;
use crate::{
    BlockSpan, CatchHandler, Cfg, ControlBody, ControlNode, ControlNodeId, ControlRun, LoopKind,
    SwitchCase,
};

use super::super::builder::FunctionBuilder;
use super::super::dialect::LiftDialect;
use super::super::flat::{FlatFunction, LiftEdge};
use super::super::shape::{Handler, HandlerKind, SwitchArm};
use super::super::view::Entity;
use super::block::Expect;
use super::scratch::{Scratch, Slot};
use super::{Context, Entry, Fallible, Lifter};

/// Appends the instructions of `span`, with their points, to `list`.
fn extend_entries<'a, I, E>(cfg: &'a Cfg<I, E>, list: &mut Vec<Entry<'a, I>>, span: BlockSpan) {
    list.extend(span.points().zip(span.instructions(cfg)));
}

/// The loop-continuation condition, negated when the raw condition exits on
/// true; `None` when negation is needed but unavailable.
///
/// The exact inversion comes first: the same operands under the negated
/// operation, rewritten in place with no wrapper node.
fn loop_condition<H: LiftDialect, V: DenseId>(
    scratch: &mut Scratch<H, V>,
    condition: Slot,
    exit_on_true: bool,
) -> Option<Slot> {
    if !exit_on_true {
        return Some(condition);
    }
    if scratch.rewrite_operation(condition, H::negate_operation) {
        return Some(condition);
    }
    let negation = H::logical_not()?;
    let ty = scratch.node(condition).ty.clone();
    Some(scratch.operation(negation, vec![condition], ty))
}

impl<'a, H, X, B> Lifter<'a, H, X, B>
where
    H: LiftDialect,
    X: FlatFunction<H>,
    B: FunctionBuilder<H>,
{
    /// The statements of one body, appended to `out`.
    fn translate_body(&mut self, body: ControlBody, out: &mut Vec<B::Stmt>) -> Fallible<()> {
        let tree = self.tree;
        self.translate_nodes(tree.body(body), out)
    }

    fn translate_nodes(&mut self, nodes: &[ControlNodeId], out: &mut Vec<B::Stmt>) -> Fallible<()> {
        let tree = self.tree;
        let cfg = self.cfg;
        let mut index = 0;
        while index < nodes.len() {
            let mut list: Vec<Entry<'a, X::Inst>> = Vec::new();
            let mut final_block = None;
            let mut end = index;
            while let Some(&ControlNode::Block(span)) = nodes.get(end).map(|&id| tree.node(id)) {
                extend_entries(cfg, &mut list, span);
                final_block = Some(span.block());
                end += 1;
            }
            let Some(final_block) = final_block else {
                self.translate_node(nodes[index], out)?;
                index += 1;
                continue;
            };
            // The run absorbs a directly following return or decision list.
            // Liveness beyond the merged list is the live-out of its last
            // block: a definition that feeds a later block of the same run
            // is still single-use inside the list.
            match nodes.get(end).map(|&id| *tree.node(id)) {
                Some(ControlNode::Return(span)) => {
                    extend_entries(cfg, &mut list, span);
                    let (statements, _) =
                        self.translate_list(span.block(), &list, Expect::Statements)?;
                    out.extend(statements);
                    index = end + 1;
                }
                Some(ControlNode::IfThenElse {
                    condition,
                    then_body,
                    else_body,
                }) => {
                    extend_entries(cfg, &mut list, condition);
                    self.translate_if(condition.block(), &list, then_body, else_body, out)?;
                    index = end + 1;
                }
                Some(ControlNode::Switch {
                    condition,
                    cases,
                    default_body,
                    ..
                }) => {
                    extend_entries(cfg, &mut list, condition);
                    self.translate_switch(condition.block(), &list, cases, default_body, out)?;
                    index = end + 1;
                }
                _ => {
                    let (statements, _) =
                        self.translate_list(final_block, &list, Expect::Statements)?;
                    out.extend(statements);
                    index = end;
                }
            }
        }
        Ok(())
    }

    /// The statements of one node, appended to `out`.
    pub(super) fn translate_node(
        &mut self,
        node: ControlNodeId,
        out: &mut Vec<B::Stmt>,
    ) -> Fallible<()> {
        let tree = self.tree;
        let cfg = self.cfg;
        match *tree.node(node) {
            ControlNode::Sequence(body) => self.translate_body(body, out),
            ControlNode::Block(span) | ControlNode::Return(span) => {
                let mut list = Vec::new();
                extend_entries(cfg, &mut list, span);
                let (statements, _) =
                    self.translate_list(span.block(), &list, Expect::Statements)?;
                out.extend(statements);
                Ok(())
            }
            ControlNode::IfThenElse {
                condition,
                then_body,
                else_body,
            } => {
                let mut list = Vec::new();
                extend_entries(cfg, &mut list, condition);
                self.translate_if(condition.block(), &list, then_body, else_body, out)
            }
            ControlNode::Loop { header, kind, body } => {
                let statement = self.translate_loop(header, kind, body, None)?;
                out.push(statement);
                Ok(())
            }
            ControlNode::Switch {
                condition,
                cases,
                default_body,
                ..
            } => {
                let mut list = Vec::new();
                extend_entries(cfg, &mut list, condition);
                self.translate_switch(condition.block(), &list, cases, default_body, out)
            }
            ControlNode::Break { label } => {
                let label = match label {
                    Some(block) => Some(self.block_label(block)),
                    None => self.break_label(),
                };
                out.push(self.builder.break_(label.as_deref()));
                Ok(())
            }
            ControlNode::Continue { label } => {
                let label = label.map(|block| self.block_label(block));
                out.push(self.builder.continue_(label.as_deref()));
                Ok(())
            }
            ControlNode::Label { block, body } => {
                let name = self.block_label(block);
                // A label that wraps exactly one loop names that loop, so a
                // labeled break resolves without a second wrapper.
                if let [only] = tree.body(body)
                    && let ControlNode::Loop { header, kind, body } = *tree.node(*only)
                {
                    let statement = self.translate_loop(header, kind, body, Some(&name))?;
                    out.push(statement);
                    return Ok(());
                }
                let mut statements = Vec::new();
                self.translate_body(body, &mut statements)?;
                out.push(self.builder.labeled(&name, &statements));
                Ok(())
            }
            ControlNode::Goto { target } => {
                let label = self.block_label(target);
                out.push(self.builder.goto(&label));
                Ok(())
            }
            ControlNode::TryCatch {
                try_body,
                handlers,
                finally_body,
            } => {
                let statement = self.translate_try(try_body, handlers, finally_body)?;
                out.push(statement);
                Ok(())
            }
            ControlNode::Guarded { .. } => {
                Err(String::from("predicated regions have no HLIL translation"))
            }
        }
    }

    fn translate_if(
        &mut self,
        block: BlockId,
        list: &[Entry<'a, X::Inst>],
        then_body: ControlBody,
        else_body: ControlBody,
        out: &mut Vec<B::Stmt>,
    ) -> Fallible<()> {
        let (statements, end) = self.translate_list(block, list, Expect::Branch)?;
        out.extend(statements);
        let end = end.expect("a branch list ends in its branch");
        let condition = self.emit(&end.scratch, end.value);
        let mut then_statements = Vec::new();
        self.translate_body(then_body, &mut then_statements)?;
        let mut else_statements = Vec::new();
        self.translate_body(else_body, &mut else_statements)?;
        let statement = self
            .builder
            .if_(condition, &then_statements, &else_statements);
        self.record(end.point, Entity::Statement(statement));
        out.push(statement);
        Ok(())
    }

    fn translate_try(
        &mut self,
        try_body: ControlBody,
        handlers: ControlRun<CatchHandler>,
        finally_body: ControlBody,
    ) -> Fallible<B::Stmt> {
        let tree = self.tree;
        let mut body = Vec::new();
        self.translate_body(try_body, &mut body)?;
        let mut kinds = Vec::new();
        let mut handler_bodies: Vec<Vec<B::Stmt>> = Vec::new();
        for handler in tree.handlers(handlers) {
            kinds.push(match handler.kind {
                region::HandlerKind::Catch => HandlerKind::Catch,
                region::HandlerKind::CatchAll => HandlerKind::CatchAll,
                region::HandlerKind::Fault => HandlerKind::Fault,
                // The funclet is emitted separately by the sweep, so its
                // predicate has no structured body here.
                region::HandlerKind::Filter { .. } => HandlerKind::Filter { filter_body: &[] },
                region::HandlerKind::Finally => {
                    return Err(String::from(
                        "a finally handler arm outside the finally body",
                    ));
                }
            });
            let mut statements = Vec::new();
            self.translate_body(handler.body, &mut statements)?;
            handler_bodies.push(statements);
        }
        let mut finally_statements = Vec::new();
        self.translate_body(finally_body, &mut finally_statements)?;
        let handlers: Vec<Handler<'_, H, B::Stmt, B::Var>> = kinds
            .into_iter()
            .zip(&handler_bodies)
            .map(|(kind, body)| Handler {
                kind,
                binding: None,
                caught_types: &[],
                body,
            })
            .collect();
        Ok(self.builder.try_(&body, &handlers, &finally_statements))
    }

    /// The explicit label that an unlabeled break of the tree needs.
    ///
    /// A plain break of the tree always exits the innermost **loop**, while an
    /// unlabeled HLIL break exits the innermost loop *or switch*. A break
    /// inside an intervening switch therefore names its loop.
    fn break_label(&mut self) -> Option<String> {
        let mut crossed_switch = false;
        for context in self.contexts.iter_mut().rev() {
            match &context.loop_label {
                None => crossed_switch = true,
                Some(label) => {
                    if crossed_switch {
                        context.needs_label = true;
                        return Some(label.clone());
                    }
                    return None;
                }
            }
        }
        None
    }

    fn translate_switch(
        &mut self,
        block: BlockId,
        list: &[Entry<'a, X::Inst>],
        cases: ControlRun<SwitchCase>,
        default_body: ControlBody,
        out: &mut Vec<B::Stmt>,
    ) -> Fallible<()> {
        let tree = self.tree;
        let cfg = self.cfg;
        let (statements, end) = self.translate_list(block, list, Expect::Switch)?;
        out.extend(statements);
        let end = end.expect("a dispatch list ends in its dispatch");
        let scrutinee = self.emit(&end.scratch, end.value);
        self.contexts.push(Context {
            loop_label: None,
            needs_label: false,
        });
        let mut arm_values: Vec<Vec<H::Constant>> = Vec::new();
        let mut arm_bodies: Vec<Vec<B::Stmt>> = Vec::new();
        for case in tree.cases(cases) {
            let mut values = Vec::new();
            for &edge in tree.edges(case.edges) {
                values.extend(
                    cfg.edge(edge)
                        .data()
                        .payload()
                        .case_values()
                        .iter()
                        .cloned(),
                );
            }
            if values.is_empty() {
                return Err(format!(
                    "switch case at block {} has no case values",
                    case.block
                ));
            }
            let mut body = Vec::new();
            self.translate_body(case.body, &mut body)?;
            arm_values.push(values);
            arm_bodies.push(body);
        }
        let mut default_statements = Vec::new();
        self.translate_body(default_body, &mut default_statements)?;
        self.contexts.pop();
        let arms: Vec<SwitchArm<'_, H, B::Stmt>> = arm_values
            .iter()
            .zip(&arm_bodies)
            .map(|(values, body)| SwitchArm { values, body })
            .collect();
        let statement = self.builder.switch(scrutinee, &arms, &default_statements);
        self.record(end.point, Entity::Statement(statement));
        out.push(statement);
        Ok(())
    }

    /// One loop, wrapped in a label when `outer_label` names it or an
    /// unlabeled break inside a switch had to.
    fn translate_loop(
        &mut self,
        header: BlockId,
        kind: LoopKind,
        body: ControlBody,
        outer_label: Option<&str>,
    ) -> Fallible<B::Stmt> {
        let label = outer_label.map_or_else(|| self.block_label(header), String::from);
        self.contexts.push(Context {
            loop_label: Some(label.clone()),
            needs_label: false,
        });
        let lifted = self.translate_loop_kind(kind, body);
        let context = self
            .contexts
            .pop()
            .expect("the loop context pushed above is still on the stack");
        let statement = lifted?;
        if context.needs_label || outer_label.is_some() {
            return Ok(self.builder.labeled(&label, &[statement]));
        }
        Ok(statement)
    }

    fn translate_loop_kind(&mut self, kind: LoopKind, body: ControlBody) -> Fallible<B::Stmt> {
        let tree = self.tree;
        let cfg = self.cfg;
        match kind {
            LoopKind::Endless => {
                let mut statements = Vec::new();
                self.translate_body(body, &mut statements)?;
                Ok(self.builder.loop_(&statements))
            }
            LoopKind::While {
                condition,
                exit_on_true,
            } => {
                let spans = tree.spans(condition);
                let mut list = Vec::new();
                for &span in spans {
                    extend_entries(cfg, &mut list, span);
                }
                let condition_block = spans
                    .last()
                    .ok_or_else(|| String::from("a pre-tested loop has no condition block"))?
                    .block();
                let (condition_statements, end) =
                    self.translate_list(condition_block, &list, Expect::Branch)?;
                let mut end = end.expect("a branch list ends in its branch");
                let mut body_statements = Vec::new();
                self.translate_body(body, &mut body_statements)?;
                if condition_statements.is_empty()
                    && let Some(value) =
                        loop_condition::<H, X::Var>(&mut end.scratch, end.value, exit_on_true)
                {
                    let condition = self.emit(&end.scratch, value);
                    let statement = self.builder.while_(condition, &body_statements);
                    self.record(end.point, Entity::Statement(statement));
                    return Ok(statement);
                }
                // The condition needs statements, or its negation is
                // unavailable: state the test explicitly at the loop top.
                let test = self.exit_test(&end.scratch, end.value, exit_on_true, end.point);
                let mut loop_body = condition_statements;
                loop_body.push(test);
                loop_body.extend(body_statements);
                Ok(self.builder.loop_(&loop_body))
            }
            LoopKind::DoWhile {
                latch,
                continue_on_true,
            } => {
                let mut body_statements = Vec::new();
                self.translate_body(body, &mut body_statements)?;
                let mut list = Vec::new();
                extend_entries(cfg, &mut list, latch);
                let (condition_statements, end) =
                    self.translate_list(latch.block(), &list, Expect::Branch)?;
                let mut end = end.expect("a branch list ends in its branch");
                if condition_statements.is_empty()
                    && let Some(value) =
                        loop_condition::<H, X::Var>(&mut end.scratch, end.value, !continue_on_true)
                {
                    let condition = self.emit(&end.scratch, value);
                    let statement = self.builder.do_while(&body_statements, condition);
                    self.record(end.point, Entity::Statement(statement));
                    return Ok(statement);
                }
                // State the post-test explicitly at the loop bottom.
                let test = self.exit_test(&end.scratch, end.value, !continue_on_true, end.point);
                let mut loop_body = body_statements;
                loop_body.extend(condition_statements);
                loop_body.push(test);
                Ok(self.builder.loop_(&loop_body))
            }
        }
    }

    /// An explicit `if (...) break;` exit test with the given polarity.
    fn exit_test(
        &mut self,
        scratch: &Scratch<H, X::Var>,
        condition: Slot,
        exit_on_true: bool,
        point: ProgramPoint,
    ) -> B::Stmt {
        let condition = self.emit(scratch, condition);
        let break_statement = self.builder.break_(None);
        let (then_body, else_body): (&[B::Stmt], &[B::Stmt]) = if exit_on_true {
            (&[break_statement], &[])
        } else {
            (&[], &[break_statement])
        };
        let statement = self.builder.if_(condition, then_body, else_body);
        self.record(point, Entity::Statement(statement));
        statement
    }
}
