//! Statement translation over the control tree.
//!
//! The tree names flat instructions only through block spans. A run of
//! consecutive block leaves in one body translates as a single instruction
//! list, together with a return, branch, or dispatch that follows it, so that
//! expression inlining sees the whole linear region instead of one fragment
//! per block.
//!
//! The walk keeps its own stacks, so the nesting depth of the tree costs no
//! call frames. A body that is being translated and a construct that waits
//! for its child bodies are frames of one explicit stack. The statements of
//! every open body sit on one shared stack, and a construct hands the builder
//! the slices that hold its children, then replaces them with its own
//! statement. The builder therefore sees its calls in the order of the
//! recursive formulation: a construct's own list, then its bodies, then the
//! construct.

extern crate alloc;

use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use crate::ProgramPoint;
use crate::block::BlockId;
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
use super::block::{Expect, ListEnd};
use super::scratch::{Scratch, Slot};
use super::{Context, Entry, Fallible, Lifter};

/// Appends the instructions of `span`, with their points, to `list`.
fn extend_entries<'a, I, E>(cfg: &'a Cfg<I, E>, list: &mut Vec<Entry<'a, I>>, span: BlockSpan) {
    list.extend(span.points().zip(span.instructions(cfg)));
}

/// The structured class of a handler arm, or `None` for a `finally`
/// handler, which belongs to the finally body and not to an arm.
fn handler_kind<'s, S>(kind: region::HandlerKind) -> Option<HandlerKind<'s, S>> {
    Some(match kind {
        region::HandlerKind::Catch => HandlerKind::Catch,
        region::HandlerKind::CatchAll => HandlerKind::CatchAll,
        region::HandlerKind::Fault => HandlerKind::Fault,
        // The funclet is emitted separately by the sweep, so its predicate
        // has no structured body here.
        region::HandlerKind::Filter { .. } => HandlerKind::Filter { filter_body: &[] },
        region::HandlerKind::Finally => return None,
    })
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

/// The state of the walk over the tree.
pub(super) struct Walk<'a, H, X, B>
where
    H: LiftDialect,
    X: FlatFunction<H>,
    B: FunctionBuilder<H>,
{
    frames: Vec<Frame<'a, H, X, B>>,
    /// The statements of every open body, the innermost body last. The root
    /// body is what remains when the walk ends.
    pub(super) statements: Vec<B::Stmt>,
    /// Where on `statements` each finished child body of the open constructs
    /// ends, in the order that the bodies finished.
    ends: Vec<usize>,
}

impl<H, X, B> Walk<'_, H, X, B>
where
    H: LiftDialect,
    X: FlatFunction<H>,
    B: FunctionBuilder<H>,
{
    pub(super) const fn new() -> Self {
        Self {
            frames: Vec::new(),
            statements: Vec::new(),
            ends: Vec::new(),
        }
    }
}

/// One step that waits for the frames above it.
enum Frame<'a, H, X, B>
where
    H: LiftDialect,
    X: FlatFunction<H>,
    B: FunctionBuilder<H>,
{
    /// The nodes of one body from `index` on.
    Nodes {
        nodes: &'a [ControlNodeId],
        index: usize,
    },
    /// A construct that waits for its child bodies.
    Open(Open<'a, H, X, B>),
}

/// A construct whose child bodies are being translated.
struct Open<'a, H, X, B>
where
    H: LiftDialect,
    X: FlatFunction<H>,
    B: FunctionBuilder<H>,
{
    kind: Kind<'a, H, X, B>,
    /// The height of the statement stack where its first body begins.
    first: usize,
    /// The number of its bodies that have been started, and one more once
    /// every body has finished.
    started: usize,
}

/// What a construct needs, besides its bodies, to become a statement.
enum Kind<'a, H, X, B>
where
    H: LiftDialect,
    X: FlatFunction<H>,
    B: FunctionBuilder<H>,
{
    If {
        point: ProgramPoint,
        condition: B::Expr,
        then_body: ControlBody,
        else_body: ControlBody,
    },
    Switch {
        point: ProgramPoint,
        scrutinee: B::Expr,
        cases: &'a [SwitchCase],
        /// The case values of each arm that has started.
        values: Vec<Vec<H::Constant>>,
        default_body: ControlBody,
    },
    Loop {
        label: String,
        /// Whether the label was the tree's own: a label that wraps the loop
        /// is kept even when no break needs it.
        labeled: bool,
        tail: LoopTail<H, X, B>,
        body: ControlBody,
    },
    Try {
        handlers: &'a [CatchHandler],
        try_body: ControlBody,
        finally_body: ControlBody,
    },
    Label {
        name: String,
        body: ControlBody,
    },
}

/// What a loop owns besides its body.
enum LoopTail<H, X, B>
where
    H: LiftDialect,
    X: FlatFunction<H>,
    B: FunctionBuilder<H>,
{
    Endless,
    /// A pre-tested loop, whose condition list was translated before the
    /// body.
    While {
        condition_statements: Vec<B::Stmt>,
        end: ListEnd<H, X::Var>,
        exit_on_true: bool,
    },
    /// A post-tested loop, whose latch is translated after the body.
    DoWhile {
        latch: BlockSpan,
        continue_on_true: bool,
    },
}

impl<'a, H, X, B> Lifter<'a, H, X, B>
where
    H: LiftDialect,
    X: FlatFunction<H>,
    B: FunctionBuilder<H>,
{
    /// Translates the whole tree, leaving the statements of its root on
    /// `walk.statements`.
    pub(super) fn translate_tree(&mut self) -> Fallible<()> {
        let tree = self.tree;
        if let Some(frame) = self.enter_node(tree.root())? {
            self.walk.frames.push(frame);
        }
        while let Some(frame) = self.walk.frames.pop() {
            match frame {
                Frame::Nodes { nodes, index } => self.step_nodes(nodes, index)?,
                Frame::Open(mut open) => {
                    if let Some(body) = self.next_body(&mut open)? {
                        self.walk.frames.push(Frame::Open(open));
                        self.walk.frames.push(Frame::Nodes {
                            nodes: tree.body(body),
                            index: 0,
                        });
                    } else {
                        self.finish(open)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Translates the nodes of one body from `index` on, up to the first
    /// construct that has bodies of its own. That construct becomes a frame
    /// above the rest of this body.
    fn step_nodes(&mut self, nodes: &'a [ControlNodeId], mut index: usize) -> Fallible<()> {
        let tree = self.tree;
        let cfg = self.cfg;
        while index < nodes.len() {
            let mut list: Vec<Entry<'a, X::Inst>> = Vec::new();
            let mut final_block = None;
            let mut end = index;
            while let Some(&ControlNode::Block(span)) = nodes.get(end).map(|&id| tree.node(id)) {
                extend_entries(cfg, &mut list, span);
                final_block = Some(span.block());
                end += 1;
            }
            let opened = if let Some(final_block) = final_block {
                // The run absorbs a directly following return or decision
                // list. Liveness beyond the merged list is the live-out of its
                // last block: a definition that feeds a later block of the
                // same run is still single-use inside the list.
                match nodes.get(end).map(|&id| *tree.node(id)) {
                    Some(ControlNode::Return(span)) => {
                        extend_entries(cfg, &mut list, span);
                        self.push_statements(span.block(), &list)?;
                        index = end + 1;
                        None
                    }
                    Some(ControlNode::IfThenElse {
                        condition,
                        then_body,
                        else_body,
                    }) => {
                        extend_entries(cfg, &mut list, condition);
                        index = end + 1;
                        Some(self.open_if(condition.block(), &list, then_body, else_body)?)
                    }
                    Some(ControlNode::Switch {
                        condition,
                        cases,
                        default_body,
                        ..
                    }) => {
                        extend_entries(cfg, &mut list, condition);
                        index = end + 1;
                        Some(self.open_switch(condition.block(), &list, cases, default_body)?)
                    }
                    _ => {
                        self.push_statements(final_block, &list)?;
                        index = end;
                        None
                    }
                }
            } else {
                let node = nodes[index];
                index += 1;
                self.enter_node(node)?
            };
            if let Some(frame) = opened {
                self.walk.frames.push(Frame::Nodes { nodes, index });
                self.walk.frames.push(frame);
                return Ok(());
            }
        }
        Ok(())
    }

    /// Translates `list` as plain statements onto the statement stack.
    fn push_statements(&mut self, block: BlockId, list: &[Entry<'a, X::Inst>]) -> Fallible<()> {
        let (statements, _) = self.translate_list(block, list, Expect::Statements)?;
        self.walk.statements.extend(statements);
        Ok(())
    }

    /// The statements of one node, pushed on the statement stack, or the
    /// frame of a node that has bodies to translate first.
    fn enter_node(&mut self, node: ControlNodeId) -> Fallible<Option<Frame<'a, H, X, B>>> {
        let tree = self.tree;
        let cfg = self.cfg;
        match *tree.node(node) {
            ControlNode::Sequence(body) => {
                return Ok(Some(Frame::Nodes {
                    nodes: tree.body(body),
                    index: 0,
                }));
            }
            ControlNode::Block(span) | ControlNode::Return(span) => {
                let mut list = Vec::new();
                extend_entries(cfg, &mut list, span);
                self.push_statements(span.block(), &list)?;
            }
            ControlNode::IfThenElse {
                condition,
                then_body,
                else_body,
            } => {
                let mut list = Vec::new();
                extend_entries(cfg, &mut list, condition);
                return self
                    .open_if(condition.block(), &list, then_body, else_body)
                    .map(Some);
            }
            ControlNode::Loop { header, kind, body } => {
                return self.open_loop(header, kind, body, None).map(Some);
            }
            ControlNode::Switch {
                condition,
                cases,
                default_body,
                ..
            } => {
                let mut list = Vec::new();
                extend_entries(cfg, &mut list, condition);
                return self
                    .open_switch(condition.block(), &list, cases, default_body)
                    .map(Some);
            }
            ControlNode::Break { label } => {
                let label = match label {
                    Some(block) => Some(self.block_label(block)),
                    None => self.break_label(),
                };
                let statement = self.builder.break_(label.as_deref());
                self.walk.statements.push(statement);
            }
            ControlNode::Continue { label } => {
                let label = label.map(|block| self.block_label(block));
                let statement = self.builder.continue_(label.as_deref());
                self.walk.statements.push(statement);
            }
            ControlNode::Label { block, body } => {
                let name = self.block_label(block);
                // A label that wraps exactly one loop names that loop, so a
                // labeled break resolves without a second wrapper.
                if let [only] = tree.body(body)
                    && let ControlNode::Loop { header, kind, body } = *tree.node(*only)
                {
                    return self.open_loop(header, kind, body, Some(name)).map(Some);
                }
                return Ok(Some(self.open(Kind::Label { name, body })));
            }
            ControlNode::Goto { target } => {
                let label = self.block_label(target);
                let statement = self.builder.goto(&label);
                self.walk.statements.push(statement);
            }
            ControlNode::TryCatch {
                try_body,
                handlers,
                finally_body,
            } => {
                return Ok(Some(self.open(Kind::Try {
                    handlers: tree.handlers(handlers),
                    try_body,
                    finally_body,
                })));
            }
            ControlNode::Guarded { .. } => {
                return Err(String::from("predicated regions have no HLIL translation"));
            }
        }
        Ok(None)
    }

    /// The frame of a construct whose first body starts on top of the
    /// statement stack.
    fn open(&self, kind: Kind<'a, H, X, B>) -> Frame<'a, H, X, B> {
        Frame::Open(Open {
            kind,
            first: self.walk.statements.len(),
            started: 0,
        })
    }

    /// Translates the condition list of an `if` and opens it.
    fn open_if(
        &mut self,
        block: BlockId,
        list: &[Entry<'a, X::Inst>],
        then_body: ControlBody,
        else_body: ControlBody,
    ) -> Fallible<Frame<'a, H, X, B>> {
        let (statements, end) = self.translate_list(block, list, Expect::Branch)?;
        self.walk.statements.extend(statements);
        let end = end.expect("a branch list ends in its branch");
        let condition = self.emit(&end.scratch, end.value);
        Ok(self.open(Kind::If {
            point: end.point,
            condition,
            then_body,
            else_body,
        }))
    }

    /// Translates the dispatch list of a `switch` and opens it. Its context
    /// stays on the stack until the switch finishes.
    fn open_switch(
        &mut self,
        block: BlockId,
        list: &[Entry<'a, X::Inst>],
        cases: ControlRun<SwitchCase>,
        default_body: ControlBody,
    ) -> Fallible<Frame<'a, H, X, B>> {
        let tree = self.tree;
        let (statements, end) = self.translate_list(block, list, Expect::Switch)?;
        self.walk.statements.extend(statements);
        let end = end.expect("a dispatch list ends in its dispatch");
        let scrutinee = self.emit(&end.scratch, end.value);
        self.contexts.push(Context {
            loop_label: None,
            needs_label: false,
        });
        Ok(self.open(Kind::Switch {
            point: end.point,
            scrutinee,
            cases: tree.cases(cases),
            values: Vec::new(),
            default_body,
        }))
    }

    /// Opens one loop, wrapped in a label when `outer_label` names it or an
    /// unlabeled break inside a switch has to. The condition list of a
    /// pre-tested loop translates first. The loop's context stays on the stack
    /// until the loop finishes.
    fn open_loop(
        &mut self,
        header: BlockId,
        kind: LoopKind,
        body: ControlBody,
        outer_label: Option<String>,
    ) -> Fallible<Frame<'a, H, X, B>> {
        let tree = self.tree;
        let cfg = self.cfg;
        let labeled = outer_label.is_some();
        let label = outer_label.unwrap_or_else(|| self.block_label(header));
        self.contexts.push(Context {
            loop_label: Some(label.clone()),
            needs_label: false,
        });
        let tail = match kind {
            LoopKind::Endless => LoopTail::Endless,
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
                LoopTail::While {
                    condition_statements,
                    end: end.expect("a branch list ends in its branch"),
                    exit_on_true,
                }
            }
            LoopKind::DoWhile {
                latch,
                continue_on_true,
            } => LoopTail::DoWhile {
                latch,
                continue_on_true,
            },
        };
        Ok(self.open(Kind::Loop {
            label,
            labeled,
            tail,
            body,
        }))
    }

    /// Starts the next body of `open`, or returns `None` when every body has
    /// finished. A body that starts after the first records where the one
    /// before it ended.
    fn next_body(&mut self, open: &mut Open<'a, H, X, B>) -> Fallible<Option<ControlBody>> {
        let stage = open.started;
        open.started += 1;
        if stage > 0 {
            self.walk.ends.push(self.walk.statements.len());
        }
        let tree = self.tree;
        let cfg = self.cfg;
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
            Kind::Loop { body, .. } | Kind::Label { body, .. } => (stage == 0).then_some(*body),
            Kind::Switch {
                cases,
                values,
                default_body,
                ..
            } => {
                if let Some(case) = cases.get(stage) {
                    let mut case_values = Vec::new();
                    for &edge in tree.edges(case.edges) {
                        case_values.extend(
                            cfg.edge(edge)
                                .data()
                                .payload()
                                .case_values()
                                .iter()
                                .cloned(),
                        );
                    }
                    if case_values.is_empty() {
                        return Err(format!(
                            "switch case at block {} has no case values",
                            case.block
                        ));
                    }
                    values.push(case_values);
                    Some(case.body)
                } else {
                    (stage == cases.len()).then_some(*default_body)
                }
            }
            Kind::Try {
                handlers,
                try_body,
                finally_body,
            } => match stage {
                0 => Some(*try_body),
                _ if stage <= handlers.len() => {
                    let handler = &handlers[stage - 1];
                    if handler_kind::<B::Stmt>(handler.kind).is_none() {
                        return Err(String::from(
                            "a finally handler arm outside the finally body",
                        ));
                    }
                    Some(handler.body)
                }
                _ => (stage == handlers.len() + 1).then_some(*finally_body),
            },
        })
    }

    /// Turns a construct whose bodies have all finished into its statement,
    /// in place of them on the statement stack.
    fn finish(&mut self, open: Open<'a, H, X, B>) -> Fallible<()> {
        let Open {
            kind,
            first,
            started,
        } = open;
        // The final call of `next_body` recorded the end of the last body.
        let base = self.walk.ends.len() - (started - 1);
        let statement = match kind {
            Kind::If {
                point, condition, ..
            } => {
                let then_end = self.walk.ends[base];
                let else_end = self.walk.ends[base + 1];
                let statements = &self.walk.statements;
                let statement = self.builder.if_(
                    condition,
                    &statements[first..then_end],
                    &statements[then_end..else_end],
                );
                self.record(point, Entity::Statement(statement));
                statement
            }
            Kind::Switch {
                point,
                scrutinee,
                values,
                ..
            } => {
                self.contexts.pop();
                let ends = &self.walk.ends[base..];
                let statements = &self.walk.statements;
                let mut start = first;
                let mut arms: Vec<SwitchArm<'_, H, B::Stmt>> = Vec::with_capacity(values.len());
                for (values, &end) in values.iter().zip(ends) {
                    arms.push(SwitchArm {
                        values,
                        body: &statements[start..end],
                    });
                    start = end;
                }
                let default = &statements[start..ends[values.len()]];
                let statement = self.builder.switch(scrutinee, &arms, default);
                self.record(point, Entity::Statement(statement));
                statement
            }
            Kind::Loop {
                label,
                labeled,
                tail,
                ..
            } => self.finish_loop(&label, labeled, tail, first)?,
            Kind::Try { handlers, .. } => {
                let ends = &self.walk.ends[base..];
                let statements = &self.walk.statements;
                let rebuilt: Vec<Handler<'_, H, B::Stmt, B::Var>> = handlers
                    .iter()
                    .enumerate()
                    .map(|(index, handler)| Handler {
                        kind: handler_kind(handler.kind)
                            .expect("a finally handler arm is rejected when its body starts"),
                        binding: None,
                        caught_types: &[],
                        body: &statements[ends[index]..ends[index + 1]],
                    })
                    .collect();
                let finally = ends.len() - 1;
                self.builder.try_(
                    &statements[first..ends[0]],
                    &rebuilt,
                    &statements[ends[finally - 1]..ends[finally]],
                )
            }
            Kind::Label { name, .. } => self.builder.labeled(&name, &self.walk.statements[first..]),
        };
        self.walk.ends.truncate(base);
        self.walk.statements.truncate(first);
        self.walk.statements.push(statement);
        Ok(())
    }

    /// The statement of a loop whose body sits on the statement stack from
    /// `first`, wrapped in a label when `labeled` or an unlabeled break of the
    /// tree had to name the loop.
    fn finish_loop(
        &mut self,
        label: &str,
        labeled: bool,
        tail: LoopTail<H, X, B>,
        first: usize,
    ) -> Fallible<B::Stmt> {
        let statement = match tail {
            LoopTail::Endless => self.builder.loop_(&self.walk.statements[first..]),
            LoopTail::While {
                condition_statements,
                mut end,
                exit_on_true,
            } => {
                if condition_statements.is_empty()
                    && let Some(value) =
                        loop_condition::<H, X::Var>(&mut end.scratch, end.value, exit_on_true)
                {
                    let condition = self.emit(&end.scratch, value);
                    let statement = self
                        .builder
                        .while_(condition, &self.walk.statements[first..]);
                    self.record(end.point, Entity::Statement(statement));
                    statement
                } else {
                    // The condition needs statements, or its negation is
                    // unavailable: state the test explicitly at the loop top.
                    let test = self.exit_test(&end.scratch, end.value, exit_on_true, end.point);
                    let mut loop_body = condition_statements;
                    loop_body.push(test);
                    loop_body.extend_from_slice(&self.walk.statements[first..]);
                    self.builder.loop_(&loop_body)
                }
            }
            LoopTail::DoWhile {
                latch,
                continue_on_true,
            } => {
                let mut list = Vec::new();
                extend_entries(self.cfg, &mut list, latch);
                let (condition_statements, end) =
                    self.translate_list(latch.block(), &list, Expect::Branch)?;
                let mut end = end.expect("a branch list ends in its branch");
                if condition_statements.is_empty()
                    && let Some(value) =
                        loop_condition::<H, X::Var>(&mut end.scratch, end.value, !continue_on_true)
                {
                    let condition = self.emit(&end.scratch, value);
                    let statement = self
                        .builder
                        .do_while(&self.walk.statements[first..], condition);
                    self.record(end.point, Entity::Statement(statement));
                    statement
                } else {
                    // State the post-test explicitly at the loop bottom.
                    let test =
                        self.exit_test(&end.scratch, end.value, !continue_on_true, end.point);
                    let mut loop_body = self.walk.statements[first..].to_vec();
                    loop_body.extend(condition_statements);
                    loop_body.push(test);
                    self.builder.loop_(&loop_body)
                }
            }
        };
        let context = self
            .contexts
            .pop()
            .expect("the loop context pushed on opening is still on the stack");
        if context.needs_label || labeled {
            return Ok(self.builder.labeled(label, &[statement]));
        }
        Ok(statement)
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
