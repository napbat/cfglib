//! Predicated-region wrapping over a structured tree.

extern crate alloc;
use alloc::vec::Vec;
use core::slice;

use super::node::{
    BlockSpan, CatchHandler, ControlBody, ControlNode, ControlNodeId, ControlRun, LoopKind,
    SwitchCase,
};
use super::report::LiftReport;
use super::tree::{ChildBodies, ControlTree, Tables};
use crate::cfg::Cfg;
use crate::edge::EdgeId;
use crate::{InstrInfo, Predicated, ProgramPoint};

/// A maximal run of instructions sharing one predicate.
struct PredicateRun<I: InstrInfo> {
    predicate: Option<(I::Variable, bool)>,
    span: BlockSpan,
}

impl ControlTree {
    /// Structure a [`Cfg`] and regionize its predicated instructions into
    /// [`ControlNode::Guarded`] nodes.
    ///
    /// Runs [`compute`](Self::compute), then wraps every maximal run of
    /// instructions sharing the same [`Predicated::predicate`] into a
    /// `Guarded` node whose witness is the run's first instruction.
    /// Unpredicated instructions stay in plain blocks. Predicated runs that
    /// land in a branch/dispatch header (`IfThenElse`/`Switch` `condition`)
    /// are hoisted into guarded segments before the node.
    ///
    /// Two ledgered limits: a predicate on the branch/dispatch instruction
    /// itself stays inline (unrepresentable as a region), and predicated runs
    /// inside a loop's condition ([`LoopKind`](super::LoopKind)) are not
    /// regionized — a condition re-evaluates every iteration, so hoisting
    /// would change its semantics.
    ///
    /// Splitting a block yields several [`BlockSpan`]s of it, so the tree
    /// names one block in more than one leaf.
    #[must_use]
    pub fn compute_predicated<I: Predicated, E>(cfg: &Cfg<I, E>) -> Self {
        Self::compute_predicated_with_report(cfg).0
    }

    /// Structure a [`Cfg`], regionize its predicated instructions, and report
    /// which parts degraded to unstructured flow.
    ///
    /// The report is the one [`compute_with_report`](Self::compute_with_report)
    /// gives; regionizing changes no control structure.
    #[must_use]
    pub fn compute_predicated_with_report<I: Predicated, E>(cfg: &Cfg<I, E>) -> (Self, LiftReport) {
        let (tree, report) = Self::compute_with_report(cfg);
        let mut regionizer = Regionizer {
            cfg,
            old: &tree,
            new: Tables::new(),
        };
        let root = regionizer.wrap_root(tree.root());
        let root = regionizer.finish_root(root);
        (regionizer.new.finish(root), report)
    }
}

/// A rewrite of one tree into another with predicated runs wrapped.
struct Regionizer<'a, I, E> {
    cfg: &'a Cfg<I, E>,
    old: &'a ControlTree,
    new: Tables,
}

/// What the rewrite keeps while it wraps one body or one node.
enum Frame<'a> {
    /// The statements of one old body, wrapped in order into `nodes`.
    Body {
        remaining: slice::Iter<'a, ControlNodeId>,
        nodes: Vec<ControlNode>,
    },
    /// The child bodies of one old node.
    Node(Wrapping<'a>),
}

/// A node being wrapped, between the bodies of its children.
struct Wrapping<'a> {
    /// The node in the old tree.
    node: ControlNode,
    /// The old child bodies still to wrap, in execution order.
    children: ChildBodies<'a>,
    /// The guarded segments of a branch's header, which come before it.
    prefix: Vec<ControlNode>,
    /// The condition that stays in a branch or dispatch.
    condition: Option<BlockSpan>,
    /// A loop's kind, with its condition stored in the new tree.
    kind: Option<LoopKind>,
    /// The new child bodies so far.
    bodies: Vec<ControlBody>,
    /// The new edges of each switch case so far.
    edges: Vec<ControlRun<EdgeId>>,
    /// A switch's cases, once stored.
    cases: Option<ControlRun<SwitchCase>>,
    /// A try's handlers, once stored.
    handlers: Option<ControlRun<CatchHandler>>,
}

/// The start of wrapping a node.
enum Begun<'a> {
    Done(ControlNode),
    Wrapping(Wrapping<'a>),
}

/// What a finished frame hands to the one below it.
enum Returned {
    /// The frame is new.
    Start,
    Body(ControlBody),
    Node(ControlNode),
}

/// What a node's step asks for.
enum Step {
    /// Wrap this old child body next.
    Wrap(ControlBody),
    Done(ControlNode),
}

impl<'a, I: Predicated, E> Regionizer<'a, I, E> {
    /// The root node: a sequence of one statement is that statement.
    fn finish_root(&mut self, root: ControlNode) -> ControlNodeId {
        if let ControlNode::Sequence(body) = root
            && let &[only] = self.new.body(body)
        {
            return only;
        }
        self.new.push_node(root)
    }

    /// Wrap the node `root` and everything below it.
    ///
    /// The old tree nests as deeply as the graph did, so the rewrite keeps its
    /// own stack: a [`Frame::Body`] wraps the statements of one body, and a
    /// [`Frame::Node`] wraps the child bodies of one node. Each step stores
    /// what the recursive rewrite stored at the same point, in the same order,
    /// so a tree with no predicated run is rewritten into itself.
    fn wrap_root(&mut self, root: ControlNodeId) -> ControlNode {
        let mut stack: Vec<Frame<'a>> = Vec::new();
        let mut returned = match self.begin_node(root) {
            Begun::Done(node) => return node,
            Begun::Wrapping(wrapping) => {
                stack.push(Frame::Node(wrapping));
                Returned::Start
            }
        };
        while let Some(frame) = stack.last_mut() {
            let returned_now = core::mem::replace(&mut returned, Returned::Start);
            match frame {
                Frame::Body { remaining, nodes } => {
                    if let Returned::Node(node) = returned_now {
                        nodes.push(node);
                    }
                    let mut callee = None;
                    for &id in remaining.by_ref() {
                        match self.begin_node(id) {
                            Begun::Done(node) => nodes.push(node),
                            Begun::Wrapping(wrapping) => {
                                callee = Some(Frame::Node(wrapping));
                                break;
                            }
                        }
                    }
                    if let Some(callee) = callee {
                        stack.push(callee);
                    } else {
                        let body = self.new.seal(core::mem::take(nodes));
                        stack.pop();
                        returned = Returned::Body(body);
                    }
                }
                Frame::Node(wrapping) => {
                    let wrapped = match returned_now {
                        Returned::Body(body) => Some(body),
                        Returned::Start | Returned::Node(_) => None,
                    };
                    match self.step(wrapping, wrapped) {
                        Step::Wrap(body) => stack.push(Frame::Body {
                            remaining: self.old.body(body).iter(),
                            nodes: Vec::new(),
                        }),
                        Step::Done(node) => {
                            stack.pop();
                            returned = Returned::Node(node);
                        }
                    }
                }
            }
        }
        match returned {
            Returned::Node(node) => node,
            Returned::Start | Returned::Body(_) => {
                unreachable!("the root's node is the last frame to finish")
            }
        }
    }

    /// Start wrapping the node `id`: a node without child bodies is wrapped
    /// at once, and any other waits for its children.
    fn begin_node(&mut self, id: ControlNodeId) -> Begun<'a> {
        let old = self.old;
        let node = *old.node(id);
        let mut wrapping = Wrapping {
            node,
            children: old.child_bodies(id),
            prefix: Vec::new(),
            condition: None,
            kind: None,
            bodies: Vec::new(),
            edges: Vec::new(),
            cases: None,
            handlers: None,
        };
        match node {
            ControlNode::Block(span) => Begun::Done(self.wrap_block_runs(span)),
            ControlNode::Return(span) => Begun::Done(self.wrap_return_runs(span)),
            leaf @ (ControlNode::Break { .. }
            | ControlNode::Continue { .. }
            | ControlNode::Goto { .. }) => Begun::Done(leaf),
            ControlNode::IfThenElse { condition, .. } | ControlNode::Switch { condition, .. } => {
                let (prefix, condition) = self.split_header_runs(condition);
                wrapping.prefix = prefix;
                wrapping.condition = Some(condition);
                Begun::Wrapping(wrapping)
            }
            ControlNode::Loop { kind, .. } => {
                wrapping.kind = Some(self.copy_loop_kind(kind));
                Begun::Wrapping(wrapping)
            }
            ControlNode::Sequence(_)
            | ControlNode::Label { .. }
            | ControlNode::TryCatch { .. }
            | ControlNode::Guarded { .. } => Begun::Wrapping(wrapping),
        }
    }

    /// Take the wrapped child body `wrapped`, if the last child was wrapped,
    /// and name the next child body of the node to wrap, or finish the node.
    fn step(&mut self, wrapping: &mut Wrapping<'a>, wrapped: Option<ControlBody>) -> Step {
        if let Some(body) = wrapped {
            wrapping.bodies.push(body);
        }
        self.store_before_child(wrapping);
        match wrapping.children.next() {
            Some(child) => Step::Wrap(child),
            None => Step::Done(self.finish_node(wrapping)),
        }
    }

    /// Store what the node keeps besides its bodies, in the order the
    /// recursive rewrite did: a switch case's edges before the case's body,
    /// the cases before the default body, and the handlers before `finally`.
    fn store_before_child(&mut self, wrapping: &mut Wrapping<'a>) {
        let old = self.old;
        let next = wrapping.bodies.len();
        match wrapping.node {
            ControlNode::Switch { cases, .. } => {
                let cases = old.cases(cases);
                if let Some(case) = cases.get(next) {
                    let edges = self.new.push_edges(old.edges(case.edges).to_vec());
                    wrapping.edges.push(edges);
                } else if next == cases.len() {
                    let wrapped: Vec<SwitchCase> = cases
                        .iter()
                        .zip(&wrapping.edges)
                        .zip(&wrapping.bodies)
                        .map(|((case, &edges), &body)| SwitchCase {
                            block: case.block,
                            edges,
                            body,
                        })
                        .collect();
                    wrapping.cases = Some(self.new.push_cases(wrapped));
                }
            }
            ControlNode::TryCatch { handlers, .. } => {
                let handlers = old.handlers(handlers);
                if next == handlers.len() + 1 {
                    let wrapped: Vec<CatchHandler> = handlers
                        .iter()
                        .zip(&wrapping.bodies[1..])
                        .map(|(handler, &body)| CatchHandler { body, ..*handler })
                        .collect();
                    wrapping.handlers = Some(self.new.push_handlers(wrapped));
                }
            }
            _ => {}
        }
    }

    /// The node with its child bodies wrapped.
    fn finish_node(&mut self, wrapping: &mut Wrapping<'a>) -> ControlNode {
        let bodies = &wrapping.bodies;
        let first = bodies[0];
        let last = bodies[bodies.len() - 1];
        match wrapping.node {
            ControlNode::Sequence(_) => ControlNode::Sequence(first),
            ControlNode::IfThenElse { .. } => {
                let node = ControlNode::IfThenElse {
                    condition: wrapping.condition.expect("a branch keeps its condition"),
                    then_body: first,
                    else_body: last,
                };
                self.with_prefix(core::mem::take(&mut wrapping.prefix), node)
            }
            ControlNode::Loop { header, .. } => ControlNode::Loop {
                header,
                kind: wrapping.kind.expect("a loop keeps its kind"),
                body: first,
            },
            ControlNode::Switch { default_edge, .. } => {
                let node = ControlNode::Switch {
                    condition: wrapping.condition.expect("a dispatch keeps its condition"),
                    cases: wrapping
                        .cases
                        .expect("the cases are stored before the default"),
                    default_body: last,
                    default_edge,
                };
                self.with_prefix(core::mem::take(&mut wrapping.prefix), node)
            }
            ControlNode::Label { block, .. } => ControlNode::Label { block, body: first },
            ControlNode::TryCatch { .. } => ControlNode::TryCatch {
                try_body: first,
                handlers: wrapping
                    .handlers
                    .expect("the handlers are stored before finally"),
                finally_body: last,
            },
            ControlNode::Guarded {
                predicate,
                when_true,
                ..
            } => ControlNode::Guarded {
                predicate,
                when_true,
                body: first,
            },
            ControlNode::Block(_)
            | ControlNode::Return(_)
            | ControlNode::Break { .. }
            | ControlNode::Continue { .. }
            | ControlNode::Goto { .. } => {
                unreachable!("a node without child bodies is wrapped at once")
            }
        }
    }

    /// Re-store a loop's condition spans in the new tree.
    fn copy_loop_kind(&mut self, kind: LoopKind) -> LoopKind {
        match kind {
            LoopKind::While {
                condition,
                exit_on_true,
            } => LoopKind::While {
                condition: self
                    .new
                    .push_spans(self.old.spans(condition).iter().copied()),
                exit_on_true,
            },
            other @ (LoopKind::Endless | LoopKind::DoWhile { .. }) => other,
        }
    }

    /// Group the instructions of `span` into maximal runs sharing one
    /// predicate.
    fn predicate_runs(&self, span: BlockSpan) -> Vec<PredicateRun<I>> {
        let instructions = span.instructions(self.cfg);
        let mut runs: Vec<PredicateRun<I>> = Vec::new();
        for (index, instruction) in span.range().zip(instructions) {
            let predicate = instruction.predicate();
            match runs.last_mut() {
                Some(run) if run.predicate == predicate => {
                    run.span = BlockSpan::new(span.block(), run.span.start(), index + 1);
                }
                _ => runs.push(PredicateRun {
                    predicate,
                    span: BlockSpan::new(span.block(), index, index + 1),
                }),
            }
        }
        runs
    }

    /// Wrap a segment in [`ControlNode::Guarded`] when its run is predicated.
    fn guard_segment(&mut self, run: &PredicateRun<I>, segment: ControlNode) -> ControlNode {
        match run.predicate {
            Some((_, when_true)) => ControlNode::Guarded {
                predicate: ProgramPoint {
                    block: run.span.block(),
                    inst_idx: run.span.start(),
                },
                when_true,
                body: self.new.seal(alloc::vec![segment]),
            },
            None => segment,
        }
    }

    /// Collapse a single segment; wrap several segments in a sequence.
    fn sequence_or_single(&mut self, segments: Vec<ControlNode>) -> ControlNode {
        match <[ControlNode; 1]>::try_from(segments) {
            Ok([only]) => only,
            Err(segments) => ControlNode::Sequence(self.new.seal(segments)),
        }
    }

    /// Split a block's instructions into guarded segments.
    fn wrap_block_runs(&mut self, span: BlockSpan) -> ControlNode {
        let segments = self
            .predicate_runs(span)
            .iter()
            .map(|run| self.guard_segment(run, ControlNode::Block(run.span)))
            .collect();
        self.sequence_or_single(segments)
    }

    /// Split a return block's instructions into guarded segments; the final
    /// run keeps its `Return` semantics (a predicated final run is a
    /// conditional return, e.g. ARM `bxeq lr`).
    fn wrap_return_runs(&mut self, span: BlockSpan) -> ControlNode {
        let mut runs = self.predicate_runs(span);
        let last = runs.pop();
        let mut segments: Vec<ControlNode> = runs
            .iter()
            .map(|run| self.guard_segment(run, ControlNode::Block(run.span)))
            .collect();
        if let Some(run) = last {
            segments.push(self.guard_segment(&run, ControlNode::Return(run.span)));
        }
        self.sequence_or_single(segments)
    }

    /// Split a branch/dispatch header's predicated PREFIX runs into guarded
    /// segments to hoist before the node, and return the span that stays. The
    /// final run — which contains the branch/dispatch instruction itself —
    /// always stays in place, and headers with no predicated prefix pass
    /// through untouched.
    fn split_header_runs(&mut self, condition: BlockSpan) -> (Vec<ControlNode>, BlockSpan) {
        let mut runs = self.predicate_runs(condition);
        let Some(last) = runs.pop() else {
            return (Vec::new(), condition);
        };
        if runs.iter().all(|run| run.predicate.is_none()) {
            // Nothing to regionize: the original instruction list stays.
            return (Vec::new(), condition);
        }
        let prefix = runs
            .iter()
            .map(|run| self.guard_segment(run, ControlNode::Block(run.span)))
            .collect();
        (prefix, last.span)
    }

    /// Hoist `prefix` segments before `node`, or return `node` unchanged when
    /// there is nothing to hoist.
    fn with_prefix(&mut self, mut prefix: Vec<ControlNode>, node: ControlNode) -> ControlNode {
        if prefix.is_empty() {
            return node;
        }
        prefix.push(node);
        ControlNode::Sequence(self.new.seal(prefix))
    }
}
