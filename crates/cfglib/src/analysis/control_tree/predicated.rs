//! Predicated-region wrapping over a structured tree.

extern crate alloc;
use alloc::vec::Vec;

use super::node::{BlockSpan, CatchHandler, ControlBody, ControlNode, ControlNodeId, SwitchCase};
use super::report::LiftReport;
use super::tree::{ControlTree, Tables};
use crate::cfg::Cfg;
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
        let root = regionizer.wrap_node(tree.root());
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

impl<I: Predicated, E> Regionizer<'_, I, E> {
    /// The root node: a sequence of one statement is that statement.
    fn finish_root(&mut self, root: ControlNode) -> ControlNodeId {
        if let ControlNode::Sequence(body) = root
            && let &[only] = self.new.body(body)
        {
            return only;
        }
        self.new.push_node(root)
    }

    fn wrap_body(&mut self, body: ControlBody) -> ControlBody {
        let nodes: Vec<ControlNode> = self
            .old
            .body(body)
            .iter()
            .map(|&id| self.wrap_node(id))
            .collect();
        self.new.seal(nodes)
    }

    fn wrap_node(&mut self, id: ControlNodeId) -> ControlNode {
        let old = self.old;
        match *old.node(id) {
            ControlNode::Block(span) => self.wrap_block_runs(span),
            ControlNode::Return(span) => self.wrap_return_runs(span),
            ControlNode::Sequence(body) => ControlNode::Sequence(self.wrap_body(body)),
            ControlNode::IfThenElse {
                condition,
                then_body,
                else_body,
            } => {
                let (prefix, condition) = self.split_header_runs(condition);
                let node = ControlNode::IfThenElse {
                    condition,
                    then_body: self.wrap_body(then_body),
                    else_body: self.wrap_body(else_body),
                };
                self.with_prefix(prefix, node)
            }
            ControlNode::Loop { header, kind, body } => ControlNode::Loop {
                header,
                kind: self.copy_loop_kind(kind),
                body: self.wrap_body(body),
            },
            ControlNode::Switch {
                condition,
                cases,
                default_body,
                default_edge,
            } => {
                let (prefix, condition) = self.split_header_runs(condition);
                let wrapped: Vec<SwitchCase> = old
                    .cases(cases)
                    .iter()
                    .map(|case| SwitchCase {
                        block: case.block,
                        edges: self.new.push_edges(old.edges(case.edges).to_vec()),
                        body: self.wrap_body(case.body),
                    })
                    .collect();
                let node = ControlNode::Switch {
                    condition,
                    cases: self.new.push_cases(wrapped),
                    default_body: self.wrap_body(default_body),
                    default_edge,
                };
                self.with_prefix(prefix, node)
            }
            ControlNode::Label { block, body } => ControlNode::Label {
                block,
                body: self.wrap_body(body),
            },
            ControlNode::TryCatch {
                try_body,
                handlers,
                finally_body,
            } => {
                let try_body = self.wrap_body(try_body);
                let wrapped: Vec<CatchHandler> = old
                    .handlers(handlers)
                    .iter()
                    .map(|handler| CatchHandler {
                        body: self.wrap_body(handler.body),
                        ..*handler
                    })
                    .collect();
                ControlNode::TryCatch {
                    try_body,
                    handlers: self.new.push_handlers(wrapped),
                    finally_body: self.wrap_body(finally_body),
                }
            }
            ControlNode::Guarded {
                predicate,
                when_true,
                body,
            } => ControlNode::Guarded {
                predicate,
                when_true,
                body: self.wrap_body(body),
            },
            leaf @ (ControlNode::Break { .. }
            | ControlNode::Continue { .. }
            | ControlNode::Goto { .. }) => leaf,
        }
    }

    /// Re-store a loop's condition spans in the new tree.
    fn copy_loop_kind(&mut self, kind: super::node::LoopKind) -> super::node::LoopKind {
        match kind {
            super::node::LoopKind::While {
                condition,
                exit_on_true,
            } => super::node::LoopKind::While {
                condition: self
                    .new
                    .push_spans(self.old.spans(condition).iter().copied()),
                exit_on_true,
            },
            other @ (super::node::LoopKind::Endless | super::node::LoopKind::DoWhile { .. }) => {
                other
            }
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
