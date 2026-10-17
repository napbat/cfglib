//! Whether control can reach the end of a node.

extern crate alloc;
use alloc::vec::Vec;

use super::super::node::{ControlBody, ControlNode, ControlNodeId};
use super::super::tree::Tables;

/// Which nodes of the arena control can never run to the end of.
///
/// The structurer asks after every construct it closes, and the answer reads
/// the last statement of the arms, which read theirs in turn. Asked afresh
/// each time, that walks the whole last-statement spine under the construct,
/// once per level of a deep nesting. A node's children have lower handles than
/// the node, so the answer for each node is settled once, in handle order,
/// from the answers of its children.
#[derive(Debug, Default)]
pub(super) struct Completion {
    /// The answer for each node below `never.len()`.
    never: Vec<bool>,
}

impl Completion {
    /// Whether control can never reach the end of `node`: every path through
    /// it ends in a `break`, `continue`, or return. `node` is a node whose
    /// child bodies are stored in `tables`; it may not be stored yet.
    ///
    /// The answer is conservative. A `goto` does not count: its label may sit
    /// later in the same construct (a pad's code that another arm of the same
    /// handler jumps to), and control then runs on to the end of it. A loop,
    /// which may run on or be left through a `break` of its own, and a node
    /// with a body that is empty or ends in plain code, can complete. So can
    /// a try: a handler body may end in dead code of its extent after the
    /// flow the handler really has, so only the walk that builds the try can
    /// say how its handlers end.
    pub(super) fn never_completes(&mut self, tables: &Tables, node: ControlNode) -> bool {
        while self.never.len() < tables.node_count() {
            let stored = *tables.node(ControlNodeId::from_index(self.never.len()));
            let answer = self.answer(tables, stored);
            self.never.push(answer);
        }
        self.answer(tables, node)
    }

    /// The answer for `node`, given the answers for the nodes below it.
    fn answer(&self, tables: &Tables, node: ControlNode) -> bool {
        let body_never_completes = |body: ControlBody| {
            tables
                .body(body)
                .last()
                .is_some_and(|&last| self.never[last.index()])
        };
        match node {
            ControlNode::Break { .. } | ControlNode::Continue { .. } | ControlNode::Return(_) => {
                true
            }
            ControlNode::Sequence(body)
            | ControlNode::Label { body, .. }
            | ControlNode::Guarded { body, .. } => body_never_completes(body),
            ControlNode::IfThenElse {
                then_body,
                else_body,
                ..
            } => body_never_completes(then_body) && body_never_completes(else_body),
            ControlNode::Switch {
                cases,
                default_body,
                ..
            } => {
                body_never_completes(default_body)
                    && tables
                        .cases(cases)
                        .iter()
                        .all(|case| body_never_completes(case.body))
            }
            ControlNode::Block(_)
            | ControlNode::Goto { .. }
            | ControlNode::Loop { .. }
            | ControlNode::TryCatch { .. } => false,
        }
    }
}
