//! Exact goto-label placement over a structured tree.
//!
//! Labels are applied after the whole walk so only blocks an emitted goto
//! actually targets are wrapped — proactive labeling would force consumers
//! into unstructured fallbacks for transfers that resolved structurally.

extern crate alloc;
use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use super::super::node::{ControlBody, ControlNode, ControlNodeId};
use super::Structurer;
use crate::block::BlockId;

/// The block whose emission this node anchors, when it has one.
const fn anchor(node: &ControlNode) -> Option<BlockId> {
    match node {
        ControlNode::Block(span) | ControlNode::Return(span) => Some(span.block()),
        ControlNode::IfThenElse { condition, .. } | ControlNode::Switch { condition, .. } => {
            Some(condition.block())
        }
        ControlNode::Loop { header, .. } => Some(*header),
        _ => None,
    }
}

/// One step of the walk over the tree that places labels.
enum Step {
    /// Visit the statement of `body` at `position`, and the ones after it.
    Statement { body: ControlBody, position: usize },
    /// Wrap the statement of `body` at `position` in the label of `block`.
    Wrap {
        body: ControlBody,
        position: usize,
        block: BlockId,
    },
}

impl<I, E> Structurer<'_, I, E> {
    /// Wrap the emission of every block in `pending` with its label, removing
    /// each target as it is anchored. Targets that remain afterwards have no
    /// structured anchor — dangling gotos the caller reports.
    ///
    /// A wrapped node is replaced in its parent's body by a
    /// [`ControlNode::Label`] whose body holds just that node. The label is
    /// then renumbered next to the node it wraps, so children keep lower
    /// handles than their parents; no other handle changes.
    pub(super) fn apply_labels(&mut self, body: ControlBody, pending: &mut BTreeSet<BlockId>) {
        let first_label = self.tables.node_count();
        self.wrap_anchors(body, pending);
        self.tables.settle_labels(first_label);
    }

    /// Walk the statements of `root` and everything below them in order. A
    /// node claims its block before its children do, and is wrapped once
    /// they are done, so an inner anchor never takes a label from an outer
    /// one. Nesting is as deep as the input, so the walk keeps its own stack.
    fn wrap_anchors(&mut self, root: ControlBody, pending: &mut BTreeSet<BlockId>) {
        let mut steps = alloc::vec![Step::Statement {
            body: root,
            position: 0,
        }];
        while let Some(step) = steps.pop() {
            match step {
                Step::Wrap {
                    body,
                    position,
                    block,
                } => {
                    let id = self.tables.body(body)[position];
                    let wrapped = self.label_node(block, id);
                    self.tables.body_mut(body)[position] = wrapped;
                }
                Step::Statement { body, position } => {
                    if position == body.len() || pending.is_empty() {
                        continue;
                    }
                    let id = self.tables.body(body)[position];
                    let node = *self.tables.node(id);
                    let wrap = anchor(&node).filter(|block| pending.remove(block));
                    // Pushed last-to-first, so the node's children run first,
                    // then its label, then the statements after it.
                    steps.push(Step::Statement {
                        body,
                        position: position + 1,
                    });
                    if let Some(block) = wrap {
                        steps.push(Step::Wrap {
                            body,
                            position,
                            block,
                        });
                    }
                    let children: Vec<ControlBody> = self.tables.child_bodies(node).collect();
                    steps.extend(children.into_iter().rev().map(|child| Step::Statement {
                        body: child,
                        position: 0,
                    }));
                }
            }
        }
    }

    /// A new label node for `block` whose body is the single node `id`.
    fn label_node(&mut self, block: BlockId, id: ControlNodeId) -> ControlNodeId {
        let body = self.tables.seal_ids(&[id]);
        self.tables.push_node(ControlNode::Label { block, body })
    }
}
